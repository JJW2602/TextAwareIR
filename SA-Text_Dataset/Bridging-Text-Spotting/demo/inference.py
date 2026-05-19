#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
import torch
import json
from collections import deque
import atexit
import bisect
import string

from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

from adet.utils.visualizer import TextVisualizer
from adet.config import get_cfg

# Constants
WINDOW_NAME = "Text Detection"

def decode_rec(rec):
    voc = list(string.printable[:-6])
    s = ''
    for c in rec:
        c = int(c)
        if c < len(voc):
            s += voc[c]
        elif c == len(voc):
            return s
        else:
            s += u''
    return s

def convert_to_json_serializable(tensor):
    """Convert tensors to lists for JSON serialization."""
    if hasattr(tensor, 'cpu'):
        tensor = tensor.cpu()
    if hasattr(tensor, 'numpy'):
        tensor = tensor.numpy()
    if isinstance(tensor, np.ndarray):
        return tensor.tolist()
    return tensor

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    
    # Enable parallel
    cfg.parallel = args.parallel
    cfg.freeze()
    return cfg

def initialize_results():
    """Initialize the JSON structure for results."""
    return {
        "images": [],
        "annotations": []
    }

class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, visualize=True):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
            visualize (bool): whether to generate visualization output.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cfg = cfg
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.vis_text = cfg.MODEL.TRANSFORMER.ENABLED
        self.visualize = visualize

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output, None if visualize=False.
        """
        # Run inference
        predictions = self.predictor(image)
        
        # Process polygon data and correct bounding boxes
        if "instances" in predictions and self.vis_text:
            if hasattr(predictions["instances"], "polygons"):
                # Get polygon control points
                ctrl_pnts = predictions["instances"].polygons.cpu().numpy()
                
                bboxes = []
                for ctrl_pnt in ctrl_pnts:
                    # Reshape to pairs of (x,y) coordinates
                    points = ctrl_pnt.reshape(-1, 2)
                    
                    # Convert to integer type
                    points = np.array(points, dtype=np.int32)
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(points)
                    
                    # Convert to [x0, y0, x1, y1] format
                    bboxes.append([x, y, x + w, y + h])
                
                # Convert list to tensor and add to instances - overwrite existing boxes
                device = predictions["instances"].polygons.device
                predictions["instances"].boxes = torch.tensor(bboxes, device=device)
        
        # Skip visualization if not required
        if not self.visualize:
            return predictions, None
        
        # Visualization
        # Convert image from OpenCV BGR format to Matplotlib RGB format
        vis_image = image[:, :, ::-1]
        if self.vis_text:
            visualizer = TextVisualizer(vis_image, self.metadata, instance_mode=self.instance_mode, cfg=self.cfg)
        else:
            visualizer = Visualizer(vis_image, self.metadata, instance_mode=self.instance_mode)

        if "bases" in predictions:
            self.vis_bases(predictions["bases"])
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device))
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def vis_bases(self, bases):
        basis_colors = [[2, 200, 255], [107, 220, 255], [30, 200, 255], [60, 220, 255]]
        bases = bases[0].squeeze()
        bases = (bases / 8).tanh().cpu().numpy()
        num_bases = len(bases)
        fig, axes = plt.subplots(nrows=num_bases // 2, ncols=2)
        for i, basis in enumerate(bases):
            basis = (basis + 1) / 2
            basis = basis / basis.max()
            basis_viz = np.zeros((basis.shape[0], basis.shape[1], 3), dtype=np.uint8)
            basis_viz[:, :, 0] = basis_colors[i][0]
            basis_viz[:, :, 1] = basis_colors[i][1]
            basis_viz[:, :, 2] = np.uint8(basis * 255)
            basis_viz = cv2.cvtColor(basis_viz, cv2.COLOR_HSV2RGB)
            axes[i // 2][i % 2].imshow(basis_viz)
        plt.show()
        
        
# Define the helper function for processing results in async mode
def process_result(info, predictions, results, current_annotation_id, demo, output_dir, no_visualization):
    """Process detection results and update the results dictionary."""
    # Add to results
    results["images"].append({
        "file_name": info["file_name"],
        "width": info["width"],
        "height": info["height"],
        "instances": len(predictions["instances"])
    })
    
    # Track how many annotations we add
    added_annotations = 0
    
    # Process instances
    if "instances" in predictions:
        instances = predictions["instances"]
        
        # Get data
        scores = convert_to_json_serializable(instances.scores)
        boxes = convert_to_json_serializable(instances.boxes)
        
        # Get polygons if available
        polygons = []
        if hasattr(instances, 'polygons'):
            polygons = convert_to_json_serializable(instances.polygons)
        
        # Process each detection
        for i in range(len(scores)):
            # Convert box to COCO format
            box = boxes[i]
            coco_box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            
            # Create annotation
            annotation = {
                "id": current_annotation_id + i,
                "file_name": info["file_name"],
                "bbox": coco_box,
                "score": float(scores[i])
            }
            
            # Add polygon if available
            if polygons and i < len(polygons):
                polygon = polygons[i]
                points = []
                
                for j in range(0, len(polygon), 2):
                    if j+1 < len(polygon):
                        points.append([float(polygon[j]), float(polygon[j+1])])
                
                annotation["polygon"] = points
            
            results["annotations"].append(annotation)
            added_annotations += 1
        
        # Create and save visualization if needed
        if not no_visualization:
            # We need to rerun the image through the visualizer
            # Read the image again
            img = read_image(info["path"], format="BGR")
            
            # Run visualization only (inference already done)
            _, vis_output = demo.run_on_image(img)
            if vis_output is not None:
                out_filename = os.path.join(output_dir, info["file_name"])
                vis_output.save(out_filename)
    
    return added_annotations


def get_parser():
    parser = argparse.ArgumentParser(description="Text Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input", 
        nargs="+", 
        help="A list of space separated input images or a directory of images"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations and detection results.",
        required=True
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing with multiple GPUs if available",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip visualization and only output detection results",
    )
    parser.add_argument(
        "--use-polygon",
        action="store_true",
        help="Use polygon format for detection results",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=0,
        help="Buffer size for parallel processing. If 0, use default size."
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # Create the visualization demo with visualization flag
    demo = VisualizationDemo(
        cfg, 
        parallel=args.parallel,
        visualize=not args.no_visualization
    )

    # Initialize results dictionary
    results = initialize_results()
    annotation_id = 1

    # Check if output directory exists, create if it doesn't
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Prepare input file list
    input_files = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    if os.path.isdir(args.input[0]):
        for fname in os.listdir(args.input[0]):
            # Only include files with image extensions
            if any(fname.lower().endswith(ext) for ext in image_extensions):
                input_files.append(os.path.join(args.input[0], fname))
        logger.info(f"Found {len(input_files)} images in directory {args.input[0]}")
    elif len(args.input) == 1:
        # For glob patterns, we can also filter by extension
        all_files = glob.glob(os.path.expanduser(args.input[0]))
        input_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]
        logger.info(f"Found {len(input_files)} images matching pattern {args.input[0]}")
    else:
        # For explicitly specified files, also filter by extension
        input_files = [f for f in args.input if any(f.lower().endswith(ext) for ext in image_extensions)]
        logger.info(f"Found {len(input_files)} valid image files from {len(args.input)} specified files")
    
    assert input_files, "No valid image files found"

    

    # Process images based on whether we're using parallel inference
    if args.parallel:
        # Asynchronous processing with buffer
        buffer_size = args.buffer_size if args.buffer_size > 0 else demo.predictor.default_buffer_size
        
        # Create a buffer for images and their metadata
        image_data = deque()
        
        # Process all images
        for idx, path in enumerate(tqdm.tqdm(input_files, desc="Processing images")):
            try:
                # Read the image
                img = read_image(path, format="BGR")
                height, width = img.shape[:2]
                
                # Store image metadata
                image_info = {
                    "path": path,
                    "file_name": os.path.basename(path),
                    "width": width,
                    "height": height,
                }
                
                # Add to buffer and queue for processing
                image_data.append(image_info)
                demo.predictor.put(img)
                
                # Process results when buffer is full or we're at the end
                if len(image_data) >= buffer_size or idx == len(input_files) - 1:
                    for _ in range(len(image_data)):
                        # Get image info and prediction result
                        info = image_data.popleft()
                        prediction = demo.predictor.get()
                        
                        # Process result and update annotation_id
                        added = process_result(
                            info, prediction, results, annotation_id, 
                            demo, args.output, args.no_visualization
                        )
                        annotation_id += added
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                continue
    else:
        # Synchronous processing
        for path in tqdm.tqdm(input_files, desc="Processing images"):
            try:
                # Read the image
                img = read_image(path, format="BGR")
                height, width = img.shape[:2]
                
                # Create image info
                image_info = {
                    "path": path,
                    "file_name": os.path.basename(path),
                    "width": width,
                    "height": height,
                }
                
                # Process the image
                start_time = time.time()
                predictions, vis_output = demo.run_on_image(img)
                
                # Add to results
                results["images"].append({
                    "file_name": image_info["file_name"],
                    "width": width,
                    "height": height,
                    "instances": len(predictions["instances"])
                })
                
                # Process instances
                if "instances" in predictions:
                    instances = predictions["instances"]
                    
                    # Get data
                    scores = convert_to_json_serializable(instances.scores)
                    boxes = convert_to_json_serializable(instances.boxes)
                    recs = convert_to_json_serializable(instances.recs)
                    
                    # Get polygons if available
                    polygons = []
                    if hasattr(instances, 'polygons') and args.use_polygon:
                        polygons = convert_to_json_serializable(instances.polygons)
                    
                    # Process each detection
                    for i in range(len(scores)):
                        # Convert box to COCO format
                        # box = boxes[i]
                        # coco_box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                        
                        # Create annotation
                        annotation = {
                            "id": annotation_id,
                            "file_name": image_info["file_name"],
                            # "bbox": coco_box,
                            "bbox": boxes[i],
                            "score": float(scores[i]),
                            "rec": decode_rec(recs[i])
                        }
                        
                        # Add polygon if available
                        if polygons and i < len(polygons) and args.use_polygon:
                            polygon = polygons[i]
                            points = []
                            
                            for j in range(0, len(polygon), 2):
                                if j+1 < len(polygon):
                                    points.append([float(polygon[j]), float(polygon[j+1])])
                            
                            annotation["polygon"] = points
                        
                        results["annotations"].append(annotation)
                        annotation_id += 1
                
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time
                    )
                )
                
                # Save visualization if enabled
                if not args.no_visualization and vis_output is not None:
                    out_filename = os.path.join(args.output, image_info["file_name"])
                    vis_output.save(out_filename)
                    
            except:
                results["images"].append({
                    "file_name": os.path.basename(path),
                    "width": width,
                    "height": height,
                    "instances": 0
                })
                continue

    # Save the detection results to JSON
    output_json_path = os.path.join(args.output, "text_detection_results.json")
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_json_path}")