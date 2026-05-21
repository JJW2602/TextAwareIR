import os
import json
import logging
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.cluster import DBSCAN

from .utils import read_json, ensure_dir


def calculate_overlap(region1, region2):
    """Calculate the IoU of two regions [x1, y1, x2, y2]."""
    x1 = max(region1[0], region2[0])
    y1 = max(region1[1], region2[1])
    x2 = min(region1[2], region2[2])
    y2 = min(region1[3], region2[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (region1[2] - region1[0]) * (region1[3] - region1[1])
    area2 = (region2[2] - region2[0]) * (region2[3] - region2[1])
    if area1 <= 0 or area2 <= 0: return 0.0
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def check_crop_validity(crop_region, instances):
    """Check if crop cuts any instance bbox. Returns True if valid."""
    for instance in instances:
        bbox_abs = instance.get("bbox")
        if not (isinstance(bbox_abs, list) and len(bbox_abs) == 4):
            continue
        intersects = not (
            bbox_abs[2] <= crop_region[0] or bbox_abs[0] >= crop_region[2] or
            bbox_abs[3] <= crop_region[1] or bbox_abs[1] >= crop_region[3]
        )
        fully_contained = (
            bbox_abs[0] >= crop_region[0] and bbox_abs[2] <= crop_region[2] and
            bbox_abs[1] >= crop_region[1] and bbox_abs[3] <= crop_region[3]
        )
        if intersects and not fully_contained:
            return False
    return True


def generate_sliding_window_crops(image_info, target_size, stride, max_instances):
    """Generates candidate crops using sliding window."""
    candidates = []
    width, height = image_info["width"], image_info["height"]
    instances = image_info["instances"]

    for y in range(0, height - target_size + 1, stride):
        for x in range(0, width - target_size + 1, stride):
            crop_region = [x, y, x + target_size, y + target_size]
            if check_crop_validity(crop_region, instances):
                count = 0
                for inst in instances:
                    bbox_abs = inst.get("bbox")
                    if not (isinstance(bbox_abs, list) and len(bbox_abs) == 4): continue
                    if (bbox_abs[0] >= crop_region[0] and bbox_abs[2] <= crop_region[2] and
                            bbox_abs[1] >= crop_region[1] and bbox_abs[3] <= crop_region[3]):
                        count += 1
                if 1 <= count <= max_instances:
                    candidates.append({"region": crop_region, "count": count})
    return candidates


def identify_overlapping_instances(instances, overlap_threshold=0.1):
    """Identify overlapping instances using IoU. Returns modified instances."""
    n = len(instances)
    for inst in instances: inst["overlaps"] = inst.get("overlaps", False)
    for i in range(n):
        for j in range(i + 1, n):
            bbox1 = instances[i].get("bbox")
            bbox2 = instances[j].get("bbox")
            if not (isinstance(bbox1, list) and len(bbox1) == 4 and isinstance(bbox2, list) and len(bbox2) == 4): continue
            overlap = calculate_overlap(bbox1, bbox2)
            if overlap > overlap_threshold:
                instances[i]["overlaps"] = True
                instances[j]["overlaps"] = True
    return instances


def generate_adaptive_crops(image_info, target_size, max_instances):
    """Generates crops adaptively based on text clusters, falls back to sliding window."""
    img_name = image_info.get('file_name', 'Unknown')
    width, height = image_info["width"], image_info["height"]
    instances = image_info["instances"]
    if not instances: return []

    instances = identify_overlapping_instances(instances, overlap_threshold=0.1)
    non_overlapping = [inst for inst in instances if not inst.get("overlaps", False)]

    if len(non_overlapping) < 1:
        return generate_sliding_window_crops(image_info, target_size, stride=target_size // 4, max_instances=max_instances)

    centroids = []
    for inst in non_overlapping:
        bbox = inst.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4): continue
        cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
        centroids.append([cx, cy])

    if not centroids:
        return generate_sliding_window_crops(image_info, target_size, stride=target_size // 4, max_instances=max_instances)

    X = np.array(centroids)
    if X.shape[0] == 1: labels = np.array([0])
    else:
        max_dim = max(width, height)
        if max_dim == 0:
            logging.warning(f"Image {img_name} has zero max dimension. Falling back to sliding window.") 
            return generate_sliding_window_crops(image_info, target_size, stride=target_size // 4, max_instances=max_instances)
        X_normalized = X / max_dim
        eps = 0.25 * target_size / max_dim
        try:
            clustering = DBSCAN(eps=eps, min_samples=1).fit(X_normalized)
            labels = clustering.labels_
        except Exception as dbscan_e:
            logging.error(f"DBSCAN failed for {img_name}: {dbscan_e}. Falling back to sliding window.")
            return generate_sliding_window_crops(image_info, target_size, stride=target_size // 4, max_instances=max_instances)

    clusters = {};
    for i, label in enumerate(labels):
        if label not in clusters: clusters[label] = []
        clusters[label].append(non_overlapping[i])

    candidates = []
    for label, cluster_instances in clusters.items():
        if label == -1: continue
        if len(cluster_instances) > max_instances:
            continue
        valid_cluster_instances = [inst for inst in cluster_instances if isinstance(inst.get("bbox"), list) and len(inst["bbox"]) == 4]
        if not valid_cluster_instances:
            continue
        min_x = min(inst["bbox"][0] for inst in valid_cluster_instances); min_y = min(inst["bbox"][1] for inst in valid_cluster_instances)
        max_x = max(inst["bbox"][2] for inst in valid_cluster_instances); max_y = max(inst["bbox"][3] for inst in valid_cluster_instances)
        center_x = (min_x + max_x) / 2; center_y = (min_y + max_y) / 2
        crop_left = max(0, int(center_x - target_size / 2)); crop_top = max(0, int(center_y - target_size / 2))
        if crop_left + target_size > width: crop_left = width - target_size
        if crop_top + target_size > height: crop_top = height - target_size
        crop_left = max(0, crop_left); crop_top = max(0, crop_top)
        crop_right = crop_left + target_size; crop_bottom = crop_top + target_size
        crop_region = [crop_left, crop_top, crop_right, crop_bottom]

        if check_crop_validity(crop_region, instances):
            count = 0
            for inst in instances:
                 bbox_abs = inst.get("bbox")
                 if not (isinstance(bbox_abs, list) and len(bbox_abs) == 4): continue
                 if (bbox_abs[0] >= crop_region[0] and bbox_abs[2] <= crop_region[2] and
                     bbox_abs[1] >= crop_region[1] and bbox_abs[3] <= crop_region[3]):
                     count += 1
            if 1 <= count <= max_instances:
                candidates.append({"region": crop_region, "count": count})

    if not candidates:
        return generate_sliding_window_crops(image_info, target_size, stride=target_size // 4, max_instances=max_instances)

    return candidates


def score_crop(crop, instances_in_crop, target_size):
    """Basic scoring: prefer more instances, penalize emptiness."""
    num_instances = len(instances_in_crop)
    if num_instances == 0: return 0.0
    score = min(1.0, num_instances / 10.0)
    if num_instances > 1:
        coords_x = []; coords_y = []
        for inst in instances_in_crop:
             bbox_abs = inst.get("bbox")
             if not (isinstance(bbox_abs, list) and len(bbox_abs) == 4): continue
             cx = (bbox_abs[0] + bbox_abs[2]) / 2; cy = (bbox_abs[1] + bbox_abs[3]) / 2
             coords_x.append(cx); coords_y.append(cy)
        if coords_x and coords_y:
             x_range = max(coords_x) - min(coords_x); y_range = max(coords_y) - min(coords_y)
             if target_size > 0:
                 dist_score = (x_range + y_range) / (target_size * 2)
                 score = score * 0.7 + dist_score * 0.3
             else: score = score * 0.7
    return score


def select_best_crops(candidates, instances, target_size, max_selections=3, diversity_threshold=0.3):
    """Selects diverse crops based on score, region overlap, AND instance uniqueness."""
    if not candidates: return []

    scored_candidates = []
    for cand in candidates:
        instances_in_region = []; instance_ids_in_region = set()
        for inst in instances:
            bbox_abs = inst.get("bbox"); inst_id = inst.get("id")
            if not (isinstance(bbox_abs, list) and len(bbox_abs) == 4 and inst_id is not None): continue
            if (bbox_abs[0] >= cand["region"][0] and bbox_abs[2] <= cand["region"][2] and
                    bbox_abs[1] >= cand["region"][1] and bbox_abs[3] <= cand["region"][3]):
                instances_in_region.append(inst); instance_ids_in_region.add(inst_id)
        if instances_in_region:
             cand["score"] = score_crop(cand, instances_in_region, target_size)
             cand["instance_ids"] = instance_ids_in_region
             scored_candidates.append(cand)

    scored_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    if not scored_candidates: return []

    selected = []; selected_instance_ids = set()
    top_candidate = scored_candidates[0]
    selected.append(top_candidate)
    if "instance_ids" in top_candidate: selected_instance_ids.update(top_candidate["instance_ids"])

    for candidate in scored_candidates[1:]:
        if len(selected) >= max_selections: break
        is_diverse_region = True
        for sel_crop in selected:
            overlap = calculate_overlap(candidate["region"], sel_crop["region"])
            if overlap > diversity_threshold:
                is_diverse_region = False
                break
        if not is_diverse_region: continue

        is_diverse_instances = True
        candidate_instance_ids = candidate.get("instance_ids", set())
        if not candidate_instance_ids:
             logging.warning(f"Candidate {candidate['region']} missing instance IDs during diversity check.") 
             continue
        if candidate_instance_ids.intersection(selected_instance_ids):
            is_diverse_instances = False

        if not is_diverse_instances: continue

        selected.append(candidate)
        selected_instance_ids.update(candidate_instance_ids)

    final_selected = []
    for crop in selected:
         final_selected.append({"region": crop["region"], "count": crop.get("count", 0)})
    return final_selected


def define_crop_regions(stage1_json_path, config):
    """
    Reads Stage 1 detections, defines crop regions using adaptive clustering
    (with fallback to sliding window), and returns definitions.
    """
    logging.info(f"Defining crop regions based on: {stage1_json_path}") 
    data = read_json(stage1_json_path)
    if not data or "images" not in data or "annotations" not in data:
        logging.error("Invalid Stage 1 JSON data.") 
        return []

    target_size = config['crop_size']
    max_crops = config['max_crops_per_image']
    max_instances_per_crop = 50

    image_annotations = {}
    for img in data["images"]:
        image_annotations[img["file_name"]] = {
            "width": img["width"], "height": img["height"], "instances": [], "file_name": img["file_name"]
        }
    for ann in data["annotations"]:
        if ann["file_name"] in image_annotations:
            if isinstance(ann.get("bbox"), list) and len(ann["bbox"]) == 4:
                 image_annotations[ann["file_name"]]["instances"].append(ann)

    crop_definitions = []
    image_keys = list(image_annotations.keys())
    for file_name in tqdm(image_keys, desc="Defining crops"):
        image_data = image_annotations[file_name]
        if image_data["width"] < target_size or image_data["height"] < target_size:
            continue
        if not image_data["instances"]: continue

        candidates = generate_adaptive_crops(image_data, target_size, max_instances=max_instances_per_crop)
        selected = select_best_crops(candidates, image_data["instances"], target_size, max_selections=max_crops)

        img_base_name = os.path.splitext(file_name)[0]
        for i, crop in enumerate(selected):
            crop_id = f"{img_base_name}_crop_{i}"
            crop_definitions.append({
                "original_image": file_name, "crop_id": crop_id, "crop_region": crop["region"]
            })

    logging.info(f"Defined {len(crop_definitions)} crop regions.") 
    return crop_definitions


def create_crop_images(crop_definitions, sa1b_image_dir, output_crop_dir, config):
    """Creates and saves crop image files based on definitions."""
    logging.info(f"Creating crop images in: {output_crop_dir}") 
    ensure_dir(output_crop_dir)
    crop_path_map = {}
    quality = config.get('jpeg_quality', 95)
    for definition in tqdm(crop_definitions, desc="Creating crop images"):
        original_image_path = os.path.join(sa1b_image_dir, definition["original_image"])
        crop_id = definition["crop_id"]; output_filename = f"{crop_id}.jpg"
        output_path = os.path.join(output_crop_dir, output_filename); crop_region = definition["crop_region"]
        try:
            with Image.open(original_image_path) as img:
                img_rgb = img.convert("RGB"); cropped_img = img_rgb.crop(crop_region)
                cropped_img.save(output_path, quality=quality); crop_path_map[crop_id] = output_path
        except FileNotFoundError:
            logging.warning(f"Original image not found, skipping crop: {original_image_path}") 
        except Exception as e:
            logging.error(f"Failed to create crop {output_filename} from {original_image_path}: {e}") 
    logging.info(f"Successfully created {len(crop_path_map)} crop images.") 
    return crop_path_map