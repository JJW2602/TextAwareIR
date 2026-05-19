import json
import os
import logging
from tqdm import tqdm
import copy
from collections import defaultdict

from .utils import read_json, write_json, ensure_dir


def format_final_dataset(input_json_path, output_json_path, config):
    """
    Formats the final filtered annotations into the desired dataset structure,
    adding a top-level 'blur_category' tag to each entry. Minimal logging.
    """
    data = read_json(input_json_path)
    if data is None:
        logging.error(f"Failed to read input JSON for formatting: {input_json_path}") 
        return None
    if "images" not in data or "annotations" not in data:
        if "images" in data and "annotations" in data:
            pass
        else:
            logging.error(f"Invalid JSON data structure in {input_json_path} for final formatting.") 
            return None

    crop_size = config['crop_size']
    dataset_entries = []

    annotations_by_image = defaultdict(list)
    for ann in data["annotations"]:
        annotations_by_image[ann["file_name"]].append(ann)
    images_map = {img["file_name"]: img for img in data["images"]}

    for img_dict in tqdm(data["images"], desc=f"Formatting {os.path.basename(output_json_path)}"):
        crop_filename = img_dict["file_name"]
        crop_annotations = annotations_by_image.get(crop_filename, [])
        blur_category = img_dict.get("blur_category", "Error: Tag Missing")

        entry = {
            "original_image": crop_filename,
            "crop_id": os.path.splitext(crop_filename)[0],
            "crop_region": [0, 0, crop_size, crop_size],
            "blur_category": blur_category,
            "text_instances": []
        }

        for instance in crop_annotations:
            instance_copy = copy.deepcopy(instance)
            instance_copy.pop("blur_category", None)

            text_instance = {
                "text": instance_copy.get("VLM", ""),
                "bbox": instance_copy.get("bbox", [0, 0, 0, 0]),
                "score": instance_copy.get("score", 0.0),
                "id": instance_copy.get("id", -1),
            }
            if "polygon" in instance_copy:
                text_instance["polygon"] = instance_copy["polygon"]
            entry["text_instances"].append(text_instance)

        dataset_entries.append(entry)

    final_output_data = {"entries": dataset_entries}
    write_json(final_output_data, output_json_path)
    logging.info(f"Final dataset formatting complete for {os.path.basename(output_json_path)}. Created {len(dataset_entries)} entries.")
    return output_json_path