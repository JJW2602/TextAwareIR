import json
import os
import logging
import re
import pandas as pd
from collections import defaultdict
import copy
from tqdm import tqdm

from .utils import read_json, write_json, ensure_dir, read_text_list

# --- Calculate Overlap (Copied from cropping.py for modularity) ---
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

# --- Filter 0: Duplicate Detections ---
def filter_duplicate_detections(input_json_path, output_json_path, config, iou_threshold=0.9):
    """
    Filters out highly overlapping detections within the same image (crop).
    Keeps the detection with the highest confidence score among duplicates.
    Operates on the output of Bridge Stage 2.
    """
    logging.info(f"Filtering duplicate detections from: {input_json_path} (IoU Threshold: {iou_threshold})")
    data = read_json(input_json_path)
    if not data or "annotations" not in data or "images" not in data:
        logging.error("Invalid JSON data for duplicate detection filtering.")
        return None

    annotations_in = data["annotations"]
    annotations_by_image = defaultdict(list)
    for ann in annotations_in:
        # Ensure bbox is valid before adding
        if isinstance(ann.get("bbox"), list) and len(ann["bbox"]) == 4:
            annotations_by_image[ann['file_name']].append(ann)
        else:
            logging.warning(f"Skipping annotation ID {ann.get('id')} in {ann.get('file_name')} due to invalid bbox format: {ann.get('bbox')}")


    final_annotations = []
    duplicate_removed_count = 0
    total_processed = 0

    image_filenames = list(annotations_by_image.keys())
    for filename in tqdm(image_filenames, desc="Filtering duplicates"):
        img_annotations = annotations_by_image[filename]
        n = len(img_annotations)
        if n <= 1: # No duplicates possible if 0 or 1 annotation
            final_annotations.extend(img_annotations)
            total_processed += n
            continue

        # Sort by score descending to prioritize higher confidence boxes
        img_annotations.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        kept_indices = set(range(n)) # Start assuming all are kept

        for i in range(n):
            if i not in kept_indices: continue # Skip if already marked for removal

            bbox1 = img_annotations[i]["bbox"] # [x1, y1, x2, y2] format from Bridge Stage 2

            for j in range(i + 1, n):
                if j not in kept_indices: continue # Skip if already marked for removal

                bbox2 = img_annotations[j]["bbox"]
                overlap = calculate_overlap(bbox1, bbox2)

                if overlap > iou_threshold:
                    # High overlap found. Since we sorted by score,
                    # annotation 'i' has higher or equal score than 'j'.
                    # Mark 'j' for removal.
                    kept_indices.remove(j)
                    duplicate_removed_count += 1

        # Add annotations that were kept
        kept_this_image = []
        for idx in kept_indices:
            kept_this_image.append(img_annotations[idx])

        final_annotations.extend(kept_this_image)
        total_processed += n


    # Update image instance counts (optional but good practice)
    final_annotations_by_image = defaultdict(list)
    for ann in final_annotations:
        final_annotations_by_image[ann['file_name']].append(ann)

    updated_images = []
    for img in data["images"]:
        img_copy = copy.deepcopy(img)
        img_copy['instances'] = len(final_annotations_by_image.get(img['file_name'], []))
        # Only keep images that still have annotations after filtering
        if img_copy['instances'] > 0:
            updated_images.append(img_copy)


    output_data = {
        "images": updated_images,
        "annotations": final_annotations
    }
    write_json(output_data, output_json_path)
    logging.info(f"Duplicate detection filtering complete. Kept {len(final_annotations)} annotations (removed {duplicate_removed_count}). Output: {output_json_path}")
    return output_json_path

# --- Filter 1: Empty/Invalid VLM Results ---
def filter_empty_vlm(input_json_path, output_json_path, config):
    """Filters annotations with empty or invalid VLM text."""
    data = read_json(input_json_path)
    if not data or "annotations" not in data:
        logging.error("Invalid JSON data for VLM filtering.") # Keep Error
        return None

    annotations_in = data["annotations"]
    annotations_out = []
    filtered_count = 0
    images_dict = {img["file_name"]: img for img in data.get("images", [])}

    image_reference_patterns = [
        r"the image", r"this image", r"the photo", r"this photo", r"the picture",
        r"this picture", r"the screenshot", r"cannot read", r"can't read",
        r"can not read", r"doesn't contain", r"does not contain", r"no text",
        r"not contain any text", r"no visible text", r"no readable text",
        r"no discernible text", r"no legible text", r"unable to identify",
        r"unable to determine", r"unable to read"
    ]
    image_ref_regex = re.compile('|'.join(image_reference_patterns), re.IGNORECASE)

    for ann in annotations_in:
        vlm_result = ann.get("VLM", "")
        is_invalid = False
        if not vlm_result or vlm_result.isspace(): is_invalid = True
        elif vlm_result in ["()", "''", '""', "[]", "{}", "//", "--", ".."]: is_invalid = True
        elif (vlm_result.startswith("'") and vlm_result.endswith("'") and
              (len(vlm_result) == 2 or vlm_result[1:-1].isspace())): is_invalid = True
        elif (vlm_result.startswith('"') and vlm_result.endswith('"') and
              (len(vlm_result) == 2 or vlm_result[1:-1].isspace())): is_invalid = True
        elif image_ref_regex.search(vlm_result): is_invalid = True

        if is_invalid:
            filtered_count += 1
            if ann["file_name"] in images_dict and "instances" in images_dict[ann["file_name"]]:
                images_dict[ann["file_name"]]["instances"] = max(0, images_dict[ann["file_name"]].get("instances", 1) - 1)
        else:
            annotations_out.append(ann)

    updated_images = list(images_dict.values())
    output_data = {"images": updated_images, "annotations": annotations_out}
    write_json(output_data, output_json_path)
    logging.info(f"VLM filtering complete. Kept {len(annotations_out)} annotations (removed {filtered_count}). Output: {output_json_path}")
    return output_json_path


# --- Filter 2: Compare & Merge VLMs ---
def compare_and_merge_vlms(vlm1_json_path, vlm2_json_path, output_json_path, config):
    """Merges results from two VLM JSONs."""
    data1 = read_json(vlm1_json_path)
    data2 = read_json(vlm2_json_path)
    vlm1_name = config['vlm1_name']
    vlm2_name = config['vlm2_name']

    if not data1 or "annotations" not in data1 or "images" not in data1:
        logging.error(f"Invalid data in {vlm1_json_path}") # Keep Error
        return None
    if not data2 or "annotations" not in data2:
        logging.error(f"Invalid data in {vlm2_json_path}") # Keep Error
        return None

    annotations2_map = {ann['id']: ann for ann in data2['annotations']}
    merged_annotations = []
    processed_ids2 = set()
    missing_in_2 = 0
    missing_in_1 = 0

    for ann1 in data1['annotations']:
        ann1_id = ann1['id']
        ann2 = annotations2_map.get(ann1_id)
        merged_ann = ann1.copy()
        merged_ann[vlm1_name] = ann1.get('VLM')
        if ann2:
            merged_ann[vlm2_name] = ann2.get('VLM')
            processed_ids2.add(ann1_id)
            if 'has_text' in ann2 and 'has_text' not in merged_ann:
                merged_ann['has_text'] = ann2['has_text']
        else:
            merged_ann[vlm2_name] = None
            missing_in_2 += 1
        merged_ann.pop('VLM', None)
        merged_annotations.append(merged_ann)

    for ann2_id, ann2 in annotations2_map.items():
        if ann2_id not in processed_ids2:
            missing_in_1 += 1
            merged_ann = ann2.copy()
            merged_ann[vlm1_name] = None
            merged_ann[vlm2_name] = ann2.get('VLM')
            merged_ann.pop('VLM', None)
            if 'file_name' not in merged_ann: merged_ann['file_name'] = ann2.get('file_name', 'Unknown')
            if 'bbox' not in merged_ann: merged_ann['bbox'] = ann2.get('bbox', [])
            merged_annotations.append(merged_ann)

    # Keep Warnings about missing annotations
    if missing_in_1 > 0:
        logging.warning(f"{missing_in_1} annotations found only in {os.path.basename(vlm2_json_path)}")
    if missing_in_2 > 0:
        logging.warning(f"{missing_in_2} annotations found only in {os.path.basename(vlm1_json_path)}")

    output_data = {"images": data1["images"], "annotations": merged_annotations}
    write_json(output_data, output_json_path)
    logging.info(f"VLM comparison merge complete. Total annotations: {len(merged_annotations)}. Output: {output_json_path}")
    return output_json_path


# --- Filter 3: Identify Fully Agreed Images ---
def identify_agreed_images(combined_json_path, output_list_path, config):
    """Identifies images where both VLMs agree on all text instances. Minimal logging."""
    data = read_json(combined_json_path)
    vlm1_name = config['vlm1_name']
    vlm2_name = config['vlm2_name']
    if not data or "annotations" not in data:
        logging.error("Invalid combined JSON data.") # Keep Error
        return None

    annotations_by_image = defaultdict(list)
    for ann in data["annotations"]:
        annotations_by_image[ann['file_name']].append(ann)

    agreed_image_filenames = []
    total_text_images = 0
    disagreement_images = 0
    for file_name, annotations in annotations_by_image.items():
        if not annotations: continue
        total_text_images += 1
        all_agree = True
        for ann in annotations:
            vlm1_res = ann.get(vlm1_name)
            vlm2_res = ann.get(vlm2_name)
            if vlm1_res != vlm2_res:
                all_agree = False
                break
        if all_agree:
            agreed_image_filenames.append(file_name)
        else:
            disagreement_images += 1

    try:
        ensure_dir(os.path.dirname(output_list_path))
        with open(output_list_path, 'w', encoding='utf-8') as f:
            for fname in sorted(agreed_image_filenames):
                f.write(f"{fname}\n")
        # Keep Summary Info
        logging.info(f"Identified {len(agreed_image_filenames)} images with full agreement out of {total_text_images} images with text.")
        logging.info(f"({disagreement_images} images had at least one disagreement). List saved to: {output_list_path}")
        return output_list_path
    except Exception as e:
        logging.error(f"Failed to write agreed image list to {output_list_path}: {e}") # Keep Error
        return None


# --- Filter 4: Extract Agreed Annotations ---
def extract_agreed_annotations(combined_json_path, agreed_list_path, output_json_path, config):
    """Extracts annotations only for images listed in the agreed list. Minimal logging."""
    # logging.info("Extracting annotations for fully agreed images...") # Included in time_step
    data = read_json(combined_json_path)
    agreed_filenames = read_text_list(agreed_list_path)
    vlm1_name = config['vlm1_name']

    if not data or "annotations" not in data or "images" not in data:
        logging.error("Invalid combined JSON data for extraction.") # Keep Error
        return None
    if agreed_filenames is None:
        logging.error("Agreed image list not found or empty.") # Keep Error
        return None

    agreed_filenames_set = set(agreed_filenames)
    final_annotations = []
    final_images = []
    annotations_per_image = defaultdict(list)

    for ann in data["annotations"]:
        if ann["file_name"] in agreed_filenames_set:
            new_ann = ann.copy()
            agreed_text = new_ann.get(vlm1_name)
            new_ann.pop(config['vlm1_name'], None)
            new_ann.pop(config['vlm2_name'], None)
            new_ann['VLM'] = agreed_text
            annotations_per_image[ann['file_name']].append(new_ann)

    for img in data["images"]:
        if img["file_name"] in agreed_filenames_set:
            img_copy = copy.deepcopy(img)
            instance_count = len(annotations_per_image.get(img['file_name'], []))
            if instance_count > 0:
                img_copy['instances'] = instance_count
                final_images.append(img_copy)

    final_kept_image_names = {img['file_name'] for img in final_images}
    final_annotations = [ann for img_anns in annotations_per_image.values() for ann in img_anns if ann['file_name'] in final_kept_image_names]

    output_data = {"images": final_images, "annotations": final_annotations}
    write_json(output_data, output_json_path)
    # Keep Summary Info
    logging.info(f"Extracted {len(final_annotations)} annotations from {len(final_images)} agreed images. Output: {output_json_path}")
    return output_json_path


# --- Filter 5a: Tag with Blur Category ---
def tag_with_blur(agreed_json_path, blur_csv_path, output_json_path, config):
    """Adds a 'blur_category' field to image entries. Minimal logging."""
    data = read_json(agreed_json_path)
    blur_vlm_name = config.get('blur_vlm_name', 'Qwen')
    expected_blur_col_name = f"{blur_vlm_name}_Blur"
    expected_filename_col_name = 'Filename'
    default_tag = "Error: No Blur Data"

    if not data or "annotations" not in data or "images" not in data:
        logging.error("Invalid agreed JSON data for blur tagging.") # Keep Error
        return None

    try:
        blur_df = pd.read_csv(blur_csv_path)
        actual_columns = blur_df.columns.tolist()
        actual_filename_col = None
        actual_blur_col = None
        for col in actual_columns:
            if col.lower() == expected_filename_col_name.lower(): actual_filename_col = col
            elif col.lower() == expected_blur_col_name.lower(): actual_blur_col = col
        if not actual_filename_col or not actual_blur_col:
            missing_cols_desc = []
            if not actual_filename_col: missing_cols_desc.append(f"'{expected_filename_col_name}' (case-insensitive)")
            if not actual_blur_col: missing_cols_desc.append(f"'{expected_blur_col_name}' (case-insensitive, derived from blur_vlm_name='{blur_vlm_name}')")
            logging.error(f"Blur CSV ({os.path.basename(blur_csv_path)}) missing required columns: {', '.join(missing_cols_desc)}") # Keep Error
            logging.error(f"Actual columns found were: {actual_columns}") # Keep Error
            return None
        blur_map = pd.Series(blur_df[actual_blur_col].values, index=blur_df[actual_filename_col])

        if blur_map.index.has_duplicates:
            blur_map = blur_map[~blur_map.index.duplicated(keep='first')]
        blur_map = blur_map.to_dict()

    except FileNotFoundError: logging.error(f"Blur CSV file not found: {blur_csv_path}"); return None # Keep Error
    except pd.errors.EmptyDataError: logging.error(f"Blur CSV file is empty: {blur_csv_path}"); return None # Keep Error
    except Exception as e: logging.error(f"Error reading or processing blur CSV {blur_csv_path}: {e}"); return None # Keep Error

    tagged_images = []
    missing_blur_data_count = 0
    for img_dict in data["images"]:
        new_img_dict = copy.deepcopy(img_dict)
        crop_filename = img_dict["file_name"]
        blur_category = blur_map.get(crop_filename, default_tag)
        if blur_category == default_tag:
            missing_blur_data_count += 1
        if pd.isna(blur_category):
            blur_category = default_tag
        new_img_dict['blur_category'] = blur_category
        tagged_images.append(new_img_dict)

    # Keep Warning about missing data
    if missing_blur_data_count > 0:
        logging.warning(f"Could not find blur data for {missing_blur_data_count} images in the CSV map. Tagged as '{default_tag}'.")

    output_data = {"images": tagged_images, "annotations": data["annotations"]}
    write_json(output_data, output_json_path)
    # Keep Summary Info
    logging.info(f"Blur tagging complete. Added 'blur_category' to {len(tagged_images)} image entries. Output: {output_json_path}")
    return output_json_path


# --- Filter 5b: Filter Tagged Annotations by Blur ---
def filter_tagged_by_blur(tagged_json_path, output_json_path, config):
    """Filters images/annotations based on image 'blur_category' tag. Minimal logging."""
    data = read_json(tagged_json_path)
    blur_keep_category = config.get('blur_keep_category', 'Not blurry')
    if not data or "annotations" not in data or "images" not in data:
        logging.error("Invalid tagged JSON data for blur filtering.") # Keep Error
        return None

    kept_images = []
    kept_image_filenames = set()
    original_image_count = len(data["images"])

    for img in data["images"]:
        if img.get("blur_category") == blur_keep_category:
            kept_images.append(img)
            kept_image_filenames.add(img["file_name"])

    kept_annotations = []
    original_annotation_count = len(data["annotations"])
    for ann in data["annotations"]:
        if ann["file_name"] in kept_image_filenames:
            kept_annotations.append(ann)

    final_kept_images = []
    final_kept_image_filenames = set()
    for img in kept_images:
        img_copy = copy.deepcopy(img)
        instance_count = sum(1 for ann in kept_annotations if ann['file_name'] == img['file_name'])
        if instance_count > 0:
            img_copy['instances'] = instance_count
            final_kept_images.append(img_copy)
            final_kept_image_filenames.add(img['file_name'])

    final_kept_annotations = [ann for ann in kept_annotations if ann['file_name'] in final_kept_image_filenames]
    output_data = {"images": final_kept_images, "annotations": final_kept_annotations}
    write_json(output_data, output_json_path)
    removed_image_count = original_image_count - len(final_kept_images)
    removed_annotation_count = original_annotation_count - len(final_kept_annotations)
    logging.info(f"Blur filtering complete. Kept {len(final_kept_images)} images (removed {removed_image_count}) and {len(final_kept_annotations)} annotations (removed {removed_annotation_count}) marked as '{blur_keep_category}'. Output: {output_json_path}")
    return output_json_path