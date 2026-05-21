#!/usr/bin/env python3
"""
Script to transform dataset.json between original and simplified nested formats.
Features robust matching of crops based on content rather than just IDs.
Allows conditional inclusion of metadata like blur_category at the image level during transformation.
"""
import json
import os
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Transform dataset.json between formats')
    parser.add_argument('--input', type=str, required=True, help='Path to the input dataset JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output dataset JSON file')
    parser.add_argument('--mode', type=str, choices=['transform', 'revert'], default='transform',
                        help='Operation mode: transform (to nested format) or revert (to original format)')
    parser.add_argument('--dataset_type', type=str, choices=['full', 'restoration'],
                        required=False,
                        help='Specify the type of dataset being transformed (full or restoration) to control inclusion of metadata like blur_category. Required for transform mode.')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to the original unfiltered dataset to retrieve crop region information (for revert mode)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed matching information')
    args = parser.parse_args()

    if args.mode == 'transform' and args.dataset_type is None:
        parser.error("--dataset_type is required when using --mode transform")

    return args

def transform_dataset(input_path, output_path, dataset_type):
    """
    Transform original format to nested format, placing metadata at the image level.

    Args:
        input_path (str): Path to the input dataset JSON file (original format).
        output_path (str): Path to save the output dataset JSON file (nested format).
        dataset_type (str): 'full' or 'restoration'. Controls inclusion of metadata.
    """
    print(f"Loading dataset from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_path}: {e}")
        return

    entries = data.get('entries', [])
    if not entries:
        print("Warning: No 'entries' found in the input file.")

    transformed_data = {}
    processed_entries = 0

    print(f"Transforming data to nested format (Dataset Type: {dataset_type})...")
    for entry in entries:
        if not all(k in entry for k in ['original_image', 'text_instances']):
            print(f"Warning: Skipping entry due to missing keys (original_image or text_instances): {entry.get('original_image', 'Unknown Image')}")
            continue

        original_image = entry['original_image']
        if isinstance(original_image, str) and original_image.startswith('train/'):
            original_image = original_image[6:]
        image_id = os.path.splitext(original_image)[0] # e.g., "sa_2737_crop_1"

        # --- MODIFICATION: Initialize image_id entry if new ---
        if image_id not in transformed_data:
            transformed_data[image_id] = {}
        # --- END MODIFICATION ---

        # --- MODIFICATION: Force crop_idx to "0" ---
        crop_idx = "0"
        # --- END MODIFICATION ---

        # Extract text instances
        text_instances = []
        if isinstance(entry.get('text_instances'), list):
            for instance in entry['text_instances']:
                if not isinstance(instance, dict) or not all(k in instance for k in ['bbox', 'text']):
                     print(f"Warning: Skipping instance in image {image_id} due to missing keys or wrong format: {instance}")
                     continue
                instance_data = {'bbox': instance['bbox'], 'text': instance['text']}
                polygon_data = instance.get('polygon')
                if polygon_data is not None:
                    instance_data['polygon'] = polygon_data
                text_instances.append(instance_data)
        else:
             print(f"Warning: 'text_instances' is not a list or missing in image {image_id}. Skipping instances for this entry.")

        # --- MODIFICATION: Conditionally add blur_category at the image_id level ---
        if dataset_type == 'full':
            blur_cat = entry.get('blur_category')
            if blur_cat is not None:
                # Add or overwrite blur_category at the image_id level
                transformed_data[image_id]['blur_category'] = blur_cat
            else:
                 original_crop_id_for_warning = entry.get('crop_id', 'Unknown Crop ID')
                 print(f"Warning: dataset_type is 'full' but 'blur_category' missing for entry with original_image '{entry['original_image']}' (Crop ID: {original_crop_id_for_warning}). Not adding blur_category.")
        # --- END MODIFICATION ---

        # --- MODIFICATION: Add text instances under the crop_idx key ---
        # Check if crop_idx already exists (shouldn't happen if input is clean, but handles duplicates)
        if crop_idx in transformed_data[image_id]:
            print(f"Warning: Duplicate entry found for image '{image_id}' (using crop index '{crop_idx}'). Overwriting previous text instances for this crop index.")

        # Store text instances within a dictionary under the crop_idx key
        transformed_data[image_id][crop_idx] = {"text_instances": text_instances}
        # --- END MODIFICATION ---

        processed_entries += 1

    # Save the transformed dataset
    print(f"Saving transformed data to: {output_path}")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Error: Could not write output file to {output_path}. Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")
        return

    print(f"Transformation complete. Transformed dataset saved to {output_path}")
    print(f"Original entry count processed: {processed_entries}/{len(entries)}")
    print(f"Transformed image count: {len(transformed_data)}")

    # Calculate total number of crops (indexed entries like "0")
    total_crops = sum(
        sum(1 for key in crops_dict if key != 'blur_category') # Count keys that are not 'blur_category'
        for crops_dict in transformed_data.values()
    )
    print(f"Total crop index entries (e.g., '0') in transformed data: {total_crops}")

    # --- MODIFICATION: Adjust instance counting for new structure ---
    total_instances = sum(
        len(crop_data.get("text_instances", []))
        for image_data in transformed_data.values() # image_data is dict like {"blur_category": "...", "0": {"text_instances": [...]}}
        for key, crop_data in image_data.items()   # Iterate through items in image_data
        if key != 'blur_category' and isinstance(crop_data, dict) # Process only crop index dicts
    )
    # --- END MODIFICATION ---
    print(f"Total text instances in transformed data: {total_instances}")


# --- calculate_crop_similarity and normalize_image_id remain the same ---

def calculate_crop_similarity(crop1_instances, crop2_instances):
    """
    Calculate similarity score between two crops based on their text instances

    Args:
        crop1_instances: List of text instances from first crop
        crop2_instances: List of text instances from second crop

    Returns:
        Similarity score (higher is more similar)
    """
    # If lengths are very different, they're probably not the same crop
    len_diff = abs(len(crop1_instances) - len(crop2_instances))
    if len_diff > max(3, min(len(crop1_instances), len(crop2_instances)) // 2):
        return 0

    # Count matching text instances
    matching_texts = 0
    bbox_similarity_sum = 0

    # For each instance in crop1, find best matching instance in crop2
    for inst1 in crop1_instances:
        # Ensure instance is a dictionary and has required keys
        if not isinstance(inst1, dict) or 'text' not in inst1 or 'bbox' not in inst1:
            continue
        text1 = inst1['text']
        bbox1 = inst1['bbox']
        # Basic validation for bbox format
        if not isinstance(bbox1, list) or len(bbox1) != 4:
            continue


        best_match_score = 0
        for inst2 in crop2_instances:
             # Ensure instance is a dictionary and has required keys
            if not isinstance(inst2, dict) or 'text' not in inst2 or 'bbox' not in inst2:
                continue
            text2 = inst2['text']
            bbox2 = inst2['bbox']
            # Basic validation for bbox format
            if not isinstance(bbox2, list) or len(bbox2) != 4:
                continue

            # Text must match
            if text1 != text2:
                continue

            # Calculate bbox similarity (0 to 1, higher is better)
            # Simple approach: Check if top-left coordinates are close
            try:
                x_diff = abs(bbox1[0] - bbox2[0])
                y_diff = abs(bbox1[1] - bbox2[1])
            except (TypeError, IndexError): # Handle cases where bbox elements aren't numbers or index is out of bounds
                continue


            # Normalize differences based on typical bbox size
            typical_size = 50  # Assume typical bbox dimension is around 50 pixels
            x_sim = max(0, 1 - (x_diff / typical_size))
            y_sim = max(0, 1 - (y_diff / typical_size))

            bbox_sim = (x_sim + y_sim) / 2
            match_score = bbox_sim

            if match_score > best_match_score:
                best_match_score = match_score

        if best_match_score > 0.5:  # Consider a match if similarity > 0.5
            matching_texts += 1
            bbox_similarity_sum += best_match_score

    # Calculate final similarity score
    if matching_texts == 0:
        return 0

    # Weight based on number of matching texts and their bbox similarity
    avg_bbox_similarity = bbox_similarity_sum / matching_texts if matching_texts > 0 else 0

    # Calculate percentage of matching texts (relative to smaller crop)
    # Avoid division by zero if one crop has 0 instances (though len_diff check might catch this)
    min_len = min(len(crop1_instances), len(crop2_instances))
    match_ratio = matching_texts / min_len if min_len > 0 else 0

    # Final score: combination of match ratio and bbox similarity
    similarity = 0.7 * match_ratio + 0.3 * avg_bbox_similarity

    return similarity

def normalize_image_id(image_id):
    """Normalize image ID to handle different formats"""
    # Ensure input is a string
    if not isinstance(image_id, str):
        return "" # Or raise an error, depending on desired behavior

    # Remove train/ prefix if present
    if image_id.startswith('train/'):
        image_id = image_id[6:]

    # Remove file extension if present
    image_id = os.path.splitext(image_id)[0] # Use os.path.splitext for robustness

    return image_id


def revert_dataset(input_path, output_path, reference_path=None, verbose=False):
    """
    Revert from nested format back to original format with robust crop matching.
    Handles metadata (like blur_category) stored at the image level in the nested format.

    Args:
        input_path: Path to the nested format dataset (filtered)
        output_path: Path to save the output dataset in original format
        reference_path: Path to original unfiltered dataset to retrieve crop region information
        verbose: Whether to print detailed matching information
    """
    print(f"Loading nested dataset from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            nested_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_path}: {e}")
        return

    entries = []
    total_crops_processed = 0
    crops_with_reference_match = 0
    images_without_reference = 0
    reference_load_failed = False

    # Index reference dataset by image ID if provided
    reference_by_image = defaultdict(list)
    if reference_path:
        print(f"Loading reference dataset from {reference_path}...")
        try:
            with open(reference_path, 'r', encoding='utf-8') as f:
                reference_data = json.load(f)
            ref_entries = reference_data.get('entries', [])
            if not ref_entries:
                 print("Warning: Reference file loaded but contains no 'entries'.")
            for entry in ref_entries:
                if not all(k in entry for k in ['original_image', 'crop_id', 'crop_region', 'text_instances']):
                    print(f"Warning: Skipping reference entry due to missing keys: {entry.get('crop_id', 'Unknown ID')}")
                    continue
                image_id_ref = normalize_image_id(entry['original_image'])
                reference_by_image[image_id_ref].append(entry)
            print(f"Loaded reference data for {len(reference_by_image)} unique image IDs.")
        except FileNotFoundError:
            print(f"Warning: Reference file not found at {reference_path}. Proceeding without reference data.")
            reference_path = None; reference_load_failed = True
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to decode JSON from reference file {reference_path}: {e}. Proceeding without reference data.")
            reference_path = None; reference_load_failed = True
        except Exception as e:
            print(f"Warning: An unexpected error occurred while loading reference dataset: {e}. Proceeding without reference data.")
            reference_path = None; reference_load_failed = True
    else:
        print("No reference dataset path provided. Crop regions and detailed instance info will use defaults.")

    print("Reverting data to original format...")
    # Process each image entry in the nested data
    for image_id, image_data in nested_data.items(): # image_id is e.g., "sa_2737_crop_1"
                                                     # image_data is e.g., {"blur_category": "...", "0": {...}}
        normalized_id = normalize_image_id(image_id) # Still useful for matching reference keys

        # Reconstruct original_image filename
        original_image_filename = f"{image_id}.jpg"
        if not original_image_filename.startswith('train/'):
            original_image_filename = f"train/{original_image_filename}"

        # Get reference crops for this image
        reference_crops = reference_by_image.get(normalized_id, [])
        if reference_path and not reference_load_failed and not reference_crops:
             images_without_reference += 1
             if verbose:
                 print(f"Verbose: No reference crops found for image ID '{normalized_id}' (Original key: '{image_id}').")

        # --- MODIFICATION: Get blur_category from the image level in nested data ---
        blur_category_from_input = image_data.get('blur_category')
        # --- END MODIFICATION ---

        # Process each crop index entry within the image_data
        if not isinstance(image_data, dict):
             print(f"Warning: Expected dictionary of data for image '{image_id}', got {type(image_data)}. Skipping image.")
             continue

        for key, crop_data in image_data.items():
            # Skip metadata keys like 'blur_category' at this level
            if key == 'blur_category':
                continue

            # Assume other keys are crop indices (like "0")
            crop_idx = key
            if not isinstance(crop_data, dict):
                print(f"Warning: Expected dictionary for crop data at index '{crop_idx}' for image '{image_id}', got {type(crop_data)}. Skipping crop.")
                continue

            total_crops_processed += 1

            # Extract text instances for this crop index
            text_instances = crop_data.get("text_instances", [])

            # Determine crop_id, region, blur_category (prioritize reference)
            reconstructed_crop_id = f"{image_id}_crop_{crop_idx}" # Default construction
            crop_region = [0, 0, 512, 512]
            blur_category = blur_category_from_input if blur_category_from_input is not None else "Not blurry" # Use input blur if available, else default
            matched_reference_entry = None

            # Find matching reference crop
            best_match_ref_entry = None
            best_similarity = 0
            if reference_path and not reference_load_failed and reference_crops:
                if not isinstance(text_instances, list):
                    print(f"Warning: text_instances for crop {image_id}/{crop_idx} is not a list. Cannot perform matching.")
                else:
                    for ref_entry in reference_crops:
                        ref_text_instances = ref_entry.get('text_instances')
                        if not isinstance(ref_text_instances, list): continue

                        ref_instances_for_comparison = [{'text': i['text'], 'bbox': i['bbox']} for i in ref_text_instances if isinstance(i, dict) and 'text' in i and 'bbox' in i]

                        similarity = calculate_crop_similarity(text_instances, ref_instances_for_comparison)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_ref_entry = ref_entry

                    similarity_threshold = 0.5
                    if best_similarity > similarity_threshold:
                        crops_with_reference_match += 1
                        matched_reference_entry = best_match_ref_entry
                        # --- Prioritize reference values ---
                        reconstructed_crop_id = matched_reference_entry['crop_id']
                        crop_region = matched_reference_entry['crop_region']
                        blur_category = matched_reference_entry.get('blur_category', blur_category) # Get from ref, fallback to value derived from input/default
                        # --- End Prioritization ---
                        if verbose: print(f"Verbose: Matched crop {image_id}/{crop_idx} to reference {reconstructed_crop_id} with similarity {best_similarity:.2f}")
                    elif verbose:
                        print(f"Verbose: No good reference match for {image_id}/{crop_idx} (best similarity: {best_similarity:.2f}). Using defaults/input values.")
            # If no reference or no match, reconstructed_crop_id, crop_region, blur_category retain their default/input-derived values

            # Create the output entry
            entry = {
                "original_image": original_image_filename,
                "crop_id": reconstructed_crop_id,
                "crop_region": crop_region,
                "blur_category": blur_category, # Add the determined blur category
                "text_instances": []
            }

            # Add text instances, enriching with reference data if matched
            if not isinstance(text_instances, list):
                 print(f"Warning: text_instances for crop {reconstructed_crop_id} is not a list. Skipping instance processing.")
            else:
                for i, instance in enumerate(text_instances):
                    if not isinstance(instance, dict) or 'text' not in instance or 'bbox' not in instance:
                        print(f"Warning: Skipping invalid instance in crop {reconstructed_crop_id}: {instance}")
                        continue

                    score, overlaps, polygon_from_ref = 1.0, False, None
                    instance_id_base = normalize_image_id(reconstructed_crop_id)
                    instance_id = f"{instance_id_base}_{i}"

                    matching_ref_instance = None
                    if matched_reference_entry:
                        ref_instances = matched_reference_entry.get('text_instances', [])
                        if isinstance(ref_instances, list):
                            text, bbox = instance['text'], instance['bbox']
                            best_inst_match_ref, best_inst_similarity = None, 0
                            for ref_instance in ref_instances:
                                if not isinstance(ref_instance, dict): continue
                                if ref_instance.get('text') == text:
                                    ref_bbox = ref_instance.get('bbox')
                                    similarity = 0.1 # Default if bboxes invalid
                                    if isinstance(ref_bbox, list) and len(ref_bbox) == 4 and isinstance(bbox, list) and len(bbox) == 4:
                                        try:
                                            x_diff = abs(ref_bbox[0] - bbox[0]); y_diff = abs(ref_bbox[1] - bbox[1])
                                            similarity = 1 / (1 + x_diff + y_diff)
                                        except (TypeError, IndexError): similarity = 0
                                    if similarity > best_inst_similarity:
                                        best_inst_similarity = similarity; best_inst_match_ref = ref_instance
                            if best_inst_similarity > 0.01: matching_ref_instance = best_inst_match_ref

                    if matching_ref_instance:
                        score = matching_ref_instance.get("score", score)
                        overlaps = matching_ref_instance.get("overlaps", overlaps)
                        instance_id = matching_ref_instance.get("id", instance_id)
                        polygon_from_ref = matching_ref_instance.get("polygon")

                    text_instance_output = {
                        "bbox": instance["bbox"], "text": instance["text"],
                        "score": score, "overlaps": overlaps, "id": instance_id
                    }
                    final_polygon = polygon_from_ref if polygon_from_ref is not None else instance.get('polygon')
                    if final_polygon is not None:
                         text_instance_output["polygon"] = final_polygon
                    entry["text_instances"].append(text_instance_output)

            entries.append(entry)

    # Save the reverted dataset
    output_data = {"entries": entries}
    print(f"Saving reverted data to: {output_path}")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Error: Could not write output file to {output_path}. Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")
        return

    print(f"Reversion complete. Reverted dataset saved to {output_path}")
    print(f"Total images processed from nested format: {len(nested_data)}")
    print(f"Total entries created in reverted format: {len(entries)}")
    print(f"Total crop index entries processed: {total_crops_processed}") # Renamed stat for clarity

    if reference_path and not reference_load_failed:
        match_percentage = (crops_with_reference_match / total_crops_processed * 100) if total_crops_processed > 0 else 0
        print(f"Reference matching: {crops_with_reference_match}/{total_crops_processed} crops matched ({match_percentage:.1f}%)")
        if images_without_reference > 0: print(f"Images present in input but not found in reference: {images_without_reference}")
    elif reference_load_failed: print("Reference data loading failed. Metadata restoration relies on input or defaults.")
    else: print("No reference data used. Metadata restoration relies on input or defaults.")

    total_instances_output = sum(len(entry.get("text_instances", [])) for entry in entries)
    print(f"Total text instances in reverted output: {total_instances_output}")


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return

    if args.mode == 'transform':
        transform_dataset(args.input, args.output, args.dataset_type)
    else:  # args.mode == 'revert'
        if not args.reference:
             print("Warning: Revert mode typically benefits from a --reference file to restore exact original metadata. Proceeding without it.")
        elif not os.path.exists(args.reference):
             print(f"Warning: Reference file specified but not found at {args.reference}. Proceeding without reference data.")
             args.reference = None

        revert_dataset(args.input, args.output, args.reference, args.verbose)

if __name__ == '__main__':
    main()