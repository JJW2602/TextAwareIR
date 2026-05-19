import torch
from PIL import Image
import os
import json
import logging
import tqdm
import copy
import pandas as pd
import time
from collections import Counter
import re

# Attempt to import VLM libraries
try:
    from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info  # Assuming this is available
except ImportError as e:
    logging.warning(f"Could not import VLM libraries ({e}). VLM processing will fail if attempted.")
    AutoModelForCausalLM = None
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None
    process_vision_info = None

from .utils import read_json, write_json, ensure_dir, VLMProgressFilter


# --- Internal Helper: Crop Image ---
def _crop_tight(image, bbox):
    """Crops PIL image tightly based on bbox [x1, y1, x2, y2]."""
    try:
        width, height = image.size
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        if x1 >= x2 or y1 >= y2:
            return None
        return image.crop((x1, y1, x2, y2))
    except Exception as e:
        logging.error(f"Error during tight cropping with bbox {bbox}: {e}") 
        return None


# --- Internal Helper: OVIS Inference ---
def _ovis_two_stage_inference(model, text_tokenizer, visual_tokenizer, image_tensor):
    """Performs OVIS two-stage inference (text check + OCR). Minimal logging."""
    if not model or not text_tokenizer or not visual_tokenizer:
        logging.error("OVIS model/tokenizers not loaded.") 
        return "Error: Model Load", False
    text_check = "Identify whether there is any text in the image. Answer with only 'yes' or 'no'."
    query = f'<image>\n{text_check}'
    try:
        prompt, input_ids, pixel_values = model.preprocess_inputs(query, image_tensor, max_partition=9)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
        pixel_values = [pixel_values]
        with torch.inference_mode():
            gen_kwargs = dict(max_new_tokens=32, do_sample=False, eos_token_id=model.generation_config.eos_token_id, pad_token_id=text_tokenizer.pad_token_id, use_cache=True)
            output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            has_text_response = text_tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()
        has_text = "yes" in has_text_response
        if not has_text:
            return "", False
        text_extract_prompt = "Please recognize the text in the image. Return only the raw text without any formatting or explanations. Preserve original case, especially uppercase. Return an empty string ('') if no text is detected."
        query = f'<image>\n{text_extract_prompt}'
        prompt, input_ids, _ = model.preprocess_inputs(query, image_tensor, max_partition=9)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        with torch.inference_mode():
            gen_kwargs = dict(max_new_tokens=1024, do_sample=False, eos_token_id=model.generation_config.eos_token_id, pad_token_id=text_tokenizer.pad_token_id, use_cache=True)
            output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            extracted_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return extracted_text, True
    except Exception as e:
        logging.error(f"Error during OVIS inference: {e}") 
        return "Error: Model Inference", False


# --- Internal Helper: Qwen Inference ---
def _qwen_two_stage_inference(model, processor, image):
    """Performs Qwen two-stage inference (text check + OCR). Minimal logging."""
    if not model or not processor or not process_vision_info:
        logging.error("Qwen model/processor not loaded or qwen_vl_utils missing.") 
        return "Error: Model Load", False
    messages_check = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Identify whether there is any text in the image. Answer with only 'yes' or 'no'."}]}]
    try:
        text_check = processor.apply_chat_template(messages_check, tokenize=False, add_generation_prompt=True)
        image_inputs_check, _ = process_vision_info(messages_check)
        inputs_check = processor(text=[text_check], images=image_inputs_check, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids_check = model.generate(**inputs_check, max_new_tokens=32)
            generated_ids_trimmed_check = generated_ids_check[:, inputs_check.input_ids.shape[1]:]
            has_text_response = processor.batch_decode(generated_ids_trimmed_check, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().lower()
        has_text = "yes" in has_text_response
        if not has_text:
            return "", False
        text_extract_prompt = "Please recognize the text in the image. Return only the raw text without any formatting or explanations. Preserve original case, especially uppercase. Output the text exactly as shown in the image, preserving spacing. Return an empty string ('') if no text is detected."
        messages_extract = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_extract_prompt}]}]
        text_extract = processor.apply_chat_template(messages_extract, tokenize=False, add_generation_prompt=True)
        image_inputs_extract, _ = process_vision_info(messages_extract)
        inputs_extract = processor(text=[text_extract], images=image_inputs_extract, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids_extract = model.generate(**inputs_extract, max_new_tokens=1024)
            generated_ids_trimmed_extract = generated_ids_extract[:, inputs_extract.input_ids.shape[1]:]
            extracted_text = processor.batch_decode(generated_ids_trimmed_extract, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return extracted_text, True
    except Exception as e:
        logging.error(f"Error during Qwen inference: {e}") 
        return "Error: Model Inference", False


# --- Main Function: VLM Recognition ---
def run_vlm_recognition(model_name, annotations_json_path, crop_image_dir, output_json_path, config):
    """Runs VLM recognition on text instances. Minimal logging."""
    if not AutoModelForCausalLM:
        logging.error("Transformers library not available.")
        return None
    logging.info(f"Starting VLM Recognition using {model_name}...")
    data = read_json(annotations_json_path)
    if not data or "images" not in data or "annotations" not in data:
        logging.error("Invalid annotations JSON data for VLM recognition.") 
        return None

    model = None
    processor = None
    text_tokenizer = None
    visual_tokenizer = None
    try:
        if model_name.upper() == "OVIS":
            model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-8B", torch_dtype=torch.bfloat16, multimodal_max_length=32768, trust_remote_code=True).cuda()
            text_tokenizer = model.get_text_tokenizer()
            visual_tokenizer = model.get_visual_tokenizer()
            logging.info("OVIS model loaded.") 
        elif model_name.upper() == "QWEN":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            logging.info("Qwen model loaded.") 
        else:
            logging.error(f"Unsupported VLM model name: {model_name}") 
            return None
    except Exception as e:
        logging.error(f"Failed to load {model_name} model: {e}") 
        return None

    enriched_annotations = []
    image_cache = {}
    total_annotations = len(data['annotations'])
    processed_count = 0
    error_count = 0
    # Keep Start Info with count
    logging.info(f"Processing {total_annotations} annotations...")

    # Keep Progress Bar
    for annotation in tqdm.tqdm(data['annotations'], desc=f"Running {model_name} OCR"):
        processed_count += 1
        file_name = annotation['file_name']
        image_path = os.path.join(crop_image_dir, file_name)

        if file_name not in image_cache:
            try:
                img = Image.open(image_path).convert('RGB')
                image_cache[file_name] = img
            except FileNotFoundError:
                error_count += 1
                continue # Skip silently
            except Exception as e:
                logging.error(f"Failed load image {image_path}: {e}. Skipping ann {annotation.get('id', 'N/A')}.") 
                error_count += 1
                continue
        else:
            img = image_cache[file_name]

        bbox = annotation['bbox']
        tight_crop_img = _crop_tight(img, bbox)
        if tight_crop_img is None:
            error_count += 1
            continue # Skip silently

        vlm_text = f"Error: No {model_name} Result"
        has_text = False
        try:
            if model_name.upper() == "OVIS":
                vlm_text, has_text = _ovis_two_stage_inference(model, text_tokenizer, visual_tokenizer, [tight_crop_img])
            elif model_name.upper() == "QWEN":
                vlm_text, has_text = _qwen_two_stage_inference(model, processor, tight_crop_img)
        except Exception as e:
            logging.error(f"Inference failed for ann {annotation.get('id', 'N/A')}: {e}") 
            vlm_text = "Error: Inference Exception"
            has_text = False
            error_count += 1

        new_annotation = copy.deepcopy(annotation)
        new_annotation['VLM'] = vlm_text
        new_annotation['has_text'] = has_text
        enriched_annotations.append(new_annotation)

    output_data = {"images": data["images"], "annotations": enriched_annotations}
    write_json(output_data, output_json_path)
    logging.info(f"{model_name} recognition complete. Annotations processed: {processed_count}, Errors/Skipped: {error_count}. Output saved to: {output_json_path}")

    # Cleanup
    del model, processor, text_tokenizer, visual_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info(f"{model_name} model unloaded.") 
    return output_json_path


# --- Internal Helper: Qwen Blur Check ---
def _qwen_check_blurriness(model, processor, image, inverse_prompt_flag):
    """Internal function for blur/sharpness classification. Minimal logging."""
    if not model or not processor or not process_vision_info:
        logging.error("Qwen model/processor not loaded or qwen_vl_utils missing.") 
        return "Error: Model Load"
    prompt_text = ("""
        You are an image quality inspector. Classify the image into one of the following levels based on its sharpness:
        1. "Level 1" – The image is blurry and lacks clarity.
        2. "Level 2" – The image is somewhat clear but has minor blurriness.
        3. "Level 3" – The image is very clear and sharp.
        Please answer with exactly one of the three labels: "Level 1", "Level 2", or "Level 3".
        """)
    if inverse_prompt_flag:
        valid_categories = ["level 1", "level 2", "level 3"]
        category_map = {"level 1": "Not sharp", "level 2": "Slightly sharp", "level 3": "Very sharp"}
    else:
        valid_categories = ["not blurry", "slightly blurry", "very blurry"]
        category_map = {"not blurry": "Not blurry", "slightly blurry": "Slightly blurry", "very blurry": "Very blurry"}

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
            response_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        processed_response = response_text.strip().lower().rstrip('.').strip()
        if processed_response in valid_categories:
            return category_map.get(processed_response, "Error: Mapping Failed")
        else:
            # Reduce logging for unexpected responses
            if re.search(r"cannot determine|unable to classify|information is insufficient", processed_response, re.IGNORECASE):
                return "Error: Cannot Classify"
            elif re.search(r"text|content|read", processed_response, re.IGNORECASE):
                return "Error: Focused on Text"
            else:
                return "Error: Unknown Category"
    except Exception as e:
        logging.error(f"Error during Qwen blur inference: {e}") 
        return "Error: Model Inference"


# --- Blur Assessment ---
def run_blur_assessment(agreed_filenames, crop_image_dir, output_csv_path, config):
    """Runs blur assessment using Qwen ONLY on the provided list of crop filenames. Minimal logging."""
    if not Qwen2_5_VLForConditionalGeneration:
        logging.error("Transformers library not available.") 
        return None
    if not agreed_filenames:
        logging.warning("No agreed filenames provided for blur assessment. Skipping.") 
        return None

    blur_vlm_name = config.get('blur_vlm_name', 'Qwen')
    run_identifier = f"{blur_vlm_name}_Blur"
    logging.info(f"Starting Blur Assessment using {blur_vlm_name} on {len(agreed_filenames)} agreed crops...")
    inverse_prompt_flag = config.get('blur_inverse_prompt', False)

    model = None
    processor = None
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        logging.info("Qwen model loaded for blur assessment.") 
    except Exception as e:
        logging.error(f"Failed to load Qwen model for blur assessment: {e}") 
        return None

    start_time = time.time()
    current_run_results = {}
    processed_count = 0
    error_count = 0
    skipped_images = []

    for filename_key in tqdm.tqdm(agreed_filenames, desc=f"Assessing Blur ({run_identifier})"):
        image_path = os.path.join(crop_image_dir, filename_key)
        relative_path = filename_key
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')
            raw_model_category = _qwen_check_blurriness(model, processor, image, inverse_prompt_flag)
            final_category_for_csv = None
            if isinstance(raw_model_category, str) and "Error:" in raw_model_category:
                final_category_for_csv = raw_model_category
                error_count += 1
                skipped_images.append(relative_path)
            elif inverse_prompt_flag:
                mapping = {"Very sharp": "Not blurry", "Slightly sharp": "Slightly blurry", "Not sharp": "Very blurry"}
                final_category_for_csv = mapping.get(raw_model_category, raw_model_category)
                if final_category_for_csv == raw_model_category and raw_model_category not in mapping:
                    error_count += 1
                    skipped_images.append(relative_path)
                if "Error:" not in final_category_for_csv:
                    processed_count += 1
            else:
                final_category_for_csv = raw_model_category
                if "Error:" not in final_category_for_csv:
                    processed_count += 1
            current_run_results[filename_key] = final_category_for_csv
        except FileNotFoundError:
            error_count += 1
            skipped_images.append(relative_path)
            current_run_results[filename_key] = "Error: File Not Found"
        except Exception as loop_err:
            logging.error(f"Unexpected error processing {image_path} for blur: {loop_err}") 
            error_count += 1
            skipped_images.append(relative_path)
            current_run_results[filename_key] = "Error: Processing Loop"

    if current_run_results:
        category_counts = Counter(current_run_results.values())
        logging.info(f"\n--- Blur Assessment Counts ('{run_identifier}') ---")
        for category, count in sorted(category_counts.items()):
            logging.info(f"  {category}: {count}")
        logging.info("-----------------------------------------")
    else:
        logging.warning("No blur assessment results generated.") 
        del model, processor
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

    results_list_for_df = [{'Filename': key, run_identifier: value} for key, value in current_run_results.items()]
    new_results_df = pd.DataFrame(results_list_for_df)
    merged_df = new_results_df

    try:
        ensure_dir(os.path.dirname(output_csv_path))
        merged_df.to_csv(output_csv_path, index=False, encoding='utf-8', na_rep='')
        logging.info(f"Successfully saved blur assessment results to {output_csv_path}")
    except Exception as e:
        logging.error(f"Error saving blur CSV to {output_csv_path}: {e}") 

    total_time = time.time() - start_time
    # Keep Summary Info Block
    logging.info(f"\n===== Blur Assessment Summary (Run: '{run_identifier}') =====")
    logging.info(f"Images assessed successfully: {processed_count}")
    logging.info(f"Images skipped or failed: {error_count}")
    if skipped_images:
        logging.info(f"Skipped/Failed images: {len(skipped_images)}")
    logging.info(f"Total time for blur assessment: {total_time:.2f} seconds")
    logging.info(f"Results saved in: {output_csv_path}")
    logging.info(f"========================================================")

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Qwen model for blur assessment unloaded.")
    return output_csv_path