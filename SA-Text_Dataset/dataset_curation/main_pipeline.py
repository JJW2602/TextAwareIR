import argparse
import yaml
import os
import logging
import time
import shutil
import sys
import re

# Import pipeline steps from src
from src import utils
from src import bridge_runner
from src import cropping
from src import vlm_processing
from src import filtering
from src import formatting


# Helper function for timing steps
def time_step(step_name, func, *args, **kwargs):
    """Times a pipeline step and logs the duration using the ROOT logger (for file)."""
    logging.info(f"--- Starting: {step_name} ---")
    step_start_time = time.time()
    result = func(*args, **kwargs)
    step_end_time = time.time()
    duration = step_end_time - step_start_time
    logging.info(f"--- Finished: {step_name} (Duration: {duration:.2f} seconds) ---")
    if result is None:
        logging.error(f"--- Step Failed: {step_name} ---")
        raise RuntimeError(f"Step '{step_name}' failed.")
    return result


def main():
    valid_stages = [
        'start', 'cropping', 'bridge_stage2',
        'filter_duplicates', 'vlm1_recognition', 'vlm2_recognition',
        'vlm_filtering', 'vlm_comparison', 'agreement_extraction',
        'blur_assessment', 'blur_tag_filter', 'final_formatting'
    ]

    parser = argparse.ArgumentParser(description="SA-1B Text Restoration Dataset Curation Pipeline")
    parser.add_argument("--config", required=True, help="Path to the base configuration YAML file")
    parser.add_argument("--sa1b_subfolder", type=str, default=None, help="Override the 'sa1b_subfolder' from the config file (e.g., 'sa_000001')")
    parser.add_argument("--output_suffix", type=str, default=None, help="Suffix to append to output directory names (e.g., '_000001')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--duplicate_iou_thresh", type=float, default=0.9, help="IoU threshold for filtering duplicate detections (default: 0.9)")
    parser.add_argument(
        "--start_from",
        type=str,
        default="start",
        choices=valid_stages,
        help="Stage to start the pipeline from. Assumes previous stages are complete and outputs exist."
    )
    parser.add_argument(
        "--run_only_stage",
        type=str,
        default=None,
        choices=valid_stages,
        help="Run ONLY the specified stage and then exit successfully. Requires appropriate --start_from setting."
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    try:
        with open(args.config, 'r') as f: config = yaml.safe_load(f)
        print(f"Loaded base configuration from: {args.config}")
    except Exception as e: print(f"FATAL: Error loading config file {args.config}: {e}"); sys.exit(1)

    # --- Apply Overrides & Determine Suffix ---
    overridden_keys = []
    if args.sa1b_subfolder: config['sa1b_subfolder'] = args.sa1b_subfolder; overridden_keys.append('sa1b_subfolder')
    if args.output_suffix: output_suffix_raw = args.output_suffix; overridden_keys.append('output_suffix')
    else: output_suffix_raw = f"_{config.get('sa1b_subfolder', 'default')}"
    output_suffix = output_suffix_raw.replace('/', '_').replace('\\', '_')

    # --- Define Paths (incorporating suffix) ---
    sa1b_input_dir = os.path.join(config['sa1b_base_dir'], config['sa1b_subfolder'])
    intermediate_base = config['intermediate_output_base_dir'] + output_suffix
    final_dataset_output_dir = config['final_dataset_dir'] + output_suffix
    intermediate_dir = os.path.join(intermediate_base, "intermediate")
    crop_image_dir = os.path.join(intermediate_base, "cropped_images")

    # --- Setup Logging AFTER determining output paths ---
    utils.setup_logging(config, intermediate_base)

    # --- Set Random Seed ---
    utils.seed_everything(args.seed)

    logging.info("Pipeline Started.")
    pipeline_start_time = time.time()
    logging.info(f"Starting from stage: '{args.start_from}'")
    logging.info(f"Using configuration: {config}")
    if overridden_keys: logging.info(f"Applied overrides for: {', '.join(overridden_keys)}")
    logging.info(f"Using output suffix: '{output_suffix}'")
    logging.info(f"Intermediate base directory: {intermediate_base}")
    logging.info(f"Final dataset directory: {final_dataset_output_dir}")

    # --- Ensure Output Dirs Exist ---
    utils.ensure_dir(intermediate_base); utils.ensure_dir(intermediate_dir)
    utils.ensure_dir(crop_image_dir); utils.ensure_dir(final_dataset_output_dir)

    # Define intermediate/final file paths
    stage1_json = os.path.join(intermediate_dir, f"bridge_stage1_results{output_suffix}.json")
    crop_definitions_path = os.path.join(intermediate_dir, f"crop_definitions{output_suffix}.json")
    stage2_raw_json = os.path.join(intermediate_dir, f"bridge_stage2_raw_results{output_suffix}.json") 
    stage2_filtered_json = os.path.join(intermediate_dir, f"bridge_stage2_filtered_results{output_suffix}.json") 
    vlm1_raw_json = os.path.join(intermediate_dir, f"{config['vlm1_name']}_raw{output_suffix}.json")
    vlm2_raw_json = os.path.join(intermediate_dir, f"{config['vlm2_name']}_raw{output_suffix}.json")
    vlm1_filtered_json = os.path.join(intermediate_dir, f"{config['vlm1_name']}_filtered{output_suffix}.json")
    vlm2_filtered_json = os.path.join(intermediate_dir, f"{config['vlm2_name']}_filtered{output_suffix}.json")
    combined_json = os.path.join(intermediate_dir, f"vlm_combined{output_suffix}.json")
    agreed_list_txt = os.path.join(intermediate_dir, f"agreed_image_list{output_suffix}.txt")
    agreed_json = os.path.join(intermediate_dir, f"agreed_annotations{output_suffix}.json")
    blur_csv = os.path.join(intermediate_dir, f"blur_assessment{output_suffix}.csv")
    tagged_json_intermediate = os.path.join(intermediate_dir, f"tagged_annotations_intermediate{output_suffix}.json")
    restoration_json_intermediate = os.path.join(intermediate_dir, f"restoration_annotations_intermediate{output_suffix}.json")
    full_dataset_final_json = os.path.join(final_dataset_output_dir, f"full_dataset{output_suffix}.json")
    restoration_dataset_final_json = os.path.join(final_dataset_output_dir, f"restoration_dataset{output_suffix}.json")

    pipeline_steps = {}

    if args.run_only_stage:
        try:
            start_idx = valid_stages.index(args.start_from)
            only_idx = valid_stages.index(args.run_only_stage)
            if only_idx < start_idx:
                 logging.error(f"--run_only_stage ('{args.run_only_stage}') cannot be earlier than --start_from ('{args.start_from}')")
                 sys.exit(1)
            logging.info(f"Pipeline configured to run ONLY stage: '{args.run_only_stage}' and then exit.")
        except ValueError:
            logging.error(f"Invalid stage name provided for --start_from or --run_only_stage.")
            sys.exit(1)

    try:
        start_index = valid_stages.index(args.start_from)
    except ValueError:
        logging.error(f"Invalid --start_from value: {args.start_from}")
        sys.exit(1)

    def should_run(stage_name):
        try:
            return valid_stages.index(stage_name) >= start_index
        except ValueError:
            logging.error(f"Internal error: Invalid stage name '{stage_name}' used in should_run check.")
            return False

    def check_required_inputs(stage_name):
        logging.info(f"Checking required inputs for starting stage '{stage_name}'...")
        required_ok = True
        def check_and_set(key, path):
            nonlocal required_ok
            if not os.path.exists(path): logging.error(f"Missing required input for stage '{stage_name}': {path}"); required_ok = False
            return required_ok
        def check_dir_and_set(key, path):
             nonlocal required_ok
             if not os.path.isdir(path) or not os.listdir(path): logging.error(f"Missing required input for stage '{stage_name}': Populated directory {path}"); required_ok = False
             return required_ok

        if stage_name == 'cropping': check_and_set('stage1_json', stage1_json)
        elif stage_name == 'bridge_stage2': check_dir_and_set('crop_image_dir', crop_image_dir)
        elif stage_name == 'filter_duplicates': 
             check_and_set('stage2_raw_json', stage2_raw_json) 
        elif stage_name == 'vlm1_recognition': 
            check_and_set('stage2_filtered_json', stage2_filtered_json) 
            check_dir_and_set('crop_image_dir', crop_image_dir)
        elif stage_name == 'vlm2_recognition':
            check_and_set('stage2_filtered_json', stage2_filtered_json) 
            check_dir_and_set('crop_image_dir', crop_image_dir)
            check_and_set('vlm1_raw_json', vlm1_raw_json) 
        elif stage_name == 'vlm_filtering':
            check_and_set('vlm1_raw_json', vlm1_raw_json)
            check_and_set('vlm2_raw_json', vlm2_raw_json)
        elif stage_name == 'vlm_comparison':
            check_and_set('vlm1_filtered_json', vlm1_filtered_json)
            check_and_set('vlm2_filtered_json', vlm2_filtered_json)
        elif stage_name == 'agreement_extraction': check_and_set('combined_json', combined_json)
        elif stage_name == 'blur_assessment':
            check_and_set('agreed_json', agreed_json); check_and_set('agreed_list_txt', agreed_list_txt)
            check_dir_and_set('crop_image_dir', crop_image_dir)
        elif stage_name == 'blur_tag_filter':
            check_and_set('agreed_json', agreed_json); check_and_set('blur_csv', blur_csv)
        elif stage_name == 'final_formatting':
            check_and_set('tagged_json_intermediate', tagged_json_intermediate)
            check_and_set('restoration_json_intermediate', restoration_json_intermediate)

        if not required_ok: logging.error(f"Cannot proceed from stage '{stage_name}' due to missing inputs."); sys.exit(1)
        else: logging.info(f"Required inputs for stage '{stage_name}' found.")


    # --- Main Pipeline Execution ---
    try:
        if args.start_from != 'start':
            check_required_inputs(args.start_from)

        # --- Step 1: Bridge Stage 1 ---
        current_stage_name = 'start'
        if should_run('start'):
            stage1_output_dir = os.path.join(intermediate_dir, "bridge_stage1")
            pipeline_steps['stage1_json_temp'] = time_step(
                f"Step 1: Bridge Detection (Stage 1) [{config['sa1b_subfolder']}]",
                bridge_runner.run_bridge, config, sa1b_input_dir, stage1_output_dir, stage1=True
            )
            expected_stage1_output = os.path.join(stage1_output_dir, "text_detection_results.json")
            if os.path.exists(expected_stage1_output):
                shutil.move(expected_stage1_output, stage1_json)
                pipeline_steps['stage1_json'] = stage1_json
                logging.info(f"Stage 1 results moved to: {stage1_json}")
                if not config.get('keep_intermediate_files', True): shutil.rmtree(stage1_output_dir, ignore_errors=True)
            elif os.path.exists(stage1_json):
                 logging.warning(f"Target Stage 1 JSON {stage1_json} already exists. Using existing file.")
                 pipeline_steps['stage1_json'] = stage1_json
            else: raise RuntimeError(f"Bridge Stage 1 output JSON not found at expected location: {expected_stage1_output}")
            if args.run_only_stage == current_stage_name:
                logging.info(f"Completed requested stage '{args.run_only_stage}'. Exiting pipeline.")
                sys.exit(0)
        elif 'stage1_json' not in pipeline_steps:
             pipeline_steps['stage1_json'] = stage1_json


        # --- Step 2 & 3: Cropping ---
        current_stage_name = 'cropping'
        if should_run('cropping'):
            # --- Step 2: Define Crop Regions ---
            step2_name = f"Step 2: Define Crop Regions [{config['sa1b_subfolder']}]"
            crop_definitions = time_step(
                f"Step 2: Define Crop Regions [{config['sa1b_subfolder']}]",
                cropping.define_crop_regions, pipeline_steps['stage1_json'], config
            )
            crop_definitions_output_path = os.path.join(intermediate_dir, f"crop_definitions{output_suffix}.json")
            if crop_definitions:
                utils.write_json(crop_definitions, crop_definitions_output_path)
                logging.info(f"Saved crop definitions to: {crop_definitions_output_path}")
            else:
                logging.warning("No crop definitions were generated in Step 2. Nothing saved.")
            if args.run_only_stage == current_stage_name:
                logging.info(f"Completed requested stage '{args.run_only_stage}' (defined crops). Exiting pipeline.")
                sys.exit(0)
            
            # --- Step 3: Create Crop Images ---
            step3_name = f"Step 3: Create Crop Images [{config['sa1b_subfolder']}]"
            if crop_definitions: # Only run if definitions were created
                 time_step(
                    step3_name,
                    cropping.create_crop_images, crop_definitions, sa1b_input_dir, crop_image_dir, config
                 )
                 pipeline_steps['crop_image_dir'] = crop_image_dir
            else:
                 logging.warning(f"Skipping '{step3_name}' because crop definitions were empty.")
        elif 'crop_image_dir' not in pipeline_steps:
             pipeline_steps['crop_image_dir'] = crop_image_dir


        # --- Step 4: Bridge Stage 2 ---
        current_stage_name = 'bridge_stage2'
        if should_run('bridge_stage2'):
            stage2_output_dir = os.path.join(intermediate_dir, "bridge_stage2")
            step4_name = f"Step 4: Bridge Detection (Stage 2 on Crops) [{config['sa1b_subfolder']}]"
            pipeline_steps['stage2_raw_json_temp'] = time_step(f"Step 4: Bridge Detection (Stage 2 on Crops) [{config['sa1b_subfolder']}]", bridge_runner.run_bridge, config, pipeline_steps['crop_image_dir'], stage2_output_dir, stage1=False)
            expected_stage2_output = os.path.join(stage2_output_dir, "text_detection_results.json")
            if os.path.exists(expected_stage2_output):
                shutil.move(expected_stage2_output, stage2_raw_json) # Move to RAW json path
                pipeline_steps['stage2_raw_json'] = stage2_raw_json
                logging.info(f"Stage 2 results moved to: {stage2_raw_json}")
                if not config.get('keep_intermediate_files', True): shutil.rmtree(stage2_output_dir, ignore_errors=True)
            elif os.path.exists(stage2_raw_json):
                 logging.warning(f"Target Stage 2 RAW JSON {stage2_raw_json} already exists. Using existing file.")
                 pipeline_steps['stage2_raw_json'] = stage2_raw_json
            else: raise RuntimeError(f"Bridge Stage 2 output JSON not found at expected location: {expected_stage2_output}")
            
            if args.run_only_stage == current_stage_name:
                logging.info(f"Completed requested stage '{args.run_only_stage}'. Exiting pipeline.")
                sys.exit(0)
        elif 'stage2_raw_json' not in pipeline_steps: pipeline_steps['stage2_raw_json'] = stage2_raw_json

        current_stage_name = 'filter_duplicates'
        if should_run('filter_duplicates'):
            pipeline_steps['stage2_filtered_json'] = time_step(
                f"Step 4.5: Filter Duplicate Detections [{config['sa1b_subfolder']}]",
                filtering.filter_duplicate_detections,
                pipeline_steps['stage2_raw_json'], # Input is raw stage 2
                stage2_filtered_json,             # Output is filtered stage 2
                config,
                iou_threshold=args.duplicate_iou_thresh # Pass threshold from args
            )
            if args.run_only_stage == current_stage_name:
                logging.info(f"Completed requested stage '{args.run_only_stage}'. Exiting pipeline.")
                sys.exit(0)
        elif 'stage2_filtered_json' not in pipeline_steps: pipeline_steps['stage2_filtered_json'] = stage2_filtered_json

        # --- Step 5: VLM 1 Recognition ---
        current_stage_name = 'vlm1_recognition'
        if should_run('vlm1_recognition'):
            pipeline_steps['vlm1_raw_json'] = time_step(
                f"Step 5: VLM Recognition ({config['vlm1_name']}) [{config['sa1b_subfolder']}]",
                vlm_processing.run_vlm_recognition,
                config['vlm1_name'],
                pipeline_steps['stage2_filtered_json'], # Use FILTERED stage 2 results
                pipeline_steps['crop_image_dir'], vlm1_raw_json, config
            )

            if args.run_only_stage == current_stage_name:
                logging.info(f"Completed requested stage '{args.run_only_stage}'. Exiting pipeline.")
                sys.exit(0)

        elif 'vlm1_raw_json' not in pipeline_steps: pipeline_steps['vlm1_raw_json'] = vlm1_raw_json

        # --- Step 6: VLM 2 Recognition ---
        current_stage_name = 'vlm2_recognition'
        if should_run('vlm2_recognition'):
            pipeline_steps['vlm2_raw_json'] = time_step(
                f"Step 6: VLM Recognition ({config['vlm2_name']}) [{config['sa1b_subfolder']}]",
                vlm_processing.run_vlm_recognition,
                config['vlm2_name'],
                pipeline_steps['stage2_filtered_json'], # Use FILTERED stage 2 results
                pipeline_steps['crop_image_dir'], vlm2_raw_json, config
            )

            if args.run_only_stage == current_stage_name:
                logging.info(f"Completed requested stage '{args.run_only_stage}'. Exiting pipeline.")
                sys.exit(0)

        elif 'vlm2_raw_json' not in pipeline_steps: pipeline_steps['vlm2_raw_json'] = vlm2_raw_json


        # --- Step 7 & 8: Filter VLM Results ---
        current_stage_name = 'vlm_filtering'
        if should_run('vlm_filtering'):
            step7_name = f"Step 7: Filter Empty VLM Results ({config['vlm1_name']}) [{config['sa1b_subfolder']}]"
            pipeline_steps['vlm1_filtered_json'] = time_step(
                f"Step 7: Filter Empty VLM Results ({config['vlm1_name']}) [{config['sa1b_subfolder']}]",
                filtering.filter_empty_vlm, pipeline_steps['vlm1_raw_json'], vlm1_filtered_json, config
            )
            step8_name = f"Step 8: Filter Empty VLM Results ({config['vlm2_name']}) [{config['sa1b_subfolder']}]"
            pipeline_steps['vlm2_filtered_json'] = time_step(
                f"Step 8: Filter Empty VLM Results ({config['vlm2_name']}) [{config['sa1b_subfolder']}]",
                filtering.filter_empty_vlm, pipeline_steps['vlm2_raw_json'], vlm2_filtered_json, config
            )
            if args.run_only_stage == current_stage_name:
                logging.info(f"Completed requested stage '{args.run_only_stage}'. Exiting pipeline.")
                sys.exit(0)
        elif 'vlm1_filtered_json' not in pipeline_steps:
             pipeline_steps['vlm1_filtered_json'] = vlm1_filtered_json
             pipeline_steps['vlm2_filtered_json'] = vlm2_filtered_json


        # --- Step 9: Compare & Merge VLMs ---
        current_stage_name = 'vlm_comparison'
        if should_run('vlm_comparison'):
            step9_name = f"Step 9: Compare & Merge VLMs [{config['sa1b_subfolder']}]"
            pipeline_steps['combined_json'] = time_step(
                f"Step 9: Compare & Merge VLMs [{config['sa1b_subfolder']}]",
                filtering.compare_and_merge_vlms, pipeline_steps['vlm1_filtered_json'], pipeline_steps['vlm2_filtered_json'], combined_json, config
            )
            if args.run_only_stage == current_stage_name:
                logging.info(f"Completed requested stage '{args.run_only_stage}'. Exiting pipeline.")
                sys.exit(0)
        elif 'combined_json' not in pipeline_steps:
             pipeline_steps['combined_json'] = combined_json


        # --- Step 10 & 11: Agreement Extraction ---
        current_stage_name = 'agreement_extraction'
        if should_run('agreement_extraction'):
            step10_name = f"Step 10: Identify Fully Agreed Images [{config['sa1b_subfolder']}]"
            pipeline_steps['agreed_list_txt'] = time_step(
                f"Step 10: Identify Fully Agreed Images [{config['sa1b_subfolder']}]",
                filtering.identify_agreed_images, pipeline_steps['combined_json'], agreed_list_txt, config
            )
            step11_name = f"Step 11: Extract Agreed Annotations [{config['sa1b_subfolder']}]"
            pipeline_steps['agreed_json'] = time_step(
                f"Step 11: Extract Agreed Annotations [{config['sa1b_subfolder']}]",
                filtering.extract_agreed_annotations, pipeline_steps['combined_json'], pipeline_steps['agreed_list_txt'], agreed_json, config
            )

            if args.run_only_stage == current_stage_name:
                logging.info(f"Completed requested stage '{args.run_only_stage}'. Exiting pipeline.")
                sys.exit(0)
        elif 'agreed_list_txt' not in pipeline_steps:
             pipeline_steps['agreed_list_txt'] = agreed_list_txt
             pipeline_steps['agreed_json'] = agreed_json


        # --- Step 12: Assess Blur ---
        current_stage_name = 'blur_assessment'
        if should_run('blur_assessment'):
            agreed_filenames_list = utils.read_text_list(pipeline_steps['agreed_list_txt'])
            if agreed_filenames_list is None:
                raise RuntimeError("Failed to read agreed image list for blur assessment.")
            step12_name = f"Step 12: Assess Crop Blurriness (Agreed Crops) [{config['sa1b_subfolder']}]"            
            pipeline_steps['blur_csv'] = time_step(
                f"Step 12: Assess Crop Blurriness (Agreed Crops) [{config['sa1b_subfolder']}]",
                vlm_processing.run_blur_assessment, agreed_filenames_list, pipeline_steps['crop_image_dir'], blur_csv, config
            )

            if args.run_only_stage == current_stage_name:
                logging.info(f"Completed requested stage '{args.run_only_stage}'. Exiting pipeline.")
                sys.exit(0)

        elif 'blur_csv' not in pipeline_steps:
             pipeline_steps['blur_csv'] = blur_csv


        # --- Step 13: Tag and Filter by Blur ---
        current_stage_name = 'blur_tag_filter'
        if should_run('blur_tag_filter'):
            step13a_name = f"Step 13a: Tag Images with Blur Category [{config['sa1b_subfolder']}]"
            pipeline_steps['tagged_json_intermediate'] = time_step(
                f"Step 13a: Tag Images with Blur Category [{config['sa1b_subfolder']}]",
                filtering.tag_with_blur, pipeline_steps['agreed_json'], pipeline_steps['blur_csv'], tagged_json_intermediate, config
            )
            step13b_name = f"Step 13b: Filter Tagged Data by Blur [{config['sa1b_subfolder']}]"
            pipeline_steps['restoration_json_intermediate'] = time_step(
                f"Step 13b: Filter Tagged Data by Blur [{config['sa1b_subfolder']}]",
                filtering.filter_tagged_by_blur, pipeline_steps['tagged_json_intermediate'], restoration_json_intermediate, config
            )
            if args.run_only_stage == current_stage_name:
                logging.info(f"Completed requested stage '{args.run_only_stage}'. Exiting pipeline.")
                sys.exit(0)
        elif 'tagged_json_intermediate' not in pipeline_steps:
             pipeline_steps['tagged_json_intermediate'] = tagged_json_intermediate
             pipeline_steps['restoration_json_intermediate'] = restoration_json_intermediate


        # --- Step 14: Final Formatting ---
        current_stage_name = 'final_formatting'
        if should_run('final_formatting'):
            step14a_name = f"Step 14a: Final Formatting (Full Dataset) [{config['sa1b_subfolder']}]"            
            pipeline_steps['full_dataset_final_json'] = time_step(
                f"Step 14a: Final Formatting (Full Dataset) [{config['sa1b_subfolder']}]",
                formatting.format_final_dataset, pipeline_steps['tagged_json_intermediate'], full_dataset_final_json, config
            )
            step14b_name = f"Step 14b: Final Formatting (Restoration Dataset) [{config['sa1b_subfolder']}]"
            pipeline_steps['restoration_dataset_final_json'] = time_step(
                f"Step 14b: Final Formatting (Restoration Dataset) [{config['sa1b_subfolder']}]",
                formatting.format_final_dataset, pipeline_steps['restoration_json_intermediate'], restoration_dataset_final_json, config
            )
            if args.run_only_stage == current_stage_name:
                logging.info(f"Completed requested stage '{args.run_only_stage}'. Exiting pipeline.")
                sys.exit(0)


        logging.info(f"--- Pipeline Completed Successfully for {config['sa1b_subfolder']} ---")
        if 'full_dataset_final_json' in pipeline_steps and os.path.exists(pipeline_steps['full_dataset_final_json']):
            logging.info(f"Full dataset (tagged) saved to: {pipeline_steps['full_dataset_final_json']}")
        if 'restoration_dataset_final_json' in pipeline_steps and os.path.exists(pipeline_steps['restoration_dataset_final_json']):
            logging.info(f"Restoration dataset (filtered) saved to: {pipeline_steps['restoration_dataset_final_json']}")

    except Exception as e:
        logging.error(f"Pipeline failed for {config.get('sa1b_subfolder', 'N/A')}: {e}", exc_info=True)

    finally:
        # --- Cleanup ---
        if not config.get('keep_intermediate_files', True):
            logging.info(f"Cleaning up intermediate files for {config.get('sa1b_subfolder', 'N/A')}...")
            files_to_remove = [
                stage1_json, stage2_raw_json, stage2_filtered_json, # Updated stage 2 names
                vlm1_raw_json, vlm2_raw_json, vlm1_filtered_json, vlm2_filtered_json,
                combined_json, agreed_list_txt, agreed_json, blur_csv,
                tagged_json_intermediate, restoration_json_intermediate
            ]
            dirs_to_remove = [ os.path.join(intermediate_dir, "bridge_stage1"), os.path.join(intermediate_dir, "bridge_stage2") ]
            for f_path in files_to_remove:
                 if f_path and os.path.exists(f_path):
                     try: os.remove(f_path); logging.debug(f"Removed intermediate file: {f_path}")
                     except OSError as clean_e: logging.warning(f"Could not remove intermediate file {f_path}: {clean_e}")
            for d_path in dirs_to_remove:
                 if d_path and os.path.exists(d_path):
                     try: shutil.rmtree(d_path, ignore_errors=True); logging.debug(f"Removed intermediate directory: {d_path}")
                     except OSError as clean_e: logging.warning(f"Could not remove intermediate dir {d_path}: {clean_e}")

        pipeline_end_time = time.time()
        logging.info(f"Total pipeline execution time for {config.get('sa1b_subfolder', 'N/A')}: {pipeline_end_time - pipeline_start_time:.2f} seconds.")
        logging.shutdown()


if __name__ == "__main__":
    main()