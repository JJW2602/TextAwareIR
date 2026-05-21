import json
import os
import logging
import sys
import time
import re
import random
import numpy as np
import torch


# --- Seeding Function ---
def seed_everything(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Set random seed to {seed}")


# --- Custom Filter for Logging ---
class VLMProgressFilter(logging.Filter):
    """
    Filters out log messages containing the specific VLM progress pattern.
    (Will be less relevant now but kept for structure).
    """
    VLM_PROGRESS_MARKER = "[VLM Progress]"

    def filter(self, record):
        return self.VLM_PROGRESS_MARKER not in record.getMessage()


# --- Logging Setup ---
def setup_logging(config, output_dir):
    """
    Sets up logging to file (excluding VLM progress) and console.
    """
    log_level_console_str = config.get('log_level_console', 'INFO').upper()
    log_level_file_str = config.get('log_level_file', 'DEBUG').upper()
    log_filename = config.get('log_filename', 'pipeline.log')

    log_level_console = getattr(logging, log_level_console_str, logging.INFO)
    log_level_file = getattr(logging, log_level_file_str, logging.DEBUG)
    root_level = min(log_level_console, log_level_file)

    log_file_path = os.path.join(output_dir, log_filename)
    ensure_dir(output_dir)

    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
    root_logger.setLevel(root_level)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_console)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    print(f"Logging to console (Level: {log_level_console_str})")

    # File Handler
    try:
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level_file)
        file_handler.setFormatter(log_formatter)
        file_handler.addFilter(VLMProgressFilter())
        root_logger.addHandler(file_handler)
        print(f"Logging to file: {log_file_path} (Level: {log_level_file_str}, VLM progress excluded)")
    except Exception as e:
        print(f"ERROR: Failed to set up file logging at {log_file_path}: {e}")

    logging.info("Logging setup complete.")


# --- Filesystem Utilities ---
def ensure_dir(path):
    if path and not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"Created directory: {path}")
        except OSError as e:
            print(f"Error creating directory {path}: {e}")


def read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error reading JSON {path}: {e}")
        return None


def write_json(data, path, indent=2):
    try:
        ensure_dir(os.path.dirname(path))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        # logging.debug(f"Successfully wrote JSON to: {path}") # REMOVED DEBUG
    except Exception as e:
        logging.error(f"Error writing JSON to {path}: {e}")


def read_text_list(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Text list file not found: {path}")
        return None
    except Exception as e:
        logging.error(f"Error reading text list {path}: {e}")
        return None


def find_image_paths_os(directory, extensions=None):
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    valid_extensions = tuple(ext.lower() for ext in extensions if isinstance(ext, str) and ext.startswith('.'))
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    count = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                yield os.path.join(root, filename)
                count += 1
    logging.info(f"Found {count} potential image files in '{directory}'.") # Keep INFO summary