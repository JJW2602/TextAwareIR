#!/bin/bash

# Set TMPDIR to same filesystem as pip cache (/home) to avoid cross-device rename error (Errno 18)
mkdir -p /home/james2602/tmp
export TMPDIR=/home/james2602/tmp

# Install PyTorch
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# Install core libraries
pip install opencv-python scipy timm shapely albumentations Polygon3 pandas tqdm pyyaml

# Install VLM and Flash Attention libraries
pip install transformers==4.51.3 accelerate scikit-learn qwen_vl_utils pytz
pip install flash-attn --no-build-isolation

# Restore TMPDIR
unset TMPDIR
