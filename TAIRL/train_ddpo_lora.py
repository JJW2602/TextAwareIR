#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).with_name("train_grpo_ddpo_lora.py")), run_name="__main__")
