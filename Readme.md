# TextAwareIR

텍스트가 포함된 저화질 이미지를 복원하고 평가하기 위한 연구 모노레포입니다.

- **TAIR / TeReDiff**: text-aware image restoration baseline
- **DiffBIR**: diffusion-based blind image restoration baseline
- **Dataset pipeline**: SA-1B 기반 SA-Text 큐레이션 및 OOD text 평가셋 생성
- **TAIRL**: Bridge OCR reward를 이용한 DiffBIR ControlNet LoRA의 DDPO/GRPO fine-tuning

> 이 저장소의 Slurm 스크립트와 YAML 설정은 기본적으로
> `/scratch2/james2602/TextAwareIR` 및
> `/home/james2602/miniconda3/envs/*` 경로를 사용합니다.
> 다른 머신에서는 [Path Configuration](#path-configuration)을 먼저 확인하세요.

## Contents

- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Weights](#weights)
- [Baseline Evaluation](#baseline-evaluation)
- [Dataset Preparation](#dataset-preparation)
- [TAIRL Training](#tairl-training)
- [Outputs and Logs](#outputs-and-logs)
- [Path Configuration](#path-configuration)
- [Cluster Notes](#cluster-notes)
- [Citation](#citation)

## Repository Layout

```text
TextAwareIR/
├── TAIR/                   # TeReDiff / TAIR baseline
├── DiffBIR/                # DiffBIR baseline
├── TAIRL/                  # DDPO/GRPO + LoRA 학습 코드와 설정
├── Dataset_pipeline/
│   ├── build_ood_text_eval_dataset.py
│   └── SA-Text_Dataset/    # SA-Text 큐레이션 + Bridge text spotting
├── Dataset/
│   ├── SA-Text/            # 원본 SA-Text train
│   ├── SA-Text-test/       # degradation level 1/2/3 평가셋
│   ├── SA-Text-lv2-10000/  # TAIRL용 고정 10K HQ/LQ pair
│   ├── DrealSR/
│   ├── realsr/
│   ├── OOD-text-test/      # DrealSR + RealSR 기반 OOD text 평가셋
│   └── gopro/
├── Slurm/
│   ├── Baselines/TAIR/
│   └── Baselines/DiffBIR/
└── Results/
    ├── Baselines/
    │   ├── TAIR/
    │   ├── DiffBIR/
    │   └── Compare/
    └── TAIRL/
```

세부 구현과 upstream 사용법은 각 하위 문서를 참고하세요.

- [TAIR README](TAIR/README.md)
- [DiffBIR README](DiffBIR/README.md)
- [TAIRL README](TAIRL/README.md)
- [SA-Text pipeline README](Dataset_pipeline/SA-Text_Dataset/README.md)

## Quick Start

모든 명령은 저장소 루트에서 실행하는 것을 기준으로 합니다.

```bash
cd /scratch2/james2602/TextAwareIR
```

### Baseline 100-image evaluation

아래 `bash` 스크립트는 GPU가 이미 할당된 interactive shell에서 실행합니다.

```bash
# DiffBIR: SA-Text-test level 2, first 100 images
bash Slurm/Baselines/DiffBIR/eval_diffbir_baseline.sh

# TAIR: SA-Text-test level 2, first 100 images
bash Slurm/Baselines/TAIR/eval_tair_baseline.sh

# OOD-text-test
bash Slurm/Baselines/DiffBIR/eval_diffbir_baseline_ood_text_test.sh
bash Slurm/Baselines/TAIR/eval_tair_baseline_ood_text_test.sh
```

샘플 수와 배치는 환경변수로 변경할 수 있습니다.

```bash
NUM_IMAGES=20 START=100 BATCH_SIZE=4 \
  bash Slurm/Baselines/DiffBIR/eval_diffbir_baseline.sh
```

### Slurm submission

```bash
# DiffBIR baseline 평가
sbatch Slurm/Baselines/DiffBIR/eval_diffbir_baseline.slurm

# DiffBIR SA-Text-test 2개 chunk inference / annotation
sbatch Slurm/Baselines/DiffBIR/infer_sa_text_lv2_2gpu.slurm
sbatch Slurm/Baselines/DiffBIR/eval_sa_text_lv2_2gpu.slurm

# TAIR 2-GPU inference + annotation + DiffBIR 비교
sbatch Slurm/Baselines/TAIR/run_sa_text_lv2_2gpu.slurm

# DDPO 11-trial sweep
sbatch TAIRL/slurm/ddpo_lora_hparam.slurm

# GRPO KL / PSNR 10-trial sweep
sbatch TAIRL/slurm/grpo_lora_g8_kl_psnr_10.slurm
```

## Environment Setup

### Environment matrix

| Workflow | Environment | Python | Main role |
|---|---|---:|---|
| DiffBIR inference/eval | `diffbir` | 3.10 | Ampere/Ada GPU용 DiffBIR |
| TAIR inference | `tair` | 3.10 | TeReDiff inference/training |
| Dataset/OCR pipeline | `dataset_curation` | 3.10 | Bridge, Detectron2, OVIS, Qwen |
| TAIRL training | `diffbir` + `dataset_curation` | 3.10 | rollout/update + OCR reward subprocess |

### DiffBIR: Ampere/Ada

```bash
conda create -n diffbir python=3.10 -y
conda activate diffbir
python -m pip install --upgrade pip
python -m pip install -r DiffBIR/requirements.txt

# Repository-specific helpers and TAIRL logging
python -m pip install hydra-core pyarrow pyyaml wandb matplotlib tqdm
```

### TAIR

```bash
conda create -n tair python=3.10 -y
conda activate tair

cd TAIR
python -m pip install \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

cd detectron2
python -m pip install --no-build-isolation -e .
cd ../testr
python -m pip install --no-build-isolation -e .
cd ../..
```

### Dataset curation / Bridge reward

```bash
conda create -n dataset_curation python=3.10 -y
conda activate dataset_curation

# Ampere/Ada example
python -m pip install \
  torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install \
  opencv-python scipy timm shapely albumentations Polygon3 \
  pandas tqdm pyyaml transformers==4.51.3 accelerate \
  scikit-learn qwen_vl_utils pytz
python -m pip install flash-attn --no-build-isolation
python -m pip install setuptools==59.5.0

cd Dataset_pipeline/SA-Text_Dataset/Bridging-Text-Spotting/detectron2
python setup.py build develop
cd ..
python setup.py build develop
cd ../../..
```

간단한 환경 확인:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

## Weights

현재 기본 설정이 참조하는 주요 파일은 다음과 같습니다.

```text
DiffBIR/weights/
├── sd2.1-base-zsnr-laionaes5.ckpt
├── DiffBIR_v2.1.pt
└── realesrgan_s4_swinir_100k.pth

TAIR/weights/
├── sd2.1-base-zsnr-laionaes5.ckpt
├── DiffBIR_v2.1.pt
├── realesrgan_s4_swinir_100k.pth
└── terediff_stage3.pt

Dataset_pipeline/SA-Text_Dataset/Bridging-Text-Spotting/
└── Bridge_tt.pth
```

TAIR의 DiffBIR 관련 weight는 다음 스크립트로 받을 수 있습니다.

```bash
cd TAIR
bash download_weights.sh
```

`terediff_stage3.pt`와 text spotting checkpoint는
[TAIR README](TAIR/README.md)의 별도 다운로드 안내를 따르세요.

TAIRL은 `TAIRL/configs/*.yaml`에 적힌 파일명이 정확히 존재해야 합니다.

## Baseline Evaluation

### Standalone inference

DiffBIR은 Hydra override 형식을 사용합니다.

```bash
conda activate diffbir
cd DiffBIR

python inference.py \
  task=sr \
  model.version=v2.1 \
  io.input=/path/to/lq_images \
  io.output=/path/to/restored_images
```

TAIR은 config 파일과 입력 디렉터리를 명시합니다.

```bash
conda activate tair
cd TAIR

python infer.py \
  --config configs/val/val_terediff.yaml \
  --infer-config configs/infer/infer_terediff.yaml \
  --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
  --input /path/to/lq_images \
  --output-dir /path/to/restored_images
```

### Evaluation scripts

| Script | Default dataset | Default output |
|---|---|---|
| `Slurm/Baselines/DiffBIR/eval_diffbir_baseline.sh` | `Dataset/SA-Text-test` | `Results/Baselines/DiffBIR/sa_text_test_lv2_default_reward` |
| `Slurm/Baselines/TAIR/eval_tair_baseline.sh` | `Dataset/SA-Text-test` | `Results/Baselines/TAIR/sa_text_test_lv2_default_reward` |
| `Slurm/Baselines/DiffBIR/eval_diffbir_baseline_ood_text_test.sh` | `Dataset/OOD-text-test` | `Results/Baselines/DiffBIR/ood_text_test_lv2_default_reward` |
| `Slurm/Baselines/TAIR/eval_tair_baseline_ood_text_test.sh` | `Dataset/OOD-text-test` | `Results/Baselines/TAIR/ood_text_test_default_reward` |

각 결과 디렉터리에는 일반적으로 다음 파일이 생성됩니다.

```text
metrics_summary.json
per_image_metrics.csv
resolved_config.yaml
```

### Inference-time benchmark

```bash
# 두 baseline을 순차 실행
bash Slurm/Baselines/benchmark_inference_time_sa_text_10.sh

# 개별 Slurm job
sbatch Slurm/Baselines/DiffBIR/benchmark_diffbir_inference_time_sa_text_10.slurm
sbatch Slurm/Baselines/TAIR/benchmark_tair_inference_time_sa_text_10.slurm
```

기본 결과는 `Results/Baselines/{DiffBIR,TAIR}/inference_time/`에 저장됩니다.

## Dataset Preparation

### SA-Text curation pipeline

SA-1B 이미지에서 text-rich crop을 찾고, Bridge와 두 VLM의 결과를 조합해
최종 restoration dataset을 만듭니다.

먼저 아래 config의 입력, Bridge, 출력 경로를 현재 머신에 맞게 수정합니다.

```text
Dataset_pipeline/SA-Text_Dataset/dataset_curation/config.yaml
```

실행:

```bash
conda activate dataset_curation

python Dataset_pipeline/SA-Text_Dataset/dataset_curation/main_pipeline.py \
  --config Dataset_pipeline/SA-Text_Dataset/dataset_curation/config.yaml \
  --sa1b_subfolder sa_000000 \
  --output_suffix _sa_000000
```

Pipeline stage:

| Stage | Description |
|---|---|
| `start` | 원본 이미지 Bridge detection |
| `cropping` | text region을 포함하는 512px crop 생성 |
| `bridge_stage2` | crop에서 Bridge 재검출 |
| `filter_duplicates` | 중복 detection 제거 |
| `vlm1_recognition` | OVIS recognition |
| `vlm2_recognition` | Qwen recognition |
| `vlm_filtering` | 빈 결과 및 invalid 결과 제거 |
| `vlm_comparison` | 두 VLM 결과 병합/비교 |
| `agreement_extraction` | VLM 합의 sample 추출 |
| `blur_assessment` | Qwen 기반 blur 평가 |
| `blur_tag_filter` | blur tag 및 restoration subset 생성 |
| `final_formatting` | 최종 dataset JSON 생성 |

중간 단계부터 재시작하거나 한 단계만 실행할 수 있습니다.

```bash
python Dataset_pipeline/SA-Text_Dataset/dataset_curation/main_pipeline.py \
  --config Dataset_pipeline/SA-Text_Dataset/dataset_curation/config.yaml \
  --start_from bridge_stage2

python Dataset_pipeline/SA-Text_Dataset/dataset_curation/main_pipeline.py \
  --config Dataset_pipeline/SA-Text_Dataset/dataset_curation/config.yaml \
  --start_from blur_assessment \
  --run_only_stage blur_assessment
```

### TAIRL 10K level-2 pairs

고정 manifest의 SA-Text 10K sample을 512x512 HQ와 128x128 level-2 LQ pair로 만듭니다.

```bash
sbatch TAIRL/slurm/make_sa_text_lv2_10000.slurm

# 또는 할당된 GPU에서 직접 실행
bash TAIRL/slurm/make_sa_text_lv2_10000.sh
```

출력:

```text
Dataset/SA-Text-lv2-10000/
├── data/train-*.parquet
└── README.md
```

### OOD text evaluation set

DrealSR과 RealSR의 대응 HQ/LQ pair에서 readable text가 있는 scene을 선별합니다.
기본값은 source별 80장, 총 160장입니다.

```bash
conda activate dataset_curation

python Dataset_pipeline/build_ood_text_eval_dataset.py \
  --dreal-root Dataset/DrealSR/raw \
  --realsr-root "Dataset/realsr/RealSR (Final)" \
  --output-dir Dataset/OOD-text-test \
  --per-source 80 \
  --annotation-mode bridge \
  --write-parquet \
  --bridge-env-python \
    /home/james2602/miniconda3/envs/dataset_curation/bin/python \
  --overwrite
```

생성 과정:

1. HQ image에 Bridge OCR을 실행해 readable text가 있는 pair를 선별합니다.
2. text bbox를 포함하는 512x512 HQ crop을 만듭니다.
3. 같은 좌표를 해상도 비율에 맞춰 LQ에 투영합니다.
4. Bridge OCR 또는 Bridge + OVIS/Qwen 합의로 annotation을 만듭니다.
5. source와 scene이 중복되지 않도록 최종 sample을 선택합니다.

출력 구조:

```text
Dataset/OOD-text-test/
├── hq/
├── lq/
├── annotations/
│   ├── manifest.json
│   ├── manifest.jsonl
│   └── text_detection_results.json
├── data/
│   └── test-00000-of-00001.parquet
└── summary.json
```

주요 옵션:

- `--annotation-mode bridge`: Bridge의 `rec` 결과를 annotation으로 사용
- `--annotation-mode vlm_agreement`: Bridge stage 2와 OVIS/Qwen 합의까지 실행
- `--resize-lq-to-crop-size`: native LQ crop을 512x512로 resize
- `--allow-fewer`: source별 목표 수량보다 적어도 실패하지 않음
- `--write-parquet`: SA-Text와 유사한 parquet 추가 생성

## TAIRL Training

TAIRL은 pretrained DiffBIR를 고정하고 ControlNet에만 LoRA를 삽입합니다.

```text
[diffbir env]
LQ -> DiffBIR LoRA rollout -> restored images
                              |
                              v
[dataset_curation env]
Bridge text spotting -> bbox / recognized text
                              |
                              v
[diffbir env]
OCR reward + optional PSNR reward -> PPO/GRPO update
```

### Reward

IoU threshold 이상으로 매칭된 GT/prediction pair의 normalized Levenshtein
similarity를 사용합니다.

```text
matched_mean_reward =
  mean(max(1 - levenshtein(pred, gt) / len(gt), 0))

final_reward =
  matched_mean_reward
  - miss_penalty * missed_gt
  - false_positive_penalty * false_positive

final_reward_norm =
  matched_mean_reward
  - miss_penalty * missed_gt / max(num_gt, 1)
  - false_positive_penalty * false_positive / max(num_gt, 1)
```

설정은 `reward.variant`, `reward.miss_penalty`,
`reward.false_positive_penalty`, `reward.psnr_weight`로 제어합니다.

### Training commands

```bash
# DDPO 11-trial OAT sweep
sbatch TAIRL/slurm/ddpo_lora_hparam.slurm

# 기본 GRPO job
sbatch TAIRL/slurm/grpo_lora_g6.slurm

# GRPO reward/KL/PSNR 실험: 현재 파일 기본값은 task 0만 제출
sbatch TAIRL/slurm/grpo_lora_g8_hparam_10.slurm

# 전체 8개 trial 제출
sbatch --array=0-7 TAIRL/slurm/grpo_lora_g8_hparam_10.slurm

# KL 5개 + PSNR weight 5개
sbatch TAIRL/slurm/grpo_lora_g8_kl_psnr_10.slurm
```

> `grpo_lora_g6.yaml`과 일부 `g6` 스크립트명은 초기 실험 이름을 유지하고 있지만,
> 현재 YAML의 기본값은 `grpo.group_size: 8`,
> `grpo.generation_microbatch: 8`입니다.

직접 실행할 때는 YAML 값 뒤에 `key=value` override를 전달할 수 있습니다.

```bash
conda activate diffbir

python TAIRL/train_grpo_ddpo_lora.py \
  --config TAIRL/configs/grpo_lora_g6.yaml \
  train.train_steps=100 \
  train.output_dir=Results/TAIRL/debug_grpo \
  grpo.group_size=8 \
  grpo.generation_microbatch=1 \
  wandb.mode=offline
```

W&B online logging 전에는 `diffbir` 환경에서 한 번 로그인합니다.

```bash
python -m wandb login
```

네트워크가 없는 compute node에서는 `WANDB_MODE=offline`을 사용합니다.

## Outputs and Logs

### Baselines

```text
Results/Baselines/
├── DiffBIR/
│   ├── sa_text_test_lv2_default_reward/
│   ├── ood_text_test_lv2_default_reward/
│   └── inference_time/
├── TAIR/
│   ├── sa_text_test_lv2_default_reward/
│   ├── ood_text_test_default_reward/
│   └── inference_time/
└── Compare/
```

### TAIRL

```text
Results/TAIRL/<experiment>/<trial>/
├── metrics.jsonl
├── eval_metrics.jsonl
├── resolved_config.yaml
├── checkpoints/
├── reward_work/
└── wandb/
```

Slurm stdout/stderr:

```text
Slurm/Baselines/{TAIR,DiffBIR}/logs/
TAIRL/slurm/logs/
```

## Path Configuration

다른 경로나 conda 환경을 사용할 때 우선 확인할 위치:

1. `TAIRL/configs/ddpo_lora.yaml`
2. `TAIRL/configs/grpo_lora_g6.yaml`
3. `Dataset_pipeline/SA-Text_Dataset/dataset_curation/config.yaml`
4. 실행할 `Slurm/**/*.sh` 및 `Slurm/**/*.slurm`

대부분의 shell script는 아래 환경변수 중 일부를 지원합니다.

```bash
ROOT_DIR=/new/path/TextAwareIR
PYTHON_BIN=/new/env/bin/python
TAIR_PYTHON_BIN=/new/tair/env/bin/python
DIFFBIR_PYTHON_BIN=/new/diffbir/env/bin/python
PRED_PYTHON_BIN=/new/dataset_curation/env/bin/python
OUTPUT_DIR=/new/output/path
```

단, 일부 오래된 2-GPU 스크립트는 `ROOT_DIR`이 고정되어 있으므로 파일 안의
절대경로도 함께 수정해야 합니다.

## Cluster Notes

현재 주요 Slurm 스크립트는 다음 설정을 사용합니다.

| Purpose | Partition | QOS |
|---|---|---|
| 일반 1-GPU baseline/GRPO | `suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada` | `big_qos` |
| 기존 2-GPU/chunk pipeline | `suma_a6000` | `big_qos` |

다른 클러스터에서는 각 `.slurm` 파일의 `--partition`, `--qos`, `--gres`,
로그 경로를 수정하세요.

## Known Compatibility Notes

- `DiffBIR/diffbir/sampler/edm_sampler.py`는 `typing.Tuple`을 사용하도록 반영되어 있습니다.
- `Bridging-Text-Spotting/setup.py`는 `Pillow>=9.1`로 version conflict를 완화합니다.
- torchvision 0.22에서 BasicSR의 `functional_tensor` import error가 발생하면,
  설치된 BasicSR의 import를 `torchvision.transforms.functional` 기준으로 조정해야 합니다.

## Citation

```bibtex
@article{min2025text,
  title={Text-Aware Image Restoration with Diffusion Models},
  author={Min, Jaewon and Kim, Jin Hyeon and Cho, Paul Hyunbin and Lee, Jaeeun and Park, Jihye and Park, Minkyu and Kim, Sangpil and Park, Hyunhee and Kim, Seungryong},
  journal={arXiv preprint arXiv:2506.09993},
  year={2025}
}
```
