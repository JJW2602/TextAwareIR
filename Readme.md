# TextAwareIR

Text-Aware Image Restoration 연구 모노레포. DiffBIR 기반 복원, SA-Text 데이터셋 큐레이션 파이프라인, OCR reward 기반 RL fine-tuning을 한 저장소에서 관리합니다.

---

## Repository Layout

```
TextAwareIR/
├── TAIR/                # TeReDiff / TAIR baseline 코드
├── DiffBIR/             # DiffBIR baseline 코드
├── TAIRL/               # TAIRL 코드 작업 공간
├── Results/
│   ├── Baselines/
│   │   ├── TAIR/        # TAIR baseline 결과
│   │   ├── DiffBIR/     # DiffBIR baseline 결과
│   │   └── Compare/     # baseline 비교 eval / visualization
│   └── TAIRL/           # TAIRL 결과
├── Slurm/
│   ├── Baselines/
│   │   ├── TAIR/        # TAIR baseline 실행 스크립트
│   │   └── DiffBIR/     # DiffBIR baseline 실행 스크립트
│   └── TAIRL/           # TAIRL 실행 스크립트
├── Dataset/
│   ├── SA-Text/
│   ├── SA-Text-test/
│   └── gopro/
└── Dataset_pipeline/
    └── SA-Text_Dataset/
```

---

## Conda Environments — 한눈에

| 워크플로우 | 추천 환경 | Python | 비고 |
|---|---|---|---|
| **DiffBIR inference** (Ampere/Ada: A100, RTX 3090/4090, A6000 등) | `diffbir` | 3.10 | 표준 진입점 |
| **DiffBIR inference** (Blackwell: RTX PRO 6000 sm_120) | `diffbir_bw` | 3.10 | `ATTN_MODE=sdp` 필수 |
| **SA-Text dataset pipeline** | `dataset_curation` | 3.10 | Bridge spotter + 2× VLM 통합 단일 env |
| **TAIRL — DDPO / GRPO LoRA fine-tuning** | `diffbir` (학습) + `dataset_curation` (reward subprocess) | 3.10 | 두 env가 step마다 함께 호출됨 |

> 환경 위치: `/home/james2602/miniconda3/envs/{diffbir,diffbir_bw,dataset_curation,tair}`

---

## 1. DiffBIR Inference

DiffBIR으로 저화질 이미지 → 복원 이미지를 생성합니다.

### 1-A. 단일 GPU (수동 실행)

```bash
# Ampere / Ada GPU (A100, RTX 3090, RTX 4090, A6000 ...)
conda activate diffbir
python DiffBIR/inference.py --input <input_dir> --output <output_dir>

# Blackwell GPU (RTX PRO 6000, sm_120)
conda activate diffbir_bw
ATTN_MODE=sdp python DiffBIR/inference.py --input <input_dir> --output <output_dir>
```

> **왜 Blackwell은 별도 env가 필요한가**
> `diffbir`의 `xformers`/`flash-attn` 바이너리는 sm_120 커널을 포함하지 않습니다. `diffbir_bw`는 torch 2.7 + cu128로 빌드되어 있고, attention은 PyTorch SDPA로 우회합니다(`ATTN_MODE=sdp`).

### 1-B. SA-Text test 멀티 GPU 배치 (slurm)

`Slurm/Baselines/DiffBIR/` 에 SA-Text test set을 chunk 단위로 분산 처리하는 스크립트가 있습니다.

```bash
# sbatch (suma_a6000 파티션, 2-GPU array)
sbatch Slurm/Baselines/DiffBIR/infer_sa_text_lv2_2gpu.slurm

# 또는 인터랙티브 (2-GPU 노드 위에서)
GPU_IDS=0,1 bash Slurm/Baselines/DiffBIR/infer_sa_text_lv2_2gpu.sh
```

기본 동작
- 사용 env: `diffbir` (`PYTHON_BIN=/home/james2602/miniconda3/envs/diffbir/bin/python`)
- 입력: `Dataset/SA-Text-test/data/test-00000-of-00001.parquet`
- 출력: `Results/Baselines/DiffBIR/sa_text_test_lv2_2gpu/chunk_{0,1}/`
- 주요 env vars: `SA_TEXT_LEVEL`, `CHUNK_SIZE`, `DIFFBIR_UPSCALE`, `DIFFBIR_STEPS`, `DIFFBIR_CFG_SCALE`, `DIFFBIR_CAPTIONER`, `DIFFBIR_PRECISION`

### 1-C. SA-Text test 평가

복원 결과 위에 Bridge spotter로 OCR 예측을 뽑고, GT와 비교해 텍스트 메트릭을 계산합니다. **두 환경이 모두 필요**합니다:

```bash
sbatch Slurm/Baselines/DiffBIR/eval_sa_text_lv2_2gpu.slurm
```

| 역할 | 환경 | 이유 |
|---|---|---|
| OCR 예측 (Bridge spotter) | `dataset_curation` | detectron2 + adet 빌드 보유 |
| 평가 메트릭 계산 | `diffbir` | DiffBIR 평가 코드 의존성 |

---

## 2. SA-Text Dataset Pipeline

SA-1B → text-rich crop → 2× VLM 합의 → blur 필터링 → 최종 dataset JSON 까지 14단계로 큐레이션합니다.

### Env 셋업 (최초 1회)

```bash
conda create -n dataset_curation python=3.10 -y
conda activate dataset_curation

# PyTorch (GPU 세대에 맞춰 택1)
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124           # Ampere/Ada
# 또는
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128           # Blackwell

# 공통 의존성
pip install opencv-python scipy timm shapely albumentations Polygon3 pandas tqdm pyyaml
pip install transformers==4.51.3 accelerate scikit-learn qwen_vl_utils pytz
pip install flash-attn --no-build-isolation                      # Ampere/Ada 만

# Bridge spotter 빌드 (한 env 안에서)
pip install setuptools==59.5.0
cd Dataset_pipeline/SA-Text_Dataset/Bridging-Text-Spotting/detectron2 && python setup.py build develop && cd ..
python setup.py build develop && cd ../..
```

### 실행

```bash
conda activate dataset_curation

# config.yaml 의 sa1b_base_dir / bridge_repo_dir / output 경로를 본 머신에 맞게 수정
python Dataset_pipeline/SA-Text_Dataset/dataset_curation/main_pipeline.py \
    --config Dataset_pipeline/SA-Text_Dataset/dataset_curation/config.yaml \
    --sa1b_subfolder "sa_000000" \
    --output_suffix "_sa_000000"
```

### 14 단계 흐름 (요약)

| # | 단계 | 출력 |
|---|---|---|
| 1 | Bridge Stage 1 detection (원본) | `bridge_stage1_results*.json` |
| 2–3 | Crop region 정의 & 512px crop 생성 | `cropped_images/*.jpg` |
| 4 | Bridge Stage 2 detection (crop) | `bridge_stage2_raw_results*.json` |
| 4.5 | 중복 box 제거 (IoU≥0.9) | `bridge_stage2_filtered*.json` |
| 5–6 | VLM1 (OVIS), VLM2 (Qwen) 텍스트 인식 | `OVIS_raw*.json`, `Qwen_raw*.json` |
| 7–9 | 빈 결과 필터 → 두 VLM 결과 병합 | `vlm_combined*.json` |
| 10–11 | 두 VLM 완전 합의 이미지 추출 | `agreed_*.json/.txt` |
| 12 | Qwen blur 평가 | `blur_assessment*.csv` |
| 13 | blur 태깅 & non-blurry 필터 | `tagged_*`, `restoration_*` (intermediate) |
| 14 | 최종 포맷팅 | `full_dataset*.json`, `restoration_dataset*.json` |

> 중간 단계부터 재실행: `--start_from <stage>` / 단일 stage만: `--run_only_stage <stage>`

Stage 이름은 `main_pipeline.py:36-40` 의 `valid_stages` 리스트 참고.

---

## 3. TAIRL — RL Fine-tuning (DDPO / GRPO + LoRA)

DiffBIR controlnet에 **LoRA**를 주입하고, **Bridge spotter 기반 OCR reward**로 DDPO 또는 GRPO로 fine-tuning. 단일 진입점 `TAIRL/train_grpo_ddpo_lora.py`가 `train.algorithm` 스위치로 두 알고리즘을 처리합니다.

### Reward 정의

이미지마다 GT instance 수 `A`, IoU≥thr 매칭 `M`일 때 (`tairl/reward.py`):

```
matched_mean_reward = mean over matched pairs of  max(1 − lev/len(gt_text), 0)
missed              = A − M
final_reward        = matched_mean_reward − miss_penalty × missed
final_reward_norm   = matched_mean_reward − missed/A                  # ∈ [−1, 1]
```

config의 `reward.variant`로 `final_reward` 또는 `final_reward_norm` 선택.

### 두 env가 step마다 함께 호출되는 구조

```
[diffbir env]   rollout (DiffBIR LoRA로 LQ → restored, group_size=6 for GRPO)
       ↓ 이미지 dump (work_dir/step_*/images/)
[dataset_curation env]   bridge subprocess (detectron2 + Bridge spotter inference)
       ↓ bbox/rec JSON
[diffbir env]   compose_image_reward → PPO clip + KL → AdamW
```

config 의 `reward.bridge_env_python` 가 subprocess 호출 시 사용되는 두 번째 interpreter 경로입니다.

### 디렉터리

```
TAIRL/
├── train_grpo_ddpo_lora.py   # 진입점 (grpo / ddpo 둘 다)
├── configs/
│   ├── ddpo_lora.yaml         # DDPO baseline
│   ├── grpo_lora_g6.yaml      # GRPO (group_size=6)
│   └── hparam_search_small.yaml
├── tairl/
│   ├── data.py                # SATextParquetDataset
│   ├── ddpo_sampler.py        # spaced sampler with logprob
│   ├── lora.py                # LoRA inject / state_dict
│   └── reward.py              # BridgeReward, compose_image_reward
├── slurm/
│   ├── grpo_lora_g6.slurm     # GRPO 단일 run
│   └── ddpo_lora_hparam.slurm # DDPO LoRA hparam sweep (array 0-10)
└── train_data/sa_text_train_10000.jsonl
```

### 실행

```bash
# 단일 run
sbatch Slurm/TAIRL/grpo_lora_g6.slurm
# 또는
bash TAIRL/run_grpo_g6.sh

# DDPO hparam sweep (11개 trial array, baseline=trial 3)
sbatch TAIRL/slurm/ddpo_lora_hparam.slurm
# 또는
bash TAIRL/run_hparam_search.sh
```

DDPO sweep는 baseline `r8 / lr3e-5 / clip0.1 / kl0.02 / norm reward / miss1.0 / fp0.25` 에서 한 번에 하나씩 변경하는 OAT(one-factor-at-a-time) 구성:

| 축 | trial id |
|---|---|
| Learning rate (lr=1e-5 / 3e-5 / 1e-4 at r=4) | 0, 1, 2 |
| LoRA rank (4 / 8 / 16) | 1, 3, 6 |
| PPO clip (0.1 / 0.2) | 3, 4 |
| KL coef (0.02 / 0) | 3, 5 |
| Reward variant (norm / raw) | 3, 7 |
| Miss penalty (1.0 / 0.5) | 3, 8 |
| FP penalty (0.0 / 0.25 / 0.5) | 9, 3, 10 |

### 결과 위치

```
Results/TAIRL/
├── ddpo_lora/<run_name>/
└── grpo_lora_g6/<run_name>/
       metrics.jsonl, resolved_config.yaml, reward_work/, wandb/
```

`wandb.group=grpo` 또는 `ddpo_hparam`으로 묶이므로 wandb UI에서 한 번에 비교 가능.

---

## 클러스터 파티션 참고

| 파티션 | GPU | sm | QOS |
|---|---|---|---|
| `suma_pro6000`, `asus_pro6000` | RTX PRO 6000 Blackwell | 120 | `pro6000_qos` |
| `suma_a100` | A100 | 80 | `a100_qos` (권한 필요) |
| `suma_a6000`, `gigabyte_a6000` | A6000 | 86 | `big_qos` |
| `suma_rtx4090` | RTX 4090 | 89 | `big_qos` |
| `base_suma_rtx3090`, `big_suma_rtx3090`, `dell_rtx3090` | RTX 3090 | 86 | `base_qos` |

GPU 세대별 env 매칭
- **Blackwell (sm_120)** → `diffbir_bw` (DiffBIR inference / RL), `dataset_curation`(cu128 빌드)
- **그 외 (sm_80–89)** → `diffbir` (DiffBIR), `dataset_curation`(cu124 빌드)

---

## 알려진 패치

- `DiffBIR/diffbir/sampler/edm_sampler.py` — `torch.Tuple` → `typing.Tuple` (PyTorch 2.x 호환)
- `basicsr/data/degradations.py` — `torchvision.transforms.functional_tensor` → `functional` (torchvision 0.22 호환)
- `Dataset_pipeline/SA-Text_Dataset/Bridging-Text-Spotting/setup.py` — `Pillow==9.1` → `Pillow>=9.1` (scikit-image 충돌 회피)

---

## 인용

```bibtex
@article{min2025text,
  title={Text-Aware Image Restoration with Diffusion Models},
  author={Min, Jaewon and Kim, Jin Hyeon and Cho, Paul Hyunbin and Lee, Jaeeun and Park, Jihye and Park, Minkyu and Kim, Sangpil and Park, Hyunhee and Kim, Seungryong},
  journal={arXiv preprint arXiv:2506.09993},
  year={2025}
}
```
