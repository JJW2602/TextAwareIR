# TextAwareIR

Text-Aware Image Restoration 연구 모노레포. DiffBIR 기반 복원, SA-Text 데이터셋 큐레이션 파이프라인, OCR reward 기반 RL fine-tuning을 한 저장소에서 관리합니다.

---

## Repository Layout

```
TextAwareIR/
├── DiffBIR/             # DiffBIR 복원 모델 (inference / 향후 RL backbone)
│   ├── diffbir/             # 모델 코드
│   ├── inference.py         # 단일 GPU inference 진입점
│   ├── slurm/               # SA-Text test 멀티-GPU inference & eval 스크립트
│   ├── inputs/  results/  weights/
│   └── requirements.txt
├── SA-Text_Dataset/     # SA-1B → SA-Text 데이터셋 구축 파이프라인 (14-stage)
│   ├── dataset_curation/
│   │   ├── main_pipeline.py
│   │   ├── config.yaml
│   │   └── src/             # bridge_runner, cropping, vlm_processing, ...
│   └── Bridging-Text-Spotting/   # text spotter (Bridge + detectron2 + DiG)
├── TAIR/                # TeReDiff (ICLR 2026) 학습/추론 코드 (참고용)
├── SA-Text/             # HuggingFace parquet 학습셋 (~12 GB, gitignored)
└── SA-Text-test/        # SA-Text 테스트셋 (parquet)
```

---

## Conda Environments — 한눈에

| 워크플로우 | 추천 환경 | Python | 비고 |
|---|---|---|---|
| **DiffBIR inference** (Ampere/Ada: A100, RTX 3090/4090, A6000 등) | `diffbir` | 3.10 | 표준 진입점 |
| **DiffBIR inference** (Blackwell: RTX PRO 6000 sm_120) | `diffbir_bw` | 3.10 | `ATTN_MODE=sdp` 필수 |
| **SA-Text dataset pipeline** | `dataset_curation` | 3.10 | Bridge spotter + 2× VLM 통합 단일 env |
| **RL fine-tuning (WIP)** | `diffbir_bw` *(예정)* | 3.10 | DiffBIR backbone + OCR reward, 아직 코드 미커밋 |

> 환경 위치: `/home/james2602/miniconda3/envs/{diffbir,diffbir_bw,dataset_curation}`

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

`DiffBIR/slurm/` 에 SA-Text test set을 chunk 단위로 분산 처리하는 스크립트가 있습니다.

```bash
# sbatch (suma_a6000 파티션, 2-GPU array)
sbatch DiffBIR/slurm/infer_sa_text_lv2_2gpu.slurm

# 또는 인터랙티브 (2-GPU 노드 위에서)
GPU_IDS=0,1 bash DiffBIR/slurm/infer_sa_text_lv2_2gpu.sh
```

기본 동작
- 사용 env: `diffbir` (`PYTHON_BIN=/home/james2602/miniconda3/envs/diffbir/bin/python`)
- 입력: `SA-Text-test/data/test-00000-of-00001.parquet`
- 출력: `DiffBIR/results/sa_text_test/lv2_2gpu/chunk_{0,1}/`
- 주요 env vars: `SA_TEXT_LEVEL`, `CHUNK_SIZE`, `DIFFBIR_UPSCALE`, `DIFFBIR_STEPS`, `DIFFBIR_CFG_SCALE`, `DIFFBIR_CAPTIONER`, `DIFFBIR_PRECISION`

### 1-C. SA-Text test 평가

복원 결과 위에 Bridge spotter로 OCR 예측을 뽑고, GT와 비교해 텍스트 메트릭을 계산합니다. **두 환경이 모두 필요**합니다:

```bash
sbatch DiffBIR/slurm/eval_sa_text_lv2_2gpu.slurm
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
cd SA-Text_Dataset/Bridging-Text-Spotting/detectron2 && python setup.py build develop && cd ..
python setup.py build develop && cd ../..
```

### 실행

```bash
conda activate dataset_curation

# config.yaml 의 sa1b_base_dir / bridge_repo_dir / output 경로를 본 머신에 맞게 수정
python SA-Text_Dataset/dataset_curation/main_pipeline.py \
    --config SA-Text_Dataset/dataset_curation/config.yaml \
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

## 3. RL Fine-tuning *(Work in Progress)*

DiffBIR backbone에 **OCR reward(PaddleOCR)** + **MUSIQ reward(pyiqa)** 를 결합한 RL fine-tuning. 코드는 아직 본 저장소에 커밋되지 않았습니다.

### 계획된 환경

| 항목 | 값 |
|---|---|
| Env | **`diffbir_bw`** (Blackwell GPU 학습 가정) |
| PyTorch | 2.7.0 + cu128 |
| Attention | `ATTN_MODE=sdp` (xformers 우회) |
| Reward — OCR | `paddleocr==3.5.0` |
| Reward — Perceptual | `pyiqa==0.1.15` (MUSIQ) |
| LoRA | `peft==0.10.0` |
| Logging | `wandb==0.27.0` |
| Accelerate | `0.28.0` |

### 사용 예 (예정)

```bash
conda activate diffbir_bw
ATTN_MODE=sdp python DiffBIR/train_rl.py --config <todo>.yaml   # 코드 추가 시 업데이트
```

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
- `SA-Text_Dataset/Bridging-Text-Spotting/setup.py` — `Pillow==9.1` → `Pillow>=9.1` (scikit-image 충돌 회피)

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
