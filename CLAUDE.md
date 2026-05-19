# TextAwareIR — Project Overview

## 프로젝트 구조

```
TextAwareIR/
├── TAIR/                  # Text-Aware Image Restoration (ICLR 2026)
│   ├── terediff/          # TeReDiff 메인 모델
│   ├── testr/             # Text Spotting 모듈
│   ├── detectron2/        # Text Spotting backbone
│   ├── configs/           # train/val YAML configs
│   ├── run_script/        # 학습/평가 실행 스크립트
│   ├── train.py
│   ├── val.py
│   └── initialize.py
├── DiffBIR/               # DiffBIR (Motionblur RL 학습 backbone)
│   ├── diffbir/           # 모델 코드
│   ├── inputs/            # 입력 이미지
│   ├── results/           # 출력 결과
│   ├── weights/           # 사전학습 가중치
│   └── inference.py
├── SA-Text/               # SA-Text 학습 데이터셋 (HuggingFace parquet, 12GB)
├── SA-Text-test/          # SA-Text 테스트셋
└── SA-Text_Dataset/       # 데이터셋 구축 파이프라인 코드
```

## 연구 프로젝트

| 프로젝트 | 설명 | Skill |
|----------|------|-------|
| **TAIR** | Text-Aware Image Restoration (ICLR 2026) | `/TAIR` |
| **Motionblur** | DiffBIR 기반 RL fine-tuning (OCR reward) | `/Motionblur` |

## 환경 설정

### diffbir_bw (Blackwell GPU용 — RTX PRO 6000 sm_120)

```bash
conda activate diffbir_bw
ATTN_MODE=sdp python DiffBIR/inference.py --input <input_dir> --output <output_dir>
```

| 패키지 | 버전 | 비고 |
|--------|------|------|
| torch | 2.7.0+cu128 | Blackwell sm_120 지원 |
| xformers | 0.0.30 | ATTN_MODE=sdp로 우회 |
| transformers | 4.37.2 | LLaVA captioner 호환 |
| accelerate | 0.28.0 | |
| peft | 0.10.0 | LoRA |
| wandb | 0.27.0 | 학습 로깅 |
| pyiqa | 0.1.15 | MUSIQ reward |
| paddleocr | 3.5.0 | OCR reward |

> **주의:** xformers는 sm_120 미지원. `ATTN_MODE=sdp` 환경변수 필수.

### 일반 GPU (A100, RTX 3090, RTX 4090 등)

```bash
conda activate diffbir
python DiffBIR/inference.py --input <input_dir> --output <output_dir>
```

## 클러스터 파티션

| 파티션 | GPU | sm | QOS |
|--------|-----|----|-----|
| `suma_pro6000` | RTX PRO 6000 Blackwell | 120 | pro6000_qos |
| `asus_pro6000` | RTX PRO 6000 Blackwell | 120 | pro6000_qos |
| `suma_a100` | A100 | 80 | a100_qos (권한 필요) |
| `base_suma_rtx3090` | RTX 3090 | 86 | base_qos |

## 알려진 패치사항

- `DiffBIR/diffbir/sampler/edm_sampler.py` — `torch.Tuple` → `typing.Tuple` (PyTorch 2.x 호환)
- `basicsr/data/degradations.py` — `torchvision.transforms.functional_tensor` → `functional` (torchvision 0.22 호환)
