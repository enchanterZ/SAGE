# SAGE: Semantic-Aware Active Learning with Adaptive Budget Scheduling

> Official PyTorch implementation (NeurIPS submission)

## Overview

SAGE is a budget-constrained active learning framework for fine-grained image classification. It jointly optimizes:

1. **Semantic Utility Augmentation** — generates and filters augmented views via CLIP semantic consistency
2. **Learned Query Utility** — ranks unlabeled samples with a learned head combining uncertainty, diversity, and semantic alignment
3. **Adaptive Budget Scheduling** — an RL (PPO-Clip) scheduler that dynamically allocates augmentation strength, query size, and diversity preference across rounds

## Framework

```
┌─────────────────────────────────────────────────────┐
│                    SAGE Framework                   │
│                                                     │
│  ┌──────────┐   α_t,b_t,p_t   ┌─────────────────┐  │
│  │    RL    │ ──────────────▶ │ Augmentation    │  │
│  │Scheduler │                 │ (Sec 3.2)       │  │
│  │ (PPO)    │ ◀── reward r_t  └────────┬────────┘  │
│  └──────────┘                          │ L_sup      │
│                                        ▼            │
│                               ┌─────────────────┐  │
│                               │  Query Utility  │  │
│                               │   (Sec 3.3)     │  │
│                               └────────┬────────┘  │
│                                        │ Q_t        │
│                                        ▼            │
│                               ┌─────────────────┐  │
│                               │     Oracle      │  │
│                               │   Annotation    │  │
│                               └─────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Requirements

```
python >= 3.9
torch >= 2.0
torchvision
transformers
diffusers
stable-baselines3
open_clip_torch
tqdm
pyyaml
```

Install:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Supported datasets: CUB-200-2011, Stanford Cars, FGVC-Aircraft, Oxford Pets.

```
data/
├── CUB_200_2011/
│   ├── images/
│   └── ...
└── stanford_cars/
    └── ...
```

## Quick Start

```bash
# Train SAGE on CUB-200 with budget B=1000
python scripts/train.py \
    --config configs/cub200.yaml \
    --budget 1000 \
    --rounds 10 \
    --seed 42

# Evaluate
python scripts/evaluate.py \
    --config configs/cub200.yaml \
    --checkpoint outputs/sage_cub200.pt
```

## Configuration

See `configs/cub200.yaml` for all hyperparameters.

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `budget` | 1000 | Total annotation budget B |
| `rounds` | 10 | Number of active learning rounds |
| `M_t` | 5 | Augmentation candidates per sample |
| `lambda_aug` | 0.5 | Augmentation loss weight |
| `beta` | 0.01 | Budget consumption penalty |
| `eta` | 0.1 | Augmentation inefficiency penalty |

## Project Structure

```
SAGE/
├── sage/
│   ├── __init__.py
│   ├── model.py          # Task classifier (CLIP backbone + head)
│   ├── augmentation.py   # Semantic Utility Augmentation (Sec 3.2)
│   ├── query.py          # Learned Query Utility (Sec 3.3)
│   ├── scheduler.py      # RL Budget Scheduler (Sec 3.4)
│   ├── trainer.py        # Active learning loop
│   └── utils.py          # CLIP embeddings, metrics
├── configs/
│   ├── cub200.yaml
│   └── stanford_cars.yaml
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@inproceedings{sage2026,
  title     = {SAGE: Semantic-Aware Active Learning with Adaptive Budget Scheduling},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
