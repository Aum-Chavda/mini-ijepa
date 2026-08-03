# Mini I-JEPA — Image World Model from Scratch

Reproducing the core idea of [I-JEPA (Assran et al., CVPR 2023)](https://arxiv.org/abs/2301.08243) from scratch on a GTX 1650 Ti (4 GB VRAM).

A context encoder predicts masked patch embeddings using an EMA target encoder — no pixel reconstruction, no hand-crafted augmentations. Pure latent-space self-supervised learning.

> **Part of a 5-project robotics portfolio** targeting perception and manipulation roles at Sereact, Outsight, and NEURA Robotics.

---

## Why JEPA?

Most self-supervised vision models either reconstruct pixels (MAE) or contrast augmented views (SimCLR, DINO). Both have limitations:

- **Pixel reconstruction** wastes capacity on irrelevant low-level details (texture, lighting) that do not help semantic understanding.
- **Contrastive learning** relies on carefully engineered augmentations to define what "same" means — a human prior baked into the training signal.

JEPA takes a different path: predict **representations** of missing image regions, not pixels. The model is forced to learn *what is there* (semantics), not *exactly how it looks* (appearance).

This connects directly to LeCun's broader thesis that a world model should operate in abstract latent space, not pixel space — making JEPA a foundational architecture for robotics perception.

---

## My Background on This

Before building, I wrote a critical seminar paper at FAU Erlangen-Nürnberg reviewing **LeWorldModel** (Maes et al., 2026), a stable end-to-end JEPA trained from pixels. That review covered JEPA's theoretical grounding, its advantages over RSSM-style world models, and its limitations.

This project is the implementation side of that theoretical work: verify the intuitions experimentally on constrained hardware.

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GTX 1650 Ti, 4 GB VRAM |
| OS | Windows 11 |
| CUDA | 12.8 (cu128) |
| Python | 3.11.9 via `uv` |

---

## Architecture

```
Input Image (3 × H × W)
        │
        ▼
   Patch Embed (ViT-Tiny, frozen after warmup)
        │
   ┌────┴────────────────────┐
   │                         │
Context patches         Target patches
(unmasked)              (masked — labels)
   │                         │
Context Encoder          EMA copy of
(ViT-Tiny)               Context Encoder
   │                         │
   ▼                         ▼
Context repr.           Target repr.
   │                    (stop gradient)
   └──────► Predictor ──────►│
            (narrow MLP/     │
             Transformer)    │
                  │          │
                  ▼          ▼
               L2 loss in embedding space
```

**Key design choices:**
- No decoder, no pixel loss — prediction happens entirely in embedding space
- EMA target encoder prevents representational collapse without negative pairs
- Multi-block masking: large target blocks (semantic scale), distributed context (informative)

---

## Project Phases

- [x] Phase 1 — Config + Abstract base + Blocks + Registry
- [ ] Phase 2 — Data pipeline (STL-10 / ImageNet-100)
- [ ] Phase 3 — Training loop (context encoder + EMA + predictor)
- [ ] Phase 4 — Linear probe evaluation + visualisation

---

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| Linear probe (STL-10) | TBD | frozen encoder, 1 epoch head |
| Training time / epoch | TBD | GTX 1650 Ti, batch=32 |
| Peak VRAM | TBD | target < 3.5 GB |

---

## Literature Review

| Paper | Venue | Role |
|-------|-------|------|
| LeCun (2022) — [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) | OpenReview | Motivation: JEPA as world model |
| He et al. (2022) — [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) | CVPR 2022 | Contrast: pixel reconstruction baseline |
| Assran et al. (2023) — [I-JEPA](https://arxiv.org/abs/2301.08243) | CVPR 2023 | **Primary paper implemented** |
| Bardes et al. (2024) — [V-JEPA](https://openreview.net/forum?id=WFYbBOEOtv) | ICLR 2024 | Extension: temporal world models |
| Chavda (2026) — LeWorldModel: A Critical Review | FAU Seminar | Prior theoretical critique (author) |

---

## Project Structure

```
mini-ijepa/
├── src/
│   ├── models/
│   │   ├── base.py          # BaseEncoder + BasePredictor ABC
│   │   ├── blocks.py        # PatchEmbed, TransformerBlock
│   │   ├── jepa.py          # MiniIJEPA: context enc + EMA + predictor
│   │   └── registry.py      # ModelRegistry factory
│   ├── data/
│   │   └── dataset.py       # MaskingDataset + build_dataloader
│   ├── training/
│   │   ├── trainer.py       # Trainer + EMA update logic
│   │   ├── callbacks.py     # Callback Protocol + CheckpointSaver
│   │   └── metrics.py       # MetricTracker (loss, linear probe acc)
│   └── utils/
│       ├── config.py        # JEPAConfig dataclass
│       ├── masking.py       # MultiBlockMaskGenerator
│       └── visualize.py     # patch attention maps, loss curves
├── tests/
│   └── test_models.py
├── configs/
│   └── default.yaml
├── main.py
├── pyproject.toml
└── README.md
```

---

## Setup

```bash
uv init mini-ijepa
cd mini-ijepa
uv python pin 3.11
uv venv
uv add torch torchvision --default-index https://download.pytorch.org/whl/cu128
uv add numpy pillow pyyaml timm
uv add pytest --dev
```

---

## Key Engineering Lessons (updated as project progresses)

- TBD after Phase 1

---

## Related Projects

| Project | What | Status |
|---------|------|--------|
| [bev-cnn](https://github.com/Aum-Chavda/bev-cnn) | ResNet BEV feature extractor, 82.4% CIFAR-10 | ✅ Complete |
| [depth-anything-v2-reproduction](https://github.com/Aum-Chavda/depth-anything-v2-reproduction) | Depth Anything V2 on KITTI, AbsRel 0.1287 | 🔄 In progress |
| mini-ijepa | I-JEPA world model from scratch | 🚧 This repo |