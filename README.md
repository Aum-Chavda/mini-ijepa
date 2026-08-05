# Mini I-JEPA — Image World Model from Scratch

Reproducing the core idea of [I-JEPA (Assran et al., CVPR 2023)](https://arxiv.org/abs/2301.08243) from scratch on a GTX 1650 Ti (4 GB VRAM).

A context encoder predicts masked patch embeddings using an EMA target encoder — no pixel reconstruction, no hand-crafted augmentations. Pure latent-space self-supervised learning.

> **Part of a 5-project ML and Robotics portfolio**

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

## How It Works — Full Pipeline

### Step 1 — One Image Enters

A single RGB image of size 224×224 pixels enters the pipeline.

```
Image: [3, 224, 224]
= 3 colour channels × 224 rows × 224 columns
= 150,528 raw numbers
```

### Step 2 — Cut Into Patches

The image is divided into a 14×14 grid of non-overlapping 16×16 pixel tiles.

```
224 ÷ 16 = 14 tiles per side
14 × 14  = 196 patches total

┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
        ... 14 rows × 14 cols = 196 patches
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Each patch = 16 × 16 × 3 = 768 raw pixel values
```

### Step 3 — Embed Each Patch (PatchEmbed)

A single Conv2d projects each patch from 768 raw pixel values down to 192 numbers — a compact learned representation called an **embedding**.

```
768 raw pixels  →  Conv2d(kernel=16, stride=16)  →  192 embedding values

196 patches × 192 values = tensor [196, 192]
```

**What is an embedding?**
The 192 numbers are not pixel values. They are coordinates in a 192-dimensional space that capture the *meaning* of that patch. Two patches of sky will have similar embeddings. A patch of fur and a patch of sky will be far apart. The model learns these coordinates during training.

### Step 4 — Add Position Information

The Transformer has no built-in sense of order — it sees 196 patches as an unordered set. So we add a learnable **positional embedding** to each patch, one per position:

```
patch_0_embedding + position_0_vector  →  [192 numbers]
patch_1_embedding + position_1_vector  →  [192 numbers]
...
patch_195_embedding + position_195_vector  →  [192 numbers]

Still [196, 192] — same shape, but now each patch knows where it lives.
```

### Step 5 — Masking (Split Into Context and Target)

The 196 patches are split into two groups:

```
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│C │  │C │  │C │  │C │  │C │  │C │  │C │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │C │  │T │T │T │T │  │C │  │C │  │  │C │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│C │  │  │T │T │T │T │  │  │C │  │C │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │C │  │T │T │T │T │  │C │  │  │  │C │  │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

C = context (visible, ~98 patches, spread across image)
T = target  (masked, 4 large contiguous blocks, must be predicted)
```

Why large blocks for targets? Predicting a single patch is trivial — just interpolate neighbours. Predicting a large contiguous region forces the model to understand semantic content.

### Step 6 — Context Encoder (With Gradients)

Only the **C patches** pass through 12 Transformer blocks. Each block lets every patch attend to every other patch and update its embedding.

```
Input:   [98, 192]   ← 98 context patches, 192 dims each
          │
    TransformerBlock × 12
    (each patch looks at all other patches, updates itself)
          │
Output:  [98, 192]   ← same shape, completely different numbers
                        now each patch carries global context
```

### Step 7 — Target Encoder (No Gradients, EMA)

An identical ViT processes **all 196 patches** but with no gradients. Its weights are never updated by backprop — only by EMA:

```
After every optimizer step:
target_weights = 0.996 × target_weights + 0.004 × context_weights

Meaning: target encoder slowly follows the context encoder
         always slightly behind, always stable
```

```
Input:   [196, 192]  ← all patches
          │
    VisionTransformer (no grad)
          │
Output:  [196, 192]
          │
    pick target patch positions
          │
         [N_target, 192]  ← these are the LABELS
```

### Step 8 — Predictor

The predictor takes context embeddings + learnable mask tokens (one per target position) and predicts what the target encoder would output:

```
context embeddings:  [98, 192]
mask tokens:         [N_target, 192]  ← learnable "predict here" vectors
          │
    concatenate → [98 + N_target, 192]
          │
    TransformerBlock × 4  (narrow, intentionally small)
          │
    take last N_target outputs
          │
Output:  [N_target, 192]  ← PREDICTIONS
```

### Step 9 — Loss

```
Predicted:  [N_target, 192]   ← what the predictor thinks
Target:     [N_target, 192]   ← what the target encoder actually produced

Loss = mean( (predicted − target)² )   ← L2 / MSE

Backprop updates: context encoder + predictor
EMA updates:      target encoder (after optimizer step)
Target encoder:   never receives gradients directly
```

---

## Full Pipeline Diagram

```
ONE IMAGE [3, 224, 224]
         │
         ▼
   Cut into 196 patches of 16×16 pixels
   [196, 768]
         │
         ▼
   PatchEmbed — Conv2d projects 768 → 192
   [196, 192]
         │
         ▼
   Add positional embeddings
   [196, 192]  ← patches now know where they are
         │
         ├─────────────────────────────────────┐
         │                                     │
   MASK AND SPLIT                              │
         │                                     │
   Context [98, 192]           All patches [196, 192]
   (spread across image)                       │
         │                                     │
         ▼                                     ▼
  Context Encoder                      Target Encoder
  ViT-Tiny, 12 blocks                  ViT-Tiny, 12 blocks
  gradients flow ✓                     NO gradients ✗
  updated by backprop                  updated by EMA only
         │                                     │
  [98, 192]                             [196, 192]
  rich context embeddings                      │
         │                              pick target positions
         │                                     │
         │                             [N_tgt, 192]  ← LABELS
         │                                     │
         ▼                                     │
     Predictor                                 │
     4 Transformer blocks                      │
     context + mask tokens → predictions       │
         │                                     │
  [N_tgt, 192]  ← PREDICTIONS                 │
         │                                     │
         └──────────► L2 Loss ◄────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
         backprop                EMA update
    (context encoder           (target encoder
      + predictor)              slowly follows)
```

---

## Why This Doesn't Collapse

If the encoder learned to output zeros everywhere, the loss would be zero too. Two mechanisms prevent this:

**EMA asymmetry** — the context and target encoders are never the same network. The target is always lagging behind. The context encoder cannot trivially copy the target — it has to predict a moving reference.

**Predictor bottleneck** — the predictor is intentionally small (4 blocks, 96 dims vs 192). It cannot memorise — it must generalise. The only way to predict accurately is to build truly semantic representations.

---

## Architecture

| Component | Type | Depth | Dim | Params |
|-----------|------|-------|-----|--------|
| Context Encoder | ViT-Tiny | 12 blocks | 192 | ~5.5M |
| Target Encoder | ViT-Tiny (EMA) | 12 blocks | 192 | ~5.5M |
| Predictor | Narrow Transformer | 4 blocks | 96 | ~0.4M |
| **Total trainable** | | | | **~5.9M** |

---

## Project Phases

- [x] Phase 1 — Config + ABCs + Blocks + Full JEPA model
- [x] Phase 2 — Data pipeline (STL-10)
- [ ] Phase 3 — Training loop + EMA + callbacks
- [ ] Phase 4 — Linear probe evaluation + visualisation

---

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| Linear probe (STL-10) | TBD | frozen encoder, 1 epoch head |
| Training time / epoch | TBD | GTX 1650 Ti, batch=32 |
| Peak VRAM | TBD | target < 3.5 GB |

---
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
│   │   ├── blocks.py        # PatchEmbed, MultiHeadSelfAttention, TransformerBlock
│   │   └── jepa.py          # VisionTransformer, Predictor, MiniIJEPA
│   ├── data/
│   │   └── dataset.py       # MaskingDataset + build_dataloader
│   ├── training/
│   │   ├── trainer.py       # Trainer + EMA update logic
│   │   ├── callbacks.py     # Callback Protocol + CheckpointSaver
│   │   └── metrics.py       # MetricTracker (loss, linear probe acc)
│   └── utils/
│       ├── config.py        # JEPAConfig dataclass
│       ├── masking.py       # MultiBlockMaskGenerator
│       └── visualize.py     # attention maps, loss curves
├── tests/
│   └── test_all.py          # single test file, grows with project
├── configs/
│   └── default.yaml
├── main.py
├── pyproject.toml
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Aum-Chavda/mini-ijepa.git
cd mini-ijepa
uv python pin 3.11
uv venv
uv sync
```

> Note: PyTorch CUDA index is pre-configured in `pyproject.toml`. `uv sync` will fetch the correct `cu128` build automatically.

---

## Related Projects

| Project | What | Status |
|---------|------|--------|
| [bev-cnn](https://github.com/Aum-Chavda/bev-cnn) | Bird's-Eye View CNN, 82%+ CIFAR-10 | ✅ Complete |
| [depth-anything-v2-reproduction](https://github.com/Aum-Chavda/depth-anything-v2-reproduction) | Depth Anything V2 on KITTI, AbsRel 0.1287 | 🔄 In progress |
| mini-ijepa | I-JEPA image world model from scratch | 🚧 This repo |