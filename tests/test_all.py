"""
Single test file for mini-ijepa.
Run with: uv run python tests/test_all.py
Each section tests one module. Add new tests below as we build.
"""

import sys
import torch

print(f"Python: {sys.version}")
print(f"Torch:  {torch.__version__}")
print(f"CUDA:   {torch.cuda.is_available()}")
print("-" * 40)

# ── Config ────────────────────────────────────────────────────────────
print("[1/2] Testing config...")
from src.utils.config import JEPAConfig

cfg = JEPAConfig()
assert cfg.num_patches == 196,         f"Expected 196, got {cfg.num_patches}"
assert cfg.num_context_patches == 98,  f"Expected 98, got {cfg.num_context_patches}"
assert cfg.embed_dim == 192,           f"Expected 192, got {cfg.embed_dim}"
print("      num_patches       :", cfg.num_patches)
print("      num_context_patches:", cfg.num_context_patches)
print("      embed_dim         :", cfg.embed_dim)
print("      PASSED ✓")

# ── Base ABCs ─────────────────────────────────────────────────────────
print("[2/2] Testing base ABCs...")
from src.models.base import BaseEncoder, BasePredictor
import torch.nn as nn

class BadEncoder(BaseEncoder):
    pass  # missing forward and embed_dim

try:
    _ = BadEncoder()
    print("      FAILED ✗ — ABC did not catch missing methods")
except TypeError:
    print("      ABC correctly blocked incomplete encoder ✓")
    print("      PASSED ✓")

print("-" * 40)
print("All tests passed.")

# ── Blocks ────────────────────────────────────────────────────────────
print("[3/3] Testing blocks...")
from src.models.blocks import PatchEmbed, MultiHeadSelfAttention, TransformerBlock

device = "cuda" if torch.cuda.is_available() else "cpu"

# PatchEmbed
x = torch.randn(2, 3, 224, 224).to(device)
patch_embed = PatchEmbed(image_size=224, patch_size=16, embed_dim=192).to(device)
out = patch_embed(x)
assert out.shape == (2, 196, 192), f"PatchEmbed shape wrong: {out.shape}"
print("      PatchEmbed output :", out.shape)

# MultiHeadSelfAttention
x = torch.randn(2, 196, 192).to(device)
attn = MultiHeadSelfAttention(embed_dim=192, num_heads=3).to(device)
out = attn(x)
assert out.shape == (2, 196, 192), f"MHSA shape wrong: {out.shape}"
print("      Attention output  :", out.shape)

# TransformerBlock
x = torch.randn(2, 196, 192).to(device)
block = TransformerBlock(embed_dim=192, num_heads=3).to(device)
out = block(x)
assert out.shape == (2, 196, 192), f"TransformerBlock shape wrong: {out.shape}"
print("      TransformerBlock  :", out.shape)
print("      PASSED ✓")

print("-" * 40)
print("All tests passed.")
# ── JEPA Model ────────────────────────────────────────────────────────
print("[4/4] Testing MiniIJEPA...")
from src.models.jepa import MiniIJEPA

cfg = JEPAConfig()
model = MiniIJEPA(cfg).to(device)

# dummy batch — 2 images
images = torch.randn(2, 3, 224, 224).to(device)

# dummy indices — first 98 patches as context, next 20 as target
context_indices = torch.arange(98).unsqueeze(0).expand(2, -1).to(device)
target_indices = torch.arange(98, 118).unsqueeze(0).expand(2, -1).to(device)

loss = model(images, context_indices, target_indices)
assert loss.item() > 0, "Loss should be positive"
print("      Loss              :", round(loss.item(), 4))

# test EMA update runs without error
model.update_target_encoder()
print("      EMA update        : OK")

# confirm target encoder has no gradients
for p in model.target_encoder.parameters():
    assert not p.requires_grad, "Target encoder should be frozen"
print("      Target frozen     : OK")
print("      PASSED ✓")

print("-" * 40)
print("All tests passed.")
# ── Masking ───────────────────────────────────────────────────────────
print("[5/5] Testing masking...")
from src.utils.masking import MultiBlockMaskGenerator

mask_gen = MultiBlockMaskGenerator(cfg)
ctx_idx, tgt_idx = mask_gen(batch_size=2)

print("      context_indices shape:", ctx_idx.shape)
print("      target_indices  shape:", tgt_idx.shape)

# context and target should not overlap for each sample
for i in range(2):
    ctx_set = set(x for x in ctx_idx[i].tolist() if x >= 0)
    tgt_set = set(x for x in tgt_idx[i].tolist() if x >= 0)
    overlap = ctx_set & tgt_set
    assert len(overlap) == 0, f"Sample {i} has overlap: {overlap}"

print("      No context/target overlap: OK")
assert ctx_idx.shape[0] == 2, "Batch size should be 2"
assert tgt_idx.shape[0] == 2, "Batch size should be 2"
print("      PASSED ✓")
# ── Dataset ───────────────────────────────────────────────────────────
print("[6/6] Testing dataset...")
from src.data.dataset import STL10JEPADataset, build_dataloader

# dataset will download STL-10 on first run (~2.6GB) — be patient
print("      Loading STL-10 (downloads if not cached)...")
dataset = STL10JEPADataset(cfg, split="unlabeled", download=True)
print(f"      Dataset size     : {len(dataset)}")

# test one sample
image, ctx_idx, tgt_idx = dataset[0]
print(f"      Image shape      : {image.shape}")
print(f"      Context indices  : {ctx_idx.shape}")
print(f"      Target indices   : {tgt_idx.shape}")

assert image.shape == (3, 224, 224), f"Wrong image shape: {image.shape}"
assert ctx_idx.shape[0] > 0, f"Context indices empty: {ctx_idx.shape}"
assert ctx_idx.shape[0] <= cfg.num_patches, f"Context too large: {ctx_idx.shape}"

# test dataloader — one batch
loader = build_dataloader(cfg, split="unlabeled")
images, ctx, tgt = next(iter(loader))
print(f"      Batch images     : {images.shape}")
print(f"      Batch context    : {ctx.shape}")
print(f"      Batch target     : {tgt.shape}")

assert images.shape == (cfg.batch_size, 3, 224, 224)
print("      PASSED ✓")
# ── Metrics ───────────────────────────────────────────────────────────
print("[7/7] Testing metrics...")
from src.training.metrics import AverageMeter, MetricTracker

# AverageMeter
meter = AverageMeter("loss")
meter.update(1.0, n=32)
meter.update(0.5, n=32)
assert abs(meter.avg - 0.75) < 1e-5, f"Wrong avg: {meter.avg}"
print(f"      AverageMeter avg : {meter.avg:.4f} (expected 0.7500)")

# MetricTracker
tracker = MetricTracker()
tracker.update(loss=1.0, batch_size=32)
tracker.update(loss=0.5, batch_size=32)
record = tracker.end_epoch(epoch=1, probe_acc=0.423)
assert record["loss"] == 0.75
assert record["probe_acc"] == 0.423
print(f"      Epoch record     : {record}")
print(f"      Best loss        : {tracker.best_loss()}")
print("      PASSED ✓")
# ── Callbacks ─────────────────────────────────────────────────────────
print("[8/8] Testing callbacks...")
from src.training.callbacks import (
    CallbackProtocol, CheckpointSaver, EarlyStopping, CallbackRunner
)

# CheckpointSaver — saves when loss improves
saver = CheckpointSaver(model=model, save_dir="checkpoints/test")
assert isinstance(saver, CallbackProtocol), "CheckpointSaver must match Protocol"

saver.on_epoch_end(epoch=1, metrics={"loss": 1.2})
saver.on_epoch_end(epoch=2, metrics={"loss": 0.9})   # should save
saver.on_epoch_end(epoch=3, metrics={"loss": 1.1})   # should not save
assert saver.best_loss == 0.9, f"Wrong best loss: {saver.best_loss}"
print(f"      CheckpointSaver best_loss: {saver.best_loss}")

# EarlyStopping
stopper = EarlyStopping(patience=3)
stopper.on_epoch_end(epoch=1, metrics={"loss": 1.0})
stopper.on_epoch_end(epoch=2, metrics={"loss": 1.1})
stopper.on_epoch_end(epoch=3, metrics={"loss": 1.2})
stopper.on_epoch_end(epoch=4, metrics={"loss": 1.3})  # patience exceeded
assert stopper.should_stop == True, "EarlyStopping should have triggered"
print(f"      EarlyStopping triggered  : {stopper.should_stop}")

# CallbackRunner
runner = CallbackRunner([saver, stopper])
runner.on_epoch_end(epoch=5, metrics={"loss": 0.5})
runner.on_train_end()
print("      CallbackRunner fan-out   : OK")
print("      PASSED ✓")