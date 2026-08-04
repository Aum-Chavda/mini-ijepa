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