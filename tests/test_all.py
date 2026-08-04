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