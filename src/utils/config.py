from dataclasses import dataclass


@dataclass
class JEPAConfig:
    # ── Image ──────────────────────────────────────────────────────────
    image_size: int = 224
    patch_size: int = 16          # num_patches = (image_size // patch_size) ** 2

    # ── Context Encoder (ViT-Tiny) ─────────────────────────────────────
    embed_dim: int = 192
    encoder_depth: int = 12
    encoder_heads: int = 3        # embed_dim must be divisible by encoder_heads

    # ── Predictor ──────────────────────────────────────────────────────
    predictor_embed_dim: int = 96
    predictor_depth: int = 4

    # ── Masking ────────────────────────────────────────────────────────
    num_target_blocks: int = 4
    target_scale: tuple = (0.15, 0.2)
    target_aspect_ratio: tuple = (0.75, 1.5)
    context_scale: float = 0.5

    # ── EMA ────────────────────────────────────────────────────────────
    ema_momentum: float = 0.996

    # ── Training ───────────────────────────────────────────────────────
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 100
    warmup_epochs: int = 10
    device: str = "cuda"

    # ── Paths ──────────────────────────────────────────────────────────
    data_dir: str = "data/"
    checkpoint_dir: str = "checkpoints/"

    # ── Derived ────────────────────────────────────────────────────────
    @property
    def num_patches(self) -> int:
        """Total number of patches in one image."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_context_patches(self) -> int:
        """Approximate number of context (visible) patches."""
        return int(self.num_patches * self.context_scale)