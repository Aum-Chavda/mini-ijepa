import torch
from src.utils.config import JEPAConfig
from src.models.jepa import MiniIJEPA
from src.data.dataset import build_dataloader
from src.training.trainer import Trainer
from src.training.callbacks import CheckpointSaver, EarlyStopping


def main():
    # ── Config ────────────────────────────────────────────────────────
    cfg = JEPAConfig(
        epochs=50,
        warmup_epochs=2,
        batch_size=32,
        encoder_depth=12,
        predictor_depth=4,
        lr=5e-5,
        weight_decay=0.05,
        device="cuda" if torch.cuda.is_available() else "cpu",
        data_dir="data/",
        checkpoint_dir="checkpoints/",
    )

    print("=" * 60)
    print("Mini I-JEPA Training")
    print("=" * 60)
    print(f"Device      : {cfg.device}")
    print(f"Epochs      : {cfg.epochs}")
    print(f"Batch size  : {cfg.batch_size}")
    print(f"Embed dim   : {cfg.embed_dim}")
    print(f"Num patches : {cfg.num_patches}")
    print("=" * 60)

    # ── Model ─────────────────────────────────────────────────────────
    model = MiniIJEPA(cfg)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")
    print(f"Frozen params    : {total_params - trainable_params:,}  (target encoder)")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────
    print("Building dataloader...")
    loader = build_dataloader(cfg, split="unlabeled")
    print(f"Dataset size : {len(loader.dataset):,} images")
    print(f"Batches/epoch: {len(loader):,}")
    print("=" * 60)

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        CheckpointSaver(model, save_dir=cfg.checkpoint_dir),
        EarlyStopping(patience=45),
    ]

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        loader=loader,
        cfg=cfg,
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────
    history = trainer.fit()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("Training complete.")
    best = min(r["loss"] for r in history)
    print(f"Best loss : {best:.4f}")
    print(f"Epochs run: {len(history)}")
    print("=" * 60)


if __name__ == "__main__":
    main()