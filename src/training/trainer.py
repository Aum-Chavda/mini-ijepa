from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.config import JEPAConfig
from src.models.jepa import MiniIJEPA
from src.training.metrics import MetricTracker
from src.training.callbacks import CallbackRunner


class Trainer:
    """
    Owns the full training loop for MiniIJEPA.

    Responsibilities:
    - one epoch of training (train_epoch)
    - EMA update after every optimizer step
    - LR warmup + cosine decay schedule
    - metric tracking per batch and per epoch
    - fanning out epoch events to callbacks
    """

    def __init__(
        self,
        model:       MiniIJEPA,
        loader:      DataLoader,
        cfg:         JEPAConfig,
        callbacks:   list | None = None,
    ):
        self.model   = model
        self.loader  = loader
        self.cfg     = cfg
        self.device  = torch.device(cfg.device)
        self.metrics = MetricTracker()

        # move model to device
        self.model.to(self.device)

        # optimizer — only context encoder + predictor
        # target encoder has requires_grad=False so it's automatically excluded
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # LR schedule: linear warmup then cosine decay
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda,
        )

        # callbacks
        self.callback_runner = CallbackRunner(callbacks or [])

    def _lr_lambda(self, epoch: int) -> float:
        """
        LR multiplier as a function of epoch.
        Warmup: ramp from 0 to 1 over warmup_epochs.
        After warmup: cosine decay to 0.
        """
        if epoch < self.cfg.warmup_epochs:
            return (epoch + 1) / self.cfg.warmup_epochs
        progress = (epoch - self.cfg.warmup_epochs) / (
            max(1, self.cfg.epochs - self.cfg.warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def train_epoch(self, epoch: int) -> dict:
        """
        Run one full epoch over the dataloader.
        Returns the metrics dict for this epoch.
        """
        self.model.train()
        self.metrics.reset()

        for batch_idx, (images, ctx_idx, tgt_idx) in enumerate(self.loader):
            # move to device
            images  = images.to(self.device)
            ctx_idx = ctx_idx.to(self.device)
            tgt_idx = tgt_idx.to(self.device)

            # filter out padding (-1) — use only valid indices
            ctx_idx = self._strip_padding(ctx_idx)
            tgt_idx = self._strip_padding(tgt_idx)

            # forward + loss
            self.optimizer.zero_grad()
            loss = self.model(images, ctx_idx, tgt_idx)

            # backward
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # EMA update — must happen after optimizer step
            self.model.update_target_encoder()

            # track metrics
            self.metrics.update(loss.item(), batch_size=images.shape[0])

            # log every 50 batches
            if batch_idx % 50 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"      epoch {epoch:03d} "
                      f"batch {batch_idx:04d}/{len(self.loader)} "
                      f"loss={loss.item():.4f} "
                      f"lr={lr:.6f}")

        # step LR scheduler once per epoch
        self.scheduler.step()

        # end of epoch — record metrics, fan out to callbacks
        record = self.metrics.end_epoch(epoch)
        self.callback_runner.on_epoch_end(epoch, record)
        return record

    def fit(self) -> list[dict]:
        """
        Run the full training loop for cfg.epochs epochs.
        Returns history of all epoch metric dicts.
        """
        print(f"Starting training for {self.cfg.epochs} epochs on {self.device}")
        print(f"Trainable params: "
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        for epoch in range(1, self.cfg.epochs + 1):
            record = self.train_epoch(epoch)
            print(f"  -> Epoch {epoch:03d} | loss={record['loss']:.4f}")

            # check early stopping
            for cb in self.callback_runner.callbacks:
                if hasattr(cb, "should_stop") and cb.should_stop:
                    print(f"Early stopping at epoch {epoch}.")
                    self.callback_runner.on_train_end()
                    return self.metrics.history

        self.callback_runner.on_train_end()
        return self.metrics.history

    @staticmethod
    def _strip_padding(indices: torch.Tensor) -> torch.Tensor:
        """
        Remove -1 padding from indices tensor.
        Finds minimum valid length across all samples in the batch
        to ensure no -1 values reach torch.gather.
        """
        valid_counts = (indices >= 0).sum(dim=1)  # [B]
        min_valid = valid_counts.min().item()
        return indices[:, :min_valid]
