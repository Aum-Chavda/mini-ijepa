from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AverageMeter:
    """
    Tracks a running average of a scalar value across batches.
    Used for loss and any other per-batch metric.
    """

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0
        self.avg   = 0.0

    def update(self, value: float, n: int = 1):
        """
        Update with a new value.
        n = number of samples this value represents (batch size).
        """
        self.sum   += value * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class MetricTracker:
    """
    Tracks all metrics across one epoch.
    Owns one AverageMeter per metric.
    """

    def __init__(self):
        self.loss = AverageMeter("loss")
        self.history: list[dict] = []   # one entry per epoch

    def update(self, loss: float, batch_size: int):
        self.loss.update(loss, batch_size)

    def reset(self):
        self.loss.reset()

    def end_epoch(self, epoch: int, probe_acc: float | None = None) -> dict:
        """
        Called at end of each epoch.
        Records metrics and returns them as a dict.
        """
        record = {
            "epoch":     epoch,
            "loss":      round(self.loss.avg, 4),
            "probe_acc": round(probe_acc, 4) if probe_acc is not None else None,
        }
        self.history.append(record)
        self.reset()
        return record

    def best_loss(self) -> float:
        """Return the lowest loss seen across all epochs."""
        if not self.history:
            return float("inf")
        return min(r["loss"] for r in self.history)


def linear_probe(
    encoder: nn.Module,
    train_loader: DataLoader,
    device: str,
    num_classes: int = 10,
    epochs: int = 1,
) -> float:
    """
    Freeze encoder, train a linear head for `epochs` epochs,
    return top-1 accuracy on the training set.

    This is the standard SSL evaluation protocol:
    good representations → high linear probe accuracy
    even with a single linear layer on top.
    """
    encoder.eval()

    # ── collect all embeddings from the encoder ──────────────────────
    all_feats, all_labels = [], []

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            # forward through full encoder (all patches, no masking)
            feats = encoder(images)          # [B, N, D]
            feats = feats.mean(dim=1)        # [B, D] — average pool over patches
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    all_feats  = torch.cat(all_feats)        # [N_total, D]
    all_labels = torch.cat(all_labels)       # [N_total]

    # ── train linear head ─────────────────────────────────────────────
    head = nn.Linear(all_feats.shape[1], num_classes).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    feat_dataset = TensorDataset(all_feats, all_labels)
    feat_loader  = DataLoader(feat_dataset, batch_size=256, shuffle=True)

    head.train()
    for _ in range(epochs):
        for feats, labels in feat_loader:
            feats, labels = feats.to(device), labels.to(device)
            opt.zero_grad()
            loss_fn(head(feats), labels).backward()
            opt.step()

    # ── evaluate ──────────────────────────────────────────────────────
    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for feats, labels in feat_loader:
            feats, labels = feats.to(device), labels.to(device)
            preds = head(feats).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.shape[0]

    return correct / total
