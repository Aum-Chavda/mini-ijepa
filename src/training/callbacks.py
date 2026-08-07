from __future__ import annotations
import torch
import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class CallbackProtocol(Protocol):
    """
    Structural interface for all callbacks.
    Any class with these methods qualifies — no inheritance needed.
    runtime_checkable allows isinstance() checks at runtime.
    """

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        """Called at the end of every epoch with the epoch metrics dict."""
        ...

    def on_train_end(self) -> None:
        """Called once when training finishes."""
        ...


class CheckpointSaver:
    """
    Saves model checkpoint whenever loss improves.
    Keeps track of best loss seen so far.
    """

    def __init__(self, model: torch.nn.Module, save_dir: str):
        self.model    = model
        self.save_dir = save_dir
        self.best_loss = float("inf")
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        loss = metrics.get("loss", float("inf"))

        if loss < self.best_loss:
            self.best_loss = loss
            path = os.path.join(self.save_dir, "best_model.pth")
            torch.save({
                "epoch":       epoch,
                "loss":        loss,
                "model_state": self.model.state_dict(),
            }, path)
            print(f"      [CheckpointSaver] Saved best model at epoch {epoch} "
                f"(loss={loss:.4f}) -> {path}")
    def on_train_end(self) -> None:
        print(f"      [CheckpointSaver] Training done. "
                f"Best loss: {self.best_loss:.4f}")


class EarlyStopping:
    """
    Stops training if loss doesn't improve for `patience` epochs.
    Sets self.should_stop = True when triggered.
    """

    def __init__(self, patience: int = 30):
        self.patience    = patience
        self.best_loss   = float("inf")
        self.epochs_waited = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        loss = metrics.get("loss", float("inf"))

        if loss < self.best_loss:
            self.best_loss     = loss
            self.epochs_waited = 0
        else:
            self.epochs_waited += 1
            if self.epochs_waited >= self.patience:
                print(f"      [EarlyStopping] No improvement for "
                      f"{self.patience} epochs. Stopping.")
                self.should_stop = True

    def on_train_end(self) -> None:
        pass


class CallbackRunner:
    """
    Owns a list of callbacks and fans out events to all of them.
    Trainer only talks to CallbackRunner — not individual callbacks.
    """

    def __init__(self, callbacks: list[CallbackProtocol]):
        # verify every callback matches the Protocol at runtime
        for cb in callbacks:
            assert isinstance(cb, CallbackProtocol), \
                f"{cb} does not implement CallbackProtocol"
        self.callbacks = callbacks

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics)

    def on_train_end(self) -> None:
        for cb in self.callbacks:
            cb.on_train_end()
