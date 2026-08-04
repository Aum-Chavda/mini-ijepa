from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """
    Contract for all encoders in this project.
    Every encoder must implement forward() and expose embed_dim.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor of shape [B, 3, H, W]
        Returns:
            patch embeddings of shape [B, N, D]
            B = batch size, N = num patches, D = embed_dim
        """
        ...

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Embedding dimension this encoder produces."""
        ...


class BasePredictor(ABC, nn.Module):
    """
    Contract for the predictor network.
    Takes context embeddings, predicts target embeddings.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        context_embeddings: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_embeddings: [B, N_context, D]
            target_positions:   [B, N_target]  — patch indices to predict
        Returns:
            predicted embeddings: [B, N_target, D]
        """
        ...