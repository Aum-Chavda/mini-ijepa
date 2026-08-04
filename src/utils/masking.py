import torch
import math
import random
from src.utils.config import JEPAConfig


class MultiBlockMaskGenerator:
    """
    Generates context and target patch indices for I-JEPA training.

    Target: multiple large contiguous rectangular blocks (semantic scale)
    Context: remaining patches, spatially distributed, subsampled to ~50%
    """

    def __init__(self, cfg: JEPAConfig):
        self.num_patches = cfg.num_patches
        self.grid_size = int(math.sqrt(cfg.num_patches))  # 14
        self.num_target_blocks = cfg.num_target_blocks
        self.target_scale = cfg.target_scale
        self.target_aspect_ratio = cfg.target_aspect_ratio
        self.context_scale = cfg.context_scale

    def _sample_block(self) -> list[int]:
        """
        Sample one rectangular target block.
        Returns list of patch indices inside the block.
        """
        g = self.grid_size  # 14

        scale = random.uniform(*self.target_scale)
        num_patches_in_block = int(scale * self.num_patches)

        aspect = random.uniform(*self.target_aspect_ratio)

        h = int(round(math.sqrt(num_patches_in_block * aspect)))
        w = int(round(math.sqrt(num_patches_in_block / aspect)))

        h = max(1, min(h, g))
        w = max(1, min(w, g))

        top  = random.randint(0, g - h)
        left = random.randint(0, g - w)

        indices = []
        for row in range(top, top + h):
            for col in range(left, left + w):
                indices.append(row * g + col)

        return indices

    def __call__(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate context and target indices for a full batch.

        Returns:
            context_indices: [B, N_context]
            target_indices:  [B, N_target]
        """
        all_context = []
        all_target  = []

        for _ in range(batch_size):
            target_set = set()
            for _ in range(self.num_target_blocks):
                block = self._sample_block()
                target_set.update(block)

            target_list = sorted(target_set)

            non_target = [i for i in range(self.num_patches) if i not in target_set]
            n_context = int(self.num_patches * self.context_scale)
            n_context = min(n_context, len(non_target))

            context_list = sorted(random.sample(non_target, n_context))

            all_context.append(torch.tensor(context_list, dtype=torch.long))
            all_target.append(torch.tensor(target_list,  dtype=torch.long))

        context_indices = self._pad_and_stack(all_context)
        target_indices  = self._pad_and_stack(all_target)

        return context_indices, target_indices

    @staticmethod
    def _pad_and_stack(index_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Pad all tensors to the same length and stack into [B, N].
        Padding value is -1 — never a valid patch index (valid range 0-195).
        """
        max_len = max(t.shape[0] for t in index_list)
        padded = []
        for t in index_list:
            pad_size = max_len - t.shape[0]
            if pad_size > 0:
                padding = torch.full((pad_size,), fill_value=-1, dtype=torch.long)
                padded.append(torch.cat([t, padding]))
            else:
                padded.append(t)
        return torch.stack(padded)