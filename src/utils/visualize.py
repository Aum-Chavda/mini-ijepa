import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
from src.utils.config import JEPAConfig


def plot_loss_curve(history: list[dict], save_path: str = "outputs/loss_curve.png"):
    """
    Plot training loss per epoch from the history list.
    history = [{"epoch": 1, "loss": 0.95, ...}, ...]
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = [r["epoch"] for r in history]
    losses = [r["loss"]  for r in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, losses, linewidth=2, color="#4C72B0", label="Training Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("L2 Loss", fontsize=12)
    ax.set_title("Mini I-JEPA — Training Loss", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved → {save_path}")


def visualize_attention(
    model: torch.nn.Module,
    image: torch.Tensor,
    cfg: JEPAConfig,
    save_path: str = "outputs/attention_map.png",
    head_idx: int = 0,
):
    """
    Visualize self-attention from the last Transformer block
    of the context encoder for a single image.

    Extracts attention weights using a forward hook.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    image  = image.unsqueeze(0).to(device)   # [1, 3, H, W]

    # ── register hook on last transformer block ───────────────────────
    attention_weights = {}

    def hook_fn(module, input, output):
        # MultiHeadSelfAttention stores attn inside forward
        # we capture Q @ K before softmax via the qkv projection
        attention_weights["captured"] = True

    # hook into the attention of the last encoder block
    last_block = model.context_encoder.blocks[-1]

    # manually run forward and capture attn from MHSA
    with torch.no_grad():
        x = model.context_encoder.patch_embed(image)      # [1, N, D]
        x = x + model.context_encoder.pos_embed

        # run through all blocks, capture attention from last
        for i, block in enumerate(model.context_encoder.blocks):
            if i == len(model.context_encoder.blocks) - 1:
                # manually step through last block to get attn weights
                normed = block.norm1(x)
                B, N, D = normed.shape
                attn_module = block.attn
                qkv = attn_module.qkv(normed)
                qkv = qkv.reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * attn_module.scale
                attn = attn.softmax(dim=-1)   # [1, heads, N, N]
                captured_attn = attn
            else:
                x = block(x)

    # ── reshape attention to grid ─────────────────────────────────────
    grid = int(cfg.num_patches ** 0.5)   # 14
    attn_map = captured_attn[0, head_idx].cpu().numpy()   # [N, N]

    # mean attention from all patches to each patch (column mean)
    attn_mean = attn_map.mean(axis=0).reshape(grid, grid)   # [14, 14]

    # ── plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # original image (denormalise)
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std + mean).clip(0, 1)
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # attention map
    axes[1].imshow(attn_mean, cmap="viridis", interpolation="nearest")
    axes[1].set_title(f"Attention Map (head {head_idx})")
    axes[1].axis("off")

    plt.suptitle("Mini I-JEPA — Context Encoder Attention", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Attention map saved → {save_path}")