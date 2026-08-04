import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Splits image into patches and projects to embedding dim.
    Input:  [B, 3, H, W]
    Output: [B, N, D]  where N = (H//patch_size)^2, D = embed_dim
    """

    def __init__(self, image_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # one conv does patch extraction + linear projection in one step
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        x = self.proj(x)          # [B, D, H//P, W//P]
        x = x.flatten(2)          # [B, D, N]
        x = x.transpose(1, 2)     # [B, N, D]
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention.
    Each head attends over the full sequence independently.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5   # 1/√d — prevents dot products from getting too large

        # projects input to Q, K, V in one shot
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # compute Q, K, V
        qkv = self.qkv(x)                          # [B, N, 3D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)           # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)                    # each: [B, heads, N, head_dim]

        # attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # weighted sum of values
        x = (attn @ v)                             # [B, heads, N, head_dim]
        x = x.transpose(1, 2).reshape(B, N, D)    # [B, N, D]
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """
    One Transformer layer: Pre-LN attention + Pre-LN feedforward.
    Stacked depth times to form the full encoder.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)   # 192 * 4 = 768 hidden units
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),                             # smoother than ReLU, standard in ViT
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # attention with residual
        x = x + self.attn(self.norm1(x))
        # feedforward with residual
        x = x + self.mlp(self.norm2(x))
        return x