import torch
import torch.nn as nn
from copy import deepcopy
from src.utils.config import JEPAConfig
from src.models.blocks import PatchEmbed, TransformerBlock


class VisionTransformer(nn.Module):
    """
    Full ViT encoder: PatchEmbed + positional embedding + N TransformerBlocks.
    Used for both context encoder and target encoder.
    """

    def __init__(self, cfg: JEPAConfig, embed_dim: int, depth: int, num_heads: int):
        super().__init__()
        self.patch_embed = PatchEmbed(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            embed_dim=embed_dim,
        )
        # learnable positional embeddings — one per patch
        self.pos_embed = nn.Parameter(
            torch.zeros(1, cfg.num_patches, embed_dim)
        )
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        # small random init for positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        x = self.patch_embed(x)        # [B, N, D]
        x = x + self.pos_embed         # add position information
        x = self.blocks(x)             # N transformer blocks
        x = self.norm(x)               # final layer norm
        return x                       # [B, N, D]

    def forward_masked(
        self, x: torch.Tensor, keep_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass using only context (visible) patches.
        keep_indices: [B, N_context] — which patch indices to keep
        """
        x = self.patch_embed(x)        # [B, N, D]
        x = x + self.pos_embed         # [B, N, D]

        # gather only the context patches
        B, N, D = x.shape
        keep = keep_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, N_ctx, D]
        x = torch.gather(x, dim=1, index=keep)               # [B, N_ctx, D]

        x = self.blocks(x)
        x = self.norm(x)
        return x                       # [B, N_ctx, D]


class Predictor(nn.Module):
    """
    Narrow Transformer that predicts target embeddings from context embeddings.
    Intentionally smaller than the encoder — forces compression, prevents memorisation.
    """

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        # project context embeddings down to predictor dim
        self.input_proj = nn.Linear(cfg.embed_dim, cfg.predictor_embed_dim)

        # learnable mask token — one vector representing "predict here"
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.predictor_embed_dim))

        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg.predictor_embed_dim, num_heads=3)
              for _ in range(cfg.predictor_depth)]
        )
        self.norm = nn.LayerNorm(cfg.predictor_embed_dim)

        # project back up to encoder dim for loss computation
        self.output_proj = nn.Linear(cfg.predictor_embed_dim, cfg.embed_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context_embeddings: torch.Tensor,    # [B, N_ctx, D]
        n_target: int,                        # how many target patches to predict
    ) -> torch.Tensor:
        B = context_embeddings.shape[0]

        # project context down
        x_ctx = self.input_proj(context_embeddings)   # [B, N_ctx, pred_dim]

        # expand mask tokens for each target position
        mask_tokens = self.mask_token.expand(B, n_target, -1)  # [B, N_tgt, pred_dim]

        # concatenate: context first, then mask tokens
        x = torch.cat([x_ctx, mask_tokens], dim=1)   # [B, N_ctx + N_tgt, pred_dim]

        x = self.blocks(x)
        x = self.norm(x)

        # take only the mask token outputs (the predictions)
        x = x[:, -n_target:, :]                      # [B, N_tgt, pred_dim]
        x = self.output_proj(x)                       # [B, N_tgt, D]
        return x


class MiniIJEPA(nn.Module):
    """
    Full I-JEPA model:
    - context_encoder: trained via backprop
    - target_encoder:  EMA copy, never updated by gradients
    - predictor:       predicts target embeddings from context embeddings
    """

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg

        # context encoder — gradients flow through this
        self.context_encoder = VisionTransformer(
            cfg=cfg,
            embed_dim=cfg.embed_dim,
            depth=cfg.encoder_depth,
            num_heads=cfg.encoder_heads,
        )

        # target encoder — EMA copy, no gradients ever
        self.target_encoder = deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False   # frozen — never touched by optimizer

        # predictor
        self.predictor = Predictor(cfg)

    @torch.no_grad()
    def update_target_encoder(self):
        """
        EMA update: target = momentum * target + (1 - momentum) * context
        Called manually after every optimizer step.
        """
        m = self.cfg.ema_momentum
        for ctx_param, tgt_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            tgt_param.data = m * tgt_param.data + (1 - m) * ctx_param.data

    def forward(
        self,
        images: torch.Tensor,            # [B, 3, H, W]
        context_indices: torch.Tensor,   # [B, N_ctx]  — which patches are visible
        target_indices: torch.Tensor,    # [B, N_tgt]  — which patches to predict
    ) -> torch.Tensor:
        """
        Full forward pass. Returns L2 loss between predicted and target embeddings.
        """
        # 1. encode context patches (gradients flow)
        context_embeddings = self.context_encoder.forward_masked(
            images, context_indices
        )                                           # [B, N_ctx, D]

        # 2. encode all patches with target encoder (no gradients)
        with torch.no_grad():
            all_target_embeddings = self.target_encoder(images)  # [B, N, D]

        # 3. gather only the target patch embeddings
        B, N, D = all_target_embeddings.shape
        tgt = target_indices.unsqueeze(-1).expand(-1, -1, D)     # [B, N_tgt, D]
        target_embeddings = torch.gather(all_target_embeddings, dim=1, index=tgt)
                                                                # [B, N_tgt, D]

        # 4. predict target embeddings from context
        predicted_embeddings = self.predictor(
            context_embeddings, n_target=target_indices.shape[1]
        )                                           # [B, N_tgt, D]

        # 5. L2 loss between prediction and target (detached)
        loss = nn.functional.mse_loss(predicted_embeddings, target_embeddings.detach())
        return loss