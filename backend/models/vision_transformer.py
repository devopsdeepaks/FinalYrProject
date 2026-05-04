"""
Vision Transformer (ViT) for deepfake detection
16x16 patch embeddings, 12 transformer blocks, 768 hidden dim, 8 attention heads
"""

import torch
import torch.nn as nn
import timm


class PatchEmbedding(nn.Module):
    """Splits image into non-overlapping patches and linearly projects them."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer encoder block with pre-norm."""

    def __init__(self, embed_dim=768, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for deepfake detection.
    Paper specs: 16x16 patches, hidden_dim=768, 12 layers, 8 heads.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=2, embed_dim=768, num_layers=12,
                 num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]  # class token

    def forward(self, x):
        return self.head(self.forward_features(x))

    def get_features(self, x):
        """Return class-token embeddings (no classification head)."""
        return self.forward_features(x)


class PretrainedViTL16(nn.Module):
    """
    ViT-Large/16 pretrained on ImageNet-21k (fine-tuned on IN-1k via timm).
    Replaces the custom randomly-initialized ViT with a model that understands
    fine-grained visual structure from large-scale pretraining.
    Output: (B, 1024) class-token embeddings.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.model = timm.create_model(
            'vit_large_patch16_224.augreg_in21k_ft_in1k',
            pretrained=pretrained,
            num_classes=0,
            global_pool='token',
        )
        self.feature_dim = 1024
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        return self.model(x)  # (B, 1024)

    def get_features(self, x):
        return self.forward(x)
