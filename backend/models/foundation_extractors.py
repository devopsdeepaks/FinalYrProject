"""
Foundation model feature extractors for deepfake detection.

DINOv2: Self-supervised ViT trained on 142M images — excels at local structural
        inconsistencies (blending seams, texture discontinuities).
CLIP:   Vision-language ViT trained on 2B image-text pairs — captures semantic
        authenticity signals that complement DINOv2's structural focus.

Both are used frozen; only the downstream fusion MLP is trained.
"""

import torch
import torch.nn as nn
import timm


class DINOv2FeatureExtractor(nn.Module):
    """
    DINOv2-Base frozen feature extractor.
    Outputs (B, 768) CLS-token embeddings.
    Uses timm's dynamic position interpolation so 224px input works
    despite DINOv2's native 518px training resolution.
    """

    def __init__(
        self,
        model_name: str = 'vit_base_patch14_dinov2.lvd142m',
        img_size: int = 224,
        freeze: bool = True,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool='token',
            dynamic_img_size=True,
            img_size=img_size,
        )
        self.feature_dim = self.model.num_features
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # (B, 768)


class CLIPFeatureExtractor(nn.Module):
    """
    CLIP ViT-B/16 vision encoder frozen feature extractor.
    Outputs (B, 768) embeddings that encode semantic authenticity signals.
    Complements DINOv2's structural focus with global semantic context.
    """

    def __init__(
        self,
        model_name: str = 'vit_base_patch16_clip_224.laion2b',
        freeze: bool = True,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool='token',
        )
        self.feature_dim = self.model.num_features
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # (B, 768)
