"""
Ensemble model integrating configurable backbone + attention + ViT + foundation extractors.

Supported backbones:
  - "efficientnet_b4"  — original 17.5M param CNN (baseline)
  - "convnext_large"   — 196M params, IN-22k pretrained (Phase 1A)
  - "eva02_large"      — 303M params, masked-image-modeling pretrained (Phase 3B)

Supported ViT variants:
  - "custom"           — randomly-initialized 12-layer ViT (original)
  - "pretrained_large" — timm ViT-L/16 pretrained on IN-21k (Phase 2B)

Optional foundation extractors (frozen):
  - DINOv2-Base        — structural/local inconsistency features (Phase 2A)
  - CLIP ViT-B/16      — semantic authenticity features (Phase 3D)

Fusion strategies:
  - "concat"           — concatenate all feature vectors, then MLP
  - "cross_attention"  — CNN queries ViT/DINOv2 via cross-attention (Phase 2C)
"""

import torch
import torch.nn as nn


def _build_backbone(backbone_name: str, pretrained: bool):
    if backbone_name == "convnext_large":
        from .efficientnet_model import ConvNeXtLBackbone
        return ConvNeXtLBackbone(pretrained=pretrained)
    if backbone_name == "eva02_large":
        from .efficientnet_model import EVA02LargeBackbone
        return EVA02LargeBackbone(pretrained=pretrained)
    from .efficientnet_model import EfficientNetB4Backbone
    return EfficientNetB4Backbone(pretrained=pretrained)


def _build_vit(vit_variant: str, image_size: int, num_classes: int,
               dropout: float, vit_layers: int, freeze: bool):
    if vit_variant == "pretrained_large":
        from .vision_transformer import PretrainedViTL16
        return PretrainedViTL16(pretrained=True, freeze_backbone=freeze)
    from .vision_transformer import VisionTransformer
    return VisionTransformer(
        img_size=image_size, patch_size=16, embed_dim=768,
        num_layers=vit_layers, num_heads=8,
        num_classes=num_classes, dropout=dropout,
    )


class DeepfakeDetectionModel(nn.Module):
    """
    Hybrid ensemble deepfake detector.

    Feature flow:
        image → backbone → (optional) spatial-channel attention
              → global avg pool → cnn_pooled (B, cnn_dim)
              → (optional) ViT   → vit_feats  (B, vit_dim)
              → (optional) DINOv2→ dino_feats (B, 768)
              → (optional) CLIP  → clip_feats (B, 768)
              → fusion (concat or cross-attention)
              → MLP head → logits (B, 2)

    EVA-02 exception: already outputs flat (B, 1024) — spatial attention skipped.
    """

    def __init__(
        self,
        num_classes: int = 2,
        image_size: int = 224,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_vit: bool = True,
        vit_layers: int = 6,
        backbone_name: str = "convnext_large",
        vit_variant: str = "pretrained_large",
        freeze_vit: bool = False,
        use_dino: bool = True,
        freeze_dino: bool = True,
        use_clip: bool = False,
        fusion: str = "cross_attention",
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.use_attention = use_attention and backbone_name != "eva02_large"
        self.use_vit = use_vit
        self.use_dino = use_dino
        self.use_clip = use_clip
        self.fusion_strategy = fusion

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = _build_backbone(backbone_name, pretrained=True)
        cnn_dim = self.backbone.feature_dim

        # EVA-02 is flat; ConvNeXt/EfficientNet return spatial maps
        self._backbone_is_flat = backbone_name == "eva02_large"

        # ── Spatial-channel attention (only for spatial backbones) ────────────
        if self.use_attention:
            from .attention import FeatureAttentionModule
            self.attention = FeatureAttentionModule(cnn_dim)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── ViT ───────────────────────────────────────────────────────────────
        if use_vit:
            self.vit = _build_vit(vit_variant, image_size, num_classes,
                                  dropout, vit_layers, freeze_vit)
            vit_dim = self.vit.feature_dim if hasattr(self.vit, 'feature_dim') else 768
        else:
            vit_dim = 0

        # ── Foundation extractors ─────────────────────────────────────────────
        if use_dino:
            from .foundation_extractors import DINOv2FeatureExtractor
            self.dino = DINOv2FeatureExtractor(freeze=freeze_dino)
            dino_dim = self.dino.feature_dim
        else:
            dino_dim = 0

        if use_clip:
            from .foundation_extractors import CLIPFeatureExtractor
            self.clip = CLIPFeatureExtractor(freeze=True)
            clip_dim = self.clip.feature_dim
        else:
            clip_dim = 0

        # ── Fusion head ───────────────────────────────────────────────────────
        # For cross-attention fusion: CNN queries the primary context source
        # (ViT if available, else DINOv2). Remaining features concatenated after.
        if fusion == "cross_attention" and (use_vit or use_dino):
            from .attention import CrossAttentionFusion
            context_dim = vit_dim if use_vit else dino_dim
            self.cross_attn_fusion = CrossAttentionFusion(
                cnn_dim=cnn_dim, vit_dim=context_dim, fusion_dim=512
            )
            # After cross-attn we have 512; add remaining feature streams
            extra_dim = (dino_dim if use_dino and use_vit else 0) + clip_dim
            fusion_in = 512 + extra_dim
        else:
            self.cross_attn_fusion = None
            fusion_in = cnn_dim + vit_dim + dino_dim + clip_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def _extract_cnn(self, x, return_attention=False):
        feats = self.backbone(x)

        if self._backbone_is_flat:
            # EVA-02 already returns (B, 1024) — no spatial attention or GAP
            return feats, None

        if self.use_attention:
            if return_attention:
                feats, attn_map = self.attention(feats, return_heatmap=True)
            else:
                feats = self.attention(feats)
                attn_map = None
        else:
            attn_map = None

        pooled = self.gap(feats).flatten(1)  # (B, cnn_dim)
        return pooled, attn_map

    def forward(self, x, return_attention=False):
        cnn_pooled, attn_map = self._extract_cnn(x, return_attention)

        feature_parts = []

        if self.cross_attn_fusion is not None:
            # Primary context: ViT if enabled, else DINOv2
            if self.use_vit:
                primary = self.vit.get_features(x)          # (B, vit_dim)
                fused = self.cross_attn_fusion(cnn_pooled, primary)  # (B, 512)
                feature_parts.append(fused)
                if self.use_dino:
                    feature_parts.append(self.dino(x))       # (B, 768)
            else:
                dino_feats = self.dino(x)                    # (B, 768)
                fused = self.cross_attn_fusion(cnn_pooled, dino_feats)
                feature_parts.append(fused)
        else:
            feature_parts.append(cnn_pooled)
            if self.use_vit:
                feature_parts.append(self.vit.get_features(x))
            if self.use_dino:
                feature_parts.append(self.dino(x))

        if self.use_clip:
            feature_parts.append(self.clip(x))

        combined = torch.cat(feature_parts, dim=1)
        logits = self.classifier(combined)

        if return_attention:
            return logits, attn_map
        return logits

    def predict(self, x):
        """Return softmax probabilities."""
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)
