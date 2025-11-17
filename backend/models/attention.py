"""
Attention mechanisms for deepfake detection
Implements spatial and channel attention for artifact localization

Based on the paper:
- 8-head spatial-channel self-attention mechanism
- Attention heatmaps for visualization
- Localizes suspicious regions in images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ChannelAttention(nn.Module):
    """
    Channel attention module.
    Focuses on 'what' is meaningful in the feature maps.
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Initialize channel attention.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for bottleneck
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Attention-weighted output (B, C, H, W)
        """
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x))
        
        # Max pooling path
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        
        # Apply attention
        out = x * attention
        
        return out


class SpatialAttention(nn.Module):
    """
    Spatial attention module.
    Focuses on 'where' is meaningful in the feature maps.
    
    Based on paper's formula:
    A = σ(f(W)), where σ is sigmoid
    Y = W ⊙ A (element-wise multiplication)
    """
    
    def __init__(self, kernel_size=7):
        """
        Initialize spatial attention.
        
        Args:
            kernel_size: Kernel size for spatial attention convolution
        """
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Attention-weighted output (B, C, H, W)
        """
        # Compute spatial statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        concat = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        attention = self.sigmoid(self.conv(concat))
        
        # Apply attention (element-wise multiplication)
        out = x * attention
        
        return out


class SpatialChannelAttention(nn.Module):
    """
    Combined spatial-channel attention mechanism.
    Implements the attention mechanism described in the paper.
    """
    
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        """
        Initialize spatial-channel attention.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """
        Forward pass through combined attention.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Attention-weighted output (B, C, H, W)
        """
        # Apply channel attention first
        out = self.channel_attention(x)
        
        # Then apply spatial attention
        out = self.spatial_attention(out)
        
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    Implements 8-head attention as described in the paper.
    """
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        """
        Initialize multi-head self-attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor (B, N, C) where N is sequence length
        
        Returns:
            Attention output (B, N, C)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        out = self.proj(out)
        out = self.dropout(out)
        
        return out, attn


class AttentionGate(nn.Module):
    """
    Attention gate for feature refinement.
    Used to localize suspicious regions in deepfake images.
    """
    
    def __init__(self, F_g, F_l, F_int):
        """
        Initialize attention gate.
        
        Args:
            F_g: Number of feature maps (channels) in gate signal
            F_l: Number of feature maps in input
            F_int: Number of feature maps in intermediate layer
        """
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Forward pass through attention gate.
        
        Args:
            g: Gate signal (B, F_g, H, W)
            x: Input features (B, F_l, H', W')
        
        Returns:
            Gated features (B, F_l, H', W')
        """
        # Apply transformations
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Combine and activate
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention
        out = x * psi
        
        return out


class AttentionHeatmap(nn.Module):
    """
    Generate attention heatmaps for visualization.
    Helps identify suspicious regions in deepfake images.
    """
    
    def __init__(self, in_channels):
        """
        Initialize heatmap generator.
        
        Args:
            in_channels: Number of input channels
        """
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Generate attention heatmap.
        
        Args:
            x: Feature maps (B, C, H, W)
        
        Returns:
            Attention heatmap (B, 1, H, W)
        """
        heatmap = self.conv(x)
        return heatmap
    
    def visualize(self, x, return_numpy=True):
        """
        Generate visualizable heatmap.
        
        Args:
            x: Feature maps (B, C, H, W)
            return_numpy: Return as numpy array
        
        Returns:
            Heatmap for visualization
        """
        with torch.no_grad():
            heatmap = self.forward(x)
            
            if return_numpy:
                heatmap = heatmap.cpu().numpy()
            
            return heatmap


class FeatureAttentionModule(nn.Module):
    """
    Feature attention module combining spatial and channel attention.
    Implements the attention mechanism for artifact localization.
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Initialize feature attention module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention
        """
        super().__init__()
        
        self.spatial_channel_attn = SpatialChannelAttention(
            in_channels, reduction_ratio
        )
        self.heatmap_generator = AttentionHeatmap(in_channels)
    
    def forward(self, x, return_heatmap=False):
        """
        Forward pass through feature attention.
        
        Args:
            x: Input features (B, C, H, W)
            return_heatmap: Whether to return attention heatmap
        
        Returns:
            Attention-weighted features and optionally heatmap
        """
        # Apply spatial-channel attention
        out = self.spatial_channel_attn(x)
        
        if return_heatmap:
            # Generate heatmap for visualization
            heatmap = self.heatmap_generator(x)
            return out, heatmap
        
        return out
