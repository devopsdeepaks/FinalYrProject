"""
Local Binary Pattern (LBP) implementation for texture analysis
Captures local texture information to detect manipulation artifacts

Based on the paper's formula:
LBP = Σ(q=0 to q-1) s(i_q - i_c) * 2^q
where s(l) = 1 if l ≥ 0, else 0
LBP_norm = (LBP - min(LBP)) / (max(LBP) - min(LBP))
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from skimage.feature import local_binary_pattern


def compute_lbp(image: np.ndarray, 
                radius: int = 1, 
                n_points: Optional[int] = None,
                method: str = 'uniform') -> np.ndarray:
    """
    Compute Local Binary Pattern for an image.
    
    Args:
        image: Input image (H, W, C) or (H, W)
        radius: Radius of circle (default: 1)
        n_points: Number of circularly symmetric neighbor points (default: 8*radius)
        method: LBP method ('uniform', 'default', 'ror', 'var')
    
    Returns:
        LBP feature map (H, W) or (H, W, C)
    """
    if n_points is None:
        n_points = 8 * radius
    
    if len(image.shape) == 3:
        # Convert to grayscale or process each channel
        lbp_channels = []
        for c in range(image.shape[2]):
            lbp_c = local_binary_pattern(
                image[:, :, c], 
                n_points, 
                radius, 
                method=method
            )
            lbp_channels.append(lbp_c)
        return np.stack(lbp_channels, axis=-1)
    else:
        # Single channel
        return local_binary_pattern(image, n_points, radius, method=method)


def compute_lbp_manual(image: np.ndarray, 
                       radius: int = 1, 
                       n_points: int = 8) -> np.ndarray:
    """
    Manual implementation of LBP according to paper's formula.
    
    LBP = Σ(q=0 to q-1) s(i_q - i_c) * 2^q
    where s(l) = 1 if l ≥ 0, else 0
    
    Args:
        image: Input grayscale image (H, W)
        radius: Radius of circle
        n_points: Number of circularly symmetric neighbor points
    
    Returns:
        LBP feature map (H, W)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    h, w = image.shape
    lbp_image = np.zeros((h, w), dtype=np.float32)
    
    # Generate circular neighbor positions
    angles = 2 * np.pi * np.arange(n_points) / n_points
    dy = -radius * np.sin(angles)
    dx = radius * np.cos(angles)
    
    # Compute LBP for each pixel
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center_value = image[i, j]
            lbp_code = 0
            
            for q in range(n_points):
                # Bilinear interpolation for sub-pixel positions
                y = i + dy[q]
                x = j + dx[q]
                
                neighbor_value = _bilinear_interpolate(image, x, y)
                
                # s(i_q - i_c) * 2^q
                if neighbor_value >= center_value:
                    lbp_code += 2**q
            
            lbp_image[i, j] = lbp_code
    
    return lbp_image


def _bilinear_interpolate(image: np.ndarray, x: float, y: float) -> float:
    """
    Bilinear interpolation for sub-pixel image values.
    
    Args:
        image: Input image
        x: x-coordinate (can be fractional)
        y: y-coordinate (can be fractional)
    
    Returns:
        Interpolated pixel value
    """
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1
    
    # Clip to image boundaries
    h, w = image.shape
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    
    # Bilinear weights
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    
    # Interpolated value
    value = (wa * image[y0, x0] + 
             wb * image[y0, x1] + 
             wc * image[y1, x0] + 
             wd * image[y1, x1])
    
    return value


def normalize_lbp(lbp_features: np.ndarray) -> np.ndarray:
    """
    Normalize LBP features to [0, 1] range.
    
    Based on paper's formula:
    LBP_norm = (LBP - min(LBP)) / (max(LBP) - min(LBP))
    
    Args:
        lbp_features: LBP feature map
    
    Returns:
        Normalized LBP features in [0, 1]
    """
    if len(lbp_features.shape) == 2:
        return _normalize_single_channel(lbp_features)
    else:
        # Multi-channel normalization
        normalized_channels = []
        for c in range(lbp_features.shape[2]):
            norm_c = _normalize_single_channel(lbp_features[:, :, c])
            normalized_channels.append(norm_c)
        return np.stack(normalized_channels, axis=-1)


def _normalize_single_channel(channel: np.ndarray) -> np.ndarray:
    """
    Normalize a single channel to [0, 1].
    
    Args:
        channel: Single channel data
    
    Returns:
        Normalized channel in [0, 1]
    """
    min_val = np.min(channel)
    max_val = np.max(channel)
    
    # Avoid division by zero
    if max_val - min_val < 1e-8:
        return np.zeros_like(channel, dtype=np.float32)
    
    normalized = (channel - min_val) / (max_val - min_val)
    return normalized.astype(np.float32)


def compute_lbp_histogram(lbp_image: np.ndarray, 
                         n_bins: int = 256,
                         normalize: bool = True) -> np.ndarray:
    """
    Compute histogram of LBP features.
    
    Args:
        lbp_image: LBP feature map
        n_bins: Number of histogram bins
        normalize: Whether to normalize histogram
    
    Returns:
        LBP histogram
    """
    if len(lbp_image.shape) == 3:
        # Flatten all channels
        lbp_flat = lbp_image.reshape(-1)
    else:
        lbp_flat = lbp_image.flatten()
    
    # Compute histogram
    hist, _ = np.histogram(lbp_flat, bins=n_bins, range=(0, n_bins))
    
    if normalize:
        hist = hist.astype(np.float32)
        hist = hist / (np.sum(hist) + 1e-8)
    
    return hist


def extract_multiscale_lbp(image: np.ndarray,
                           radii: list = [1, 2, 3],
                           n_points_list: Optional[list] = None) -> dict:
    """
    Extract multi-scale LBP features for robust texture analysis.
    
    Args:
        image: Input image (H, W, C) or (H, W)
        radii: List of radii for multi-scale analysis
        n_points_list: List of n_points for each radius (default: 8*radius)
    
    Returns:
        Dictionary of LBP features at different scales
    """
    if n_points_list is None:
        n_points_list = [8 * r for r in radii]
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    lbp_features = {}
    for radius, n_points in zip(radii, n_points_list):
        lbp = compute_lbp(image, radius=radius, n_points=n_points)
        lbp_norm = normalize_lbp(lbp)
        hist = compute_lbp_histogram(lbp, n_bins=n_points + 2)
        
        lbp_features[f'radius_{radius}'] = {
            'lbp_map': lbp_norm,
            'histogram': hist,
            'mean': np.mean(lbp_norm),
            'std': np.std(lbp_norm),
            'entropy': _compute_entropy(hist)
        }
    
    return lbp_features


def _compute_entropy(histogram: np.ndarray) -> float:
    """
    Compute entropy of a histogram.
    
    Args:
        histogram: Normalized histogram
    
    Returns:
        Entropy value
    """
    # Normalize if not already normalized
    prob = histogram / (np.sum(histogram) + 1e-8)
    prob = prob[prob > 0]  # Remove zeros
    entropy = -np.sum(prob * np.log(prob + 1e-8))
    return entropy


def compute_texture_consistency_score(lbp_features: np.ndarray,
                                      window_size: int = 32) -> float:
    """
    Compute texture consistency score across the image.
    Deepfakes often have inconsistent texture patterns.
    
    Args:
        lbp_features: LBP feature map
        window_size: Size of sliding window
    
    Returns:
        Texture consistency score (lower = more inconsistent)
    """
    if len(lbp_features.shape) == 3:
        lbp_features = lbp_features[:, :, 0]
    
    h, w = lbp_features.shape
    
    # Divide image into overlapping windows
    step = window_size // 2
    window_stats = []
    
    for i in range(0, h - window_size, step):
        for j in range(0, w - window_size, step):
            window = lbp_features[i:i+window_size, j:j+window_size]
            window_stats.append({
                'mean': np.mean(window),
                'std': np.std(window)
            })
    
    # Compute variance of window statistics
    means = [s['mean'] for s in window_stats]
    stds = [s['std'] for s in window_stats]
    
    consistency_score = 1.0 / (1.0 + np.std(means) + np.std(stds))
    
    return consistency_score
