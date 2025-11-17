"""
Fast Fourier Transform (FFT) implementation for spectral analysis
Extracts frequency-domain features to detect manipulation artifacts

Based on the paper's formula:
F_u(u,v) = Σ Σ f(a,b) * e^(-j2π(ua/Q + vb/R))
|F_u(u,v)| = sqrt(Re(F_u)^2 + Im(F_u)^2)
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def apply_fft(image: np.ndarray, channels: str = 'all') -> np.ndarray:
    """
    Apply 2D Fast Fourier Transform to an image.
    
    Args:
        image: Input image (H, W, C) or (H, W)
        channels: Which channels to process ('all', 'gray', 'rgb')
    
    Returns:
        FFT magnitude spectrum (H, W) or (H, W, C)
    """
    if len(image.shape) == 3 and channels == 'gray':
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if len(image.shape) == 2:
        # Single channel image
        return _fft_single_channel(image)
    else:
        # Multi-channel image
        fft_channels = []
        for c in range(image.shape[2]):
            fft_c = _fft_single_channel(image[:, :, c])
            fft_channels.append(fft_c)
        return np.stack(fft_channels, axis=-1)


def _fft_single_channel(channel: np.ndarray) -> np.ndarray:
    """
    Apply FFT to a single channel.
    
    Args:
        channel: Single channel image (H, W)
    
    Returns:
        FFT magnitude spectrum (H, W)
    """
    # Apply 2D FFT
    fft = np.fft.fft2(channel)
    
    # Shift zero frequency to center
    fft_shifted = np.fft.fftshift(fft)
    
    # Compute magnitude: |F_u| = sqrt(Re^2 + Im^2)
    magnitude = np.abs(fft_shifted)
    
    # Apply log transform for better visualization
    magnitude_log = np.log1p(magnitude)
    
    return magnitude_log


def normalize_fft(fft_magnitude: np.ndarray) -> np.ndarray:
    """
    Normalize FFT magnitude to [0, 1] range.
    
    Based on paper's formula:
    F_u_norm(u,v) = (|F_u(u,v)| - min(|F_u|)) / (max(|F_u|) - min(|F_u|))
    
    Args:
        fft_magnitude: FFT magnitude spectrum
    
    Returns:
        Normalized FFT magnitude in [0, 1]
    """
    if len(fft_magnitude.shape) == 2:
        return _normalize_single_channel(fft_magnitude)
    else:
        # Multi-channel normalization
        normalized_channels = []
        for c in range(fft_magnitude.shape[2]):
            norm_c = _normalize_single_channel(fft_magnitude[:, :, c])
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
        return np.zeros_like(channel)
    
    normalized = (channel - min_val) / (max_val - min_val)
    return normalized


def extract_frequency_features(image: np.ndarray, 
                               high_freq_threshold: float = 0.1) -> Tuple[np.ndarray, dict]:
    """
    Extract frequency-domain features for deepfake detection.
    
    Args:
        image: Input image (H, W, C)
        high_freq_threshold: Threshold for high-frequency component extraction
    
    Returns:
        Tuple of (fft_features, statistics_dict)
    """
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply FFT
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    
    # Get image dimensions
    h, w = gray.shape
    center_h, center_w = h // 2, w // 2
    
    # Extract high-frequency components (away from center)
    radius = int(min(h, w) * high_freq_threshold)
    y, x = np.ogrid[:h, :w]
    mask = (x - center_w)**2 + (y - center_h)**2 > radius**2
    
    high_freq_magnitude = magnitude * mask
    low_freq_magnitude = magnitude * (~mask)
    
    # Compute statistics
    stats = {
        'total_energy': np.sum(magnitude**2),
        'high_freq_energy': np.sum(high_freq_magnitude**2),
        'low_freq_energy': np.sum(low_freq_magnitude**2),
        'high_freq_ratio': np.sum(high_freq_magnitude**2) / (np.sum(magnitude**2) + 1e-8),
        'spectral_entropy': _compute_spectral_entropy(magnitude),
    }
    
    # Normalize for neural network input
    fft_features = normalize_fft(np.log1p(magnitude))
    
    return fft_features, stats


def _compute_spectral_entropy(magnitude: np.ndarray) -> float:
    """
    Compute spectral entropy from FFT magnitude.
    
    Args:
        magnitude: FFT magnitude spectrum
    
    Returns:
        Spectral entropy value
    """
    # Normalize to probability distribution
    prob = magnitude / (np.sum(magnitude) + 1e-8)
    
    # Compute entropy: -Σ p*log(p)
    prob = prob[prob > 0]  # Remove zeros
    entropy = -np.sum(prob * np.log(prob + 1e-8))
    
    return entropy


def create_frequency_mask(image_shape: Tuple[int, int], 
                          freq_type: str = 'high') -> np.ndarray:
    """
    Create a frequency mask for selective filtering.
    
    Args:
        image_shape: Shape of the image (H, W)
        freq_type: 'high', 'low', or 'band'
    
    Returns:
        Binary mask for frequency filtering
    """
    h, w = image_shape
    center_h, center_w = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    if freq_type == 'high':
        # High-pass filter (outer regions)
        threshold = min(h, w) * 0.1
        mask = distance > threshold
    elif freq_type == 'low':
        # Low-pass filter (center region)
        threshold = min(h, w) * 0.3
        mask = distance <= threshold
    else:  # band
        # Band-pass filter
        inner_threshold = min(h, w) * 0.1
        outer_threshold = min(h, w) * 0.3
        mask = (distance > inner_threshold) & (distance <= outer_threshold)
    
    return mask.astype(np.float32)
