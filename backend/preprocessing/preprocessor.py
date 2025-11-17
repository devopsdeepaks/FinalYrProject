"""
Main preprocessing pipeline for deepfake detection
Combines FFT, LBP, and face detection for comprehensive preprocessing
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import torch

from .fft_transform import apply_fft, normalize_fft, extract_frequency_features
from .lbp_features import compute_lbp, normalize_lbp, extract_multiscale_lbp
from .face_detector import extract_face_region


class DeepfakePreprocessor:
    """
    Comprehensive preprocessing pipeline for deepfake detection.
    Implements FFT analysis, LBP texture features, and face detection.
    """
    
    def __init__(self, 
                 image_size: int = 224,
                 use_fft: bool = True,
                 use_lbp: bool = True,
                 face_detection_method: str = 'mtcnn',
                 face_margin: float = 0.2,
                 lbp_radius: int = 1,
                 lbp_points: int = 8):
        """
        Initialize the preprocessor.
        
        Args:
            image_size: Target image size for model input
            use_fft: Whether to extract FFT features
            use_lbp: Whether to extract LBP features
            face_detection_method: Method for face detection
            face_margin: Margin around detected face
            lbp_radius: Radius for LBP computation
            lbp_points: Number of points for LBP
        """
        self.image_size = image_size
        self.use_fft = use_fft
        self.use_lbp = use_lbp
        self.face_detection_method = face_detection_method
        self.face_margin = face_margin
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        
        # Statistics for normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, 
                   image: np.ndarray,
                   detect_face: bool = True) -> Dict[str, np.ndarray]:
        """
        Preprocess an image for deepfake detection.
        
        Args:
            image: Input image (H, W, C) in RGB format
            detect_face: Whether to detect and crop face
        
        Returns:
            Dictionary containing:
                - 'rgb': Preprocessed RGB image
                - 'fft': FFT features (if use_fft=True)
                - 'lbp': LBP features (if use_lbp=True)
                - 'face_detected': Boolean indicating if face was found
        """
        result = {}
        
        # Detect and extract face if requested
        if detect_face:
            face_image = extract_face_region(
                image,
                method=self.face_detection_method,
                output_size=(self.image_size, self.image_size),
                margin=self.face_margin
            )
            
            if face_image is None:
                # No face detected, use full image
                face_image = cv2.resize(
                    image,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_LINEAR
                )
                result['face_detected'] = False
            else:
                result['face_detected'] = True
        else:
            # Use full image
            face_image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR
            )
            result['face_detected'] = None
        
        # Normalize RGB image
        rgb_normalized = self._normalize_rgb(face_image)
        result['rgb'] = rgb_normalized
        
        # Extract FFT features if enabled
        if self.use_fft:
            fft_features, fft_stats = extract_frequency_features(face_image)
            fft_features_resized = cv2.resize(
                fft_features,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR
            )
            result['fft'] = fft_features_resized
            result['fft_stats'] = fft_stats
        
        # Extract LBP features if enabled
        if self.use_lbp:
            lbp_features = compute_lbp(
                face_image,
                radius=self.lbp_radius,
                n_points=self.lbp_points
            )
            lbp_normalized = normalize_lbp(lbp_features)
            
            # Handle multi-channel LBP
            if len(lbp_normalized.shape) == 3:
                lbp_normalized = np.mean(lbp_normalized, axis=2)
            
            # Expand to 3 channels for consistency
            lbp_3channel = np.stack([lbp_normalized] * 3, axis=-1)
            result['lbp'] = lbp_3channel
        
        return result
    
    def _normalize_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize RGB image using ImageNet statistics.
        
        Args:
            image: Input image in range [0, 255]
        
        Returns:
            Normalized image in range approximately [-2, 2]
        """
        # Convert to float and scale to [0, 1]
        image_float = image.astype(np.float32) / 255.0
        
        # Normalize with mean and std
        normalized = (image_float - self.mean) / self.std
        
        return normalized
    
    def preprocess_batch(self,
                        images: list,
                        detect_faces: bool = True) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images
            detect_faces: Whether to detect faces
        
        Returns:
            Dictionary of batched tensors
        """
        batch_results = {
            'rgb': [],
            'fft': [] if self.use_fft else None,
            'lbp': [] if self.use_lbp else None,
            'face_detected': []
        }
        
        for image in images:
            result = self.preprocess(image, detect_face=detect_faces)
            
            batch_results['rgb'].append(result['rgb'])
            batch_results['face_detected'].append(result['face_detected'])
            
            if self.use_fft:
                batch_results['fft'].append(result['fft'])
            
            if self.use_lbp:
                batch_results['lbp'].append(result['lbp'])
        
        # Convert to tensors (N, C, H, W) format
        batch_tensors = {
            'rgb': self._to_tensor_batch(batch_results['rgb'])
        }
        
        if self.use_fft:
            batch_tensors['fft'] = self._to_tensor_batch(batch_results['fft'])
        
        if self.use_lbp:
            batch_tensors['lbp'] = self._to_tensor_batch(batch_results['lbp'])
        
        batch_tensors['face_detected'] = batch_results['face_detected']
        
        return batch_tensors
    
    def _to_tensor_batch(self, images: list) -> torch.Tensor:
        """
        Convert list of images to batched tensor.
        
        Args:
            images: List of numpy arrays (H, W, C)
        
        Returns:
            Tensor of shape (N, C, H, W)
        """
        # Stack images
        batch_array = np.stack(images, axis=0)
        
        # Convert to tensor and transpose to (N, C, H, W)
        batch_tensor = torch.from_numpy(batch_array).float()
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)
        
        return batch_tensor
    
    def create_combined_features(self, 
                                preprocessed: Dict[str, np.ndarray],
                                mode: str = 'concat') -> np.ndarray:
        """
        Combine RGB, FFT, and LBP features.
        
        Args:
            preprocessed: Dictionary from preprocess()
            mode: Combination mode ('concat', 'stack', 'weighted')
        
        Returns:
            Combined feature array
        """
        features = [preprocessed['rgb']]
        
        if self.use_fft and 'fft' in preprocessed:
            fft_3channel = np.stack([preprocessed['fft']] * 3, axis=-1)
            features.append(fft_3channel)
        
        if self.use_lbp and 'lbp' in preprocessed:
            features.append(preprocessed['lbp'])
        
        if mode == 'concat':
            # Concatenate along channel dimension
            combined = np.concatenate(features, axis=-1)
        elif mode == 'stack':
            # Stack as separate samples
            combined = np.stack(features, axis=0)
        elif mode == 'weighted':
            # Weighted average (RGB: 0.5, FFT: 0.3, LBP: 0.2)
            weights = [0.5, 0.3, 0.2][:len(features)]
            weights = [w / sum(weights) for w in weights]
            combined = sum(w * f for w, f in zip(weights, features))
        else:
            raise ValueError(f"Unknown combination mode: {mode}")
        
        return combined
    
    def extract_video_frames(self,
                            video_path: str,
                            num_frames: int = 10,
                            method: str = 'uniform') -> list:
        """
        Extract frames from video for processing.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            method: Extraction method ('uniform', 'random', 'keyframes')
        
        Returns:
            List of extracted frames
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if method == 'uniform':
            # Extract frames uniformly across video
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        elif method == 'random':
            # Extract random frames
            frame_indices = np.random.choice(total_frames, num_frames, replace=False)
            frame_indices = np.sort(frame_indices)
        else:  # keyframes
            # Extract keyframes (placeholder - would need scene detection)
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
