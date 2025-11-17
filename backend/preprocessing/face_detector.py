"""
Face detection and alignment module
Supports multiple face detection backends (MTCNN, Dlib, OpenCV)
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Union
import warnings


class FaceDetector:
    """
    Face detector with multiple backend options.
    """
    
    def __init__(self, method: str = 'mtcnn', device: str = 'cpu'):
        """
        Initialize face detector.
        
        Args:
            method: Detection method ('mtcnn', 'dlib', 'opencv', 'mediapipe')
            device: Device for computation ('cpu' or 'cuda')
        """
        self.method = method.lower()
        self.device = device
        self.detector = self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the appropriate face detector."""
        if self.method == 'mtcnn':
            try:
                from mtcnn import MTCNN
                return MTCNN(device=self.device)
            except ImportError:
                warnings.warn("MTCNN not available, falling back to OpenCV")
                self.method = 'opencv'
                return self._initialize_opencv_detector()
        
        elif self.method == 'dlib':
            try:
                import dlib
                return dlib.get_frontal_face_detector()
            except ImportError:
                warnings.warn("Dlib not available, falling back to OpenCV")
                self.method = 'opencv'
                return self._initialize_opencv_detector()
        
        elif self.method == 'opencv':
            return self._initialize_opencv_detector()
        
        elif self.method == 'mediapipe':
            try:
                import mediapipe as mp
                return mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # Full range model
                    min_detection_confidence=0.5
                )
            except ImportError:
                warnings.warn("MediaPipe not available, falling back to OpenCV")
                self.method = 'opencv'
                return self._initialize_opencv_detector()
        
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def _initialize_opencv_detector(self):
        """Initialize OpenCV Haar Cascade face detector."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        return cv2.CascadeClassifier(cascade_path)
    
    def detect(self, image: np.ndarray, 
               min_confidence: float = 0.9) -> List[dict]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (H, W, C)
            min_confidence: Minimum detection confidence
        
        Returns:
            List of face detection dictionaries with keys:
                - 'box': (x, y, width, height)
                - 'confidence': detection confidence
                - 'keypoints': facial landmarks (if available)
        """
        if self.method == 'mtcnn':
            return self._detect_mtcnn(image, min_confidence)
        elif self.method == 'dlib':
            return self._detect_dlib(image)
        elif self.method == 'opencv':
            return self._detect_opencv(image)
        elif self.method == 'mediapipe':
            return self._detect_mediapipe(image, min_confidence)
    
    def _detect_mtcnn(self, image: np.ndarray, 
                     min_confidence: float) -> List[dict]:
        """Detect faces using MTCNN."""
        detections = self.detector.detect_faces(image)
        
        faces = []
        for det in detections:
            if det['confidence'] >= min_confidence:
                faces.append({
                    'box': det['box'],  # (x, y, width, height)
                    'confidence': det['confidence'],
                    'keypoints': det['keypoints']  # Dict of landmarks
                })
        
        return faces
    
    def _detect_dlib(self, image: np.ndarray) -> List[dict]:
        """Detect faces using Dlib."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        detections = self.detector(gray, 1)
        
        faces = []
        for det in detections:
            x, y = det.left(), det.top()
            w, h = det.width(), det.height()
            faces.append({
                'box': (x, y, w, h),
                'confidence': 1.0,  # Dlib doesn't provide confidence
                'keypoints': None
            })
        
        return faces
    
    def _detect_opencv(self, image: np.ndarray) -> List[dict]:
        """Detect faces using OpenCV Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        detections = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        faces = []
        for (x, y, w, h) in detections:
            faces.append({
                'box': (x, y, w, h),
                'confidence': 1.0,  # OpenCV doesn't provide confidence
                'keypoints': None
            })
        
        return faces
    
    def _detect_mediapipe(self, image: np.ndarray,
                         min_confidence: float) -> List[dict]:
        """Detect faces using MediaPipe."""
        results = self.detector.process(image)
        
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                if detection.score[0] >= min_confidence:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    faces.append({
                        'box': (x, y, width, height),
                        'confidence': detection.score[0],
                        'keypoints': None
                    })
        
        return faces


def detect_faces(image: np.ndarray,
                method: str = 'mtcnn',
                min_confidence: float = 0.9) -> List[dict]:
    """
    Convenience function to detect faces.
    
    Args:
        image: Input image (H, W, C)
        method: Detection method
        min_confidence: Minimum detection confidence
    
    Returns:
        List of face detections
    """
    detector = FaceDetector(method=method)
    return detector.detect(image, min_confidence=min_confidence)


def align_face(image: np.ndarray,
              face_box: Tuple[int, int, int, int],
              keypoints: Optional[dict] = None,
              output_size: Tuple[int, int] = (224, 224),
              margin: float = 0.2) -> np.ndarray:
    """
    Align and crop face from image.
    
    Args:
        image: Input image (H, W, C)
        face_box: Face bounding box (x, y, width, height)
        keypoints: Facial landmarks (optional)
        output_size: Desired output size
        margin: Margin around face (proportion of face size)
    
    Returns:
        Aligned face image
    """
    x, y, w, h = face_box
    
    # Add margin
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(image.shape[1], x + w + margin_w)
    y2 = min(image.shape[0], y + h + margin_h)
    
    # Crop face
    face = image[y1:y2, x1:x2]
    
    # Align if keypoints are available
    if keypoints is not None and 'left_eye' in keypoints and 'right_eye' in keypoints:
        face = _align_with_eyes(face, keypoints, (x1, y1))
    
    # Resize to output size
    face_resized = cv2.resize(face, output_size, interpolation=cv2.INTER_LINEAR)
    
    return face_resized


def _align_with_eyes(face: np.ndarray,
                    keypoints: dict,
                    face_offset: Tuple[int, int]) -> np.ndarray:
    """
    Align face using eye positions.
    
    Args:
        face: Face image
        keypoints: Facial landmarks
        face_offset: Offset of face in original image
    
    Returns:
        Aligned face image
    """
    # Get eye positions relative to face crop
    left_eye = (keypoints['left_eye'][0] - face_offset[0],
                keypoints['left_eye'][1] - face_offset[1])
    right_eye = (keypoints['right_eye'][0] - face_offset[0],
                 keypoints['right_eye'][1] - face_offset[1])
    
    # Compute rotation angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Rotate image
    center = (face.shape[1] // 2, face.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_face = cv2.warpAffine(
        face,
        rotation_matrix,
        (face.shape[1], face.shape[0]),
        flags=cv2.INTER_LINEAR
    )
    
    return aligned_face


def extract_face_region(image: np.ndarray,
                       method: str = 'mtcnn',
                       output_size: Tuple[int, int] = (224, 224),
                       margin: float = 0.2) -> Optional[np.ndarray]:
    """
    Detect and extract the largest face from an image.
    
    Args:
        image: Input image (H, W, C)
        method: Detection method
        output_size: Desired output size
        margin: Margin around face
    
    Returns:
        Extracted face image or None if no face detected
    """
    faces = detect_faces(image, method=method)
    
    if len(faces) == 0:
        return None
    
    # Select largest face
    largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
    
    # Extract and align face
    face_image = align_face(
        image,
        largest_face['box'],
        largest_face.get('keypoints'),
        output_size=output_size,
        margin=margin
    )
    
    return face_image
