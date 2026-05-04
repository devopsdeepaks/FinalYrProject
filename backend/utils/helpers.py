"""General helper utilities."""

import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image from disk and return it in RGB format."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_results(results: Dict, output_path: str) -> None:
    """Persist a results dictionary as a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
