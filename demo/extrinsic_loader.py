"""
extrinsic_loader.py

Unified loader for camera extrinsic JSON files. Two formats are recognised:

  Rodrigues format
  ----------------
  {"rvec": [rx, ry, rz], "tvec": [tx, ty, tz]}
  (rvec may be nested, e.g. [[rx], [ry], [rz]])

  Rotation-matrix format  (conflab *_zh.json files)
  --------------------------------------------------
  {"rotation": [[...3x3...]], "translation": [[tx], [ty], [tz]]}

Both return (rvec, tvec) as (3, 1) float64 ndarrays in the standard OpenCV
convention:  p_cam = R @ p_world + t.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_extrinsic(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rvec, tvec)`` as ``(3, 1)`` float64 arrays.

    Accepts both the Rodrigues (``rvec`` / ``tvec``) format and the
    rotation-matrix (``rotation`` / ``translation``) format.
    """
    with open(path) as f:
        data = json.load(f)

    if "rvec" in data:
        rvec = np.asarray(data["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(data["tvec"], dtype=np.float64).reshape(3, 1)
    elif "rotation" in data:
        R = np.asarray(data["rotation"], dtype=np.float64).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(R)  # (3, 1)
        tvec = np.asarray(data["translation"], dtype=np.float64).reshape(3, 1)
    else:
        raise ValueError(
            f"Unrecognised extrinsic format in {path!r}. "
            "Expected keys 'rvec'/'tvec' or 'rotation'/'translation'."
        )

    return rvec, tvec
