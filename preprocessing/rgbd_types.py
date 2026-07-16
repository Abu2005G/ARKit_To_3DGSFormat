"""
rgbd_types.py

Defines data classes for representing RGB-D frames and camera parameters.
Includes coordinate conventions and sensor specifications.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    """
    Represents camera calibration and pose.

    Attributes
    ----------
    intrinsics : np.ndarray (3, 3)
        Intrinsic camera matrix:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    pose : np.ndarray (4, 4)
        Extrinsic camera matrix (camera-to-world transform T_c2w).
        Standard OpenGL convention: +X right, +Y up, -Z forward,
        or Open3D convention: +X right, +Y down, +Z forward.
        ARKit typically provides camera-to-world with +X right, +Y up, -Z forward.
    width : int
        Resolution width in pixels.
    height : int
        Resolution height in pixels.
    """
    intrinsics: np.ndarray
    pose: np.ndarray
    width: int
    height: int


@dataclass
class RGBDFrame:
    """
    Represents a synchronized RGB-D frame from an ARKit dataset.

    Attributes
    ----------
    index : int
        Frame sequence index.
    timestamp : float
        Timestamp of the frame.
    tracking_state : str
        Tracking status of ARKit (e.g., 'Normal', 'Limited').
    rgb : np.ndarray (H, W, 3)
        Color image array (UINT8).
    depth : np.ndarray (H, W)
        Depth map in meters (FLOAT32).
    camera_rgb : Camera
        Calibration and pose associated with the RGB camera.
    camera_depth : Camera
        Calibration and pose associated with the depth camera.
    """
    index: int
    timestamp: float
    tracking_state: str
    rgb: np.ndarray
    depth: np.ndarray
    camera_rgb: Camera
    camera_depth: Camera
