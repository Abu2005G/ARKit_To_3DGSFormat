"""
geometry.py

Geometry processing and coordinate transformation utilities for ARKit 3D reconstruction.
Supports 2D-to-3D projection, backprojection, homogeneous coordinate transformation,
and coordinate system conversions.

Coordinate Systems
------------------
Image Space:
    (u, v) in pixels. Origin is at top-left.
Camera Space (Standard OpenCV/Open3D convention):
    +X -> right
    +Y -> down
    +Z -> forward (pointing along camera line of sight)
World Space:
    ARKit world frame (metric, gravity-aligned, right-handed system).
"""

from typing import Tuple
import numpy as np


def pixel_to_camera(
    u: float,
    v: float,
    depth: float,
    K: np.ndarray,
) -> np.ndarray:
    """
    Projects a single 2D pixel coordinate (u, v) with a corresponding depth value
    into a 3D camera space coordinate (X, Y, Z).

    Mathematical Explanation:
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

    Parameters
    ----------
    u : float
        Pixel column coordinate.
    v : float
        Pixel row coordinate.
    depth : float
        Depth in meters.
    K : np.ndarray (3, 3)
        Intrinsic camera matrix:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    Returns
    -------
    np.ndarray (3,)
        Point in 3D camera coordinates.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return np.array([x, y, z], dtype=np.float32)


def depth_to_camera(
    depth: np.ndarray,
    K_depth: np.ndarray,
) -> np.ndarray:
    """
    Converts a complete 2D depth map into a set of 3D camera coordinates (vectorized).

    Parameters
    ----------
    depth : np.ndarray (H, W)
        Depth map containing depth values in meters.
    K_depth : np.ndarray (3, 3)
        Intrinsic calibration matrix corresponding to the depth map resolution.

    Returns
    -------
    np.ndarray (H * W, 3)
        Camera space coordinates as an array of 3D points.
    """
    h, w = depth.shape
    fx = K_depth[0, 0]
    fy = K_depth[1, 1]
    cx = K_depth[0, 2]
    cy = K_depth[1, 2]

    # Create pixel coordinate grid
    u, v = np.meshgrid(
        np.arange(w),
        np.arange(h),
        indexing="xy",
    )

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    points = np.stack((x, y, z), axis=-1)
    return points.reshape(-1, 3)


def camera_to_world(
    camera_points: np.ndarray,
    T_camera_world: np.ndarray,
) -> np.ndarray:
    """
    Transforms 3D points from camera coordinate space into world coordinate space
    using a 4x4 homogeneous transformation matrix.

    Mathematical Explanation:
        P_world = T_camera_world * P_camera

    Parameters
    ----------
    camera_points : np.ndarray (N, 3)
        Points in 3D camera coordinates.
    T_camera_world : np.ndarray (4, 4)
        Camera pose matrix (rigid transform from camera to world coordinates).

    Returns
    -------
    np.ndarray (N, 3)
        Points in 3D world coordinates.
    """
    num_points = camera_points.shape[0]
    if num_points == 0:
        return camera_points.copy()

    # Convert to homogeneous coordinates (N, 4)
    ones = np.ones((num_points, 1), dtype=np.float32)
    camera_points_h = np.hstack((camera_points, ones))

    # Perform matrix multiplication T_c2w @ P_h^T, then transpose back
    world_points_h = (T_camera_world @ camera_points_h.T).T

    return world_points_h[:, :3]


def world_to_camera(
    world_points: np.ndarray,
    T_camera_world: np.ndarray,
) -> np.ndarray:
    """
    Transforms 3D points from world coordinate space into camera coordinate space
    using the inverse of the camera pose matrix.

    Mathematical Explanation:
        P_camera = (T_camera_world)^-1 * P_world

    Parameters
    ----------
    world_points : np.ndarray (N, 3)
        Points in 3D world coordinates.
    T_camera_world : np.ndarray (4, 4)
        Camera pose matrix (camera-to-world transform).

    Returns
    -------
    np.ndarray (N, 3)
        Points in 3D camera coordinates.
    """
    num_points = world_points.shape[0]
    if num_points == 0:
        return world_points.copy()

    # Invert camera-to-world to obtain world-to-camera transformation (T_w2c)
    T_world_camera = np.linalg.inv(T_camera_world)

    ones = np.ones((num_points, 1), dtype=np.float32)
    world_points_h = np.hstack((world_points, ones))

    camera_points_h = (T_world_camera @ world_points_h.T).T

    return camera_points_h[:, :3]


def project(
    camera_points: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D camera space coordinates back into 2D image plane coordinates (u, v),
    along with their corresponding depth (Z).

    Mathematical Explanation:
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        depth = Z

    Parameters
    ----------
    camera_points : np.ndarray (N, 3)
        Points in 3D camera coordinates.
    K : np.ndarray (3, 3)
        Intrinsic camera matrix.

    Returns
    -------
    pixels : np.ndarray (N, 2)
        Pixel coordinates (u, v) on the image plane.
    depths : np.ndarray (N,)
        Depth values in meters (along Z axis).
    """
    X = camera_points[:, 0]
    Y = camera_points[:, 1]
    Z = camera_points[:, 2]

    # Guard against division by zero (Z must be positive and non-zero)
    valid = Z > 1e-5
    u = np.zeros_like(X)
    v = np.zeros_like(Y)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u[valid] = fx * (X[valid] / Z[valid]) + cx
    v[valid] = fy * (Y[valid] / Z[valid]) + cy

    pixels = np.stack((u, v), axis=-1)
    return pixels, Z


def backproject(
    depth: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Utility wrapper that backprojects a complete depth map to camera coordinates.
    Equivalent to depth_to_camera.

    Parameters
    ----------
    depth : np.ndarray (H, W)
        Depth map.
    K : np.ndarray (3, 3)
        Intrinsic calibration matrix corresponding to the depth map.

    Returns
    -------
    np.ndarray (H * W, 3)
        Points in 3D camera coordinates.
    """
    return depth_to_camera(depth, K)
