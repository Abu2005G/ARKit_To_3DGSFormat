"""
validation.py

Implements validation checks for RGB inputs, depth arrays, camera calibration
matrices, camera pose matrices, tracking status, and general ARKit dataset integrity.
Prints comprehensive diagnostics in case of anomalies.
"""

from typing import Dict, Any, Tuple
import numpy as np

from preprocessing.rgbd_types import RGBDFrame
from preprocessing.config import MIN_DEPTH_METERS, MAX_DEPTH_METERS


def validate_rgb(rgb: np.ndarray) -> Tuple[bool, str]:
    """
    Validates the dimension, shape, and value range of the RGB image.

    Parameters
    ----------
    rgb : np.ndarray

    Returns
    -------
    Tuple[bool, str] : (is_valid, reason)
    """
    if not isinstance(rgb, np.ndarray):
        return False, "RGB is not a numpy array"
    
    if len(rgb.shape) != 3 or rgb.shape[2] != 3:
        return False, f"Expected RGB shape of (H, W, 3), got {rgb.shape}"

    if rgb.dtype != np.uint8:
        # Check if float in 0..1
        if np.issubdtype(rgb.dtype, np.floating):
            if np.min(rgb) < 0.0 or np.max(rgb) > 1.0:
                return False, f"RGB is float type, but range [{np.min(rgb)}, {np.max(rgb)}] is outside [0.0, 1.0]"
        else:
            return False, f"Unexpected RGB dtype: {rgb.dtype}"

    return True, "RGB is valid"


def validate_depth(depth: np.ndarray) -> Tuple[bool, str]:
    """
    Validates depth matrix shape, type, ranges, and check for numerical issues.

    Parameters
    ----------
    depth : np.ndarray

    Returns
    -------
    Tuple[bool, str] : (is_valid, reason)
    """
    if not isinstance(depth, np.ndarray):
        return False, "Depth is not a numpy array"

    if len(depth.shape) != 2:
        return False, f"Expected Depth shape of (H, W), got {depth.shape}"

    if not np.issubdtype(depth.dtype, np.floating):
        return False, f"Expected floating point depth, got {depth.dtype}"

    # Diagnostics
    total_pixels = depth.size
    nans = np.isnan(depth).sum()
    infs = np.isinf(depth).sum()
    zeros = np.sum(depth == 0.0)

    valid_mask = np.isfinite(depth) & (depth > 0)
    valid_count = np.sum(valid_mask)

    if valid_count == 0:
        return False, "Depth map contains no valid positive depth values"

    valid_vals = depth[valid_mask]
    min_val, max_val = valid_vals.min(), valid_vals.max()

    # Warn if depth range is extreme
    if min_val < MIN_DEPTH_METERS or max_val > MAX_DEPTH_METERS + 5.0:
        msg = f"Depth range [{min_val:.3f}m, {max_val:.3f}m] contains values outside typical bounds [{MIN_DEPTH_METERS}m, {MAX_DEPTH_METERS}m]"
        return True, f"Valid with warnings: {msg} (NaNs: {nans}, Infs: {infs}, Zeros: {zeros})"

    return True, f"Depth is valid (NaNs: {nans}, Infs: {infs}, Zeros: {zeros}, Min: {min_val:.3f}m, Max: {max_val:.3f}m)"


def validate_intrinsics(K: np.ndarray) -> Tuple[bool, str]:
    """
    Validates 3x3 Camera Intrinsics matrix format.

    Parameters
    ----------
    K : np.ndarray

    Returns
    -------
    Tuple[bool, str]
    """
    if not isinstance(K, np.ndarray) or K.shape != (3, 3):
        return False, f"Expected 3x3 Calibration matrix, got shape {K.shape if isinstance(K, np.ndarray) else type(K)}"

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Focal lengths and principal points must be strictly positive
    if fx <= 0 or fy <= 0:
        return False, f"Negative or zero focal length: fx={fx}, fy={fy}"
    if cx <= 0 or cy <= 0:
        return False, f"Negative or zero principal point: cx={cx}, cy={cy}"

    # General structure verification
    if K[2, 0] != 0.0 or K[2, 1] != 0.0 or K[2, 2] != 1.0:
        return False, f"Bottom row of K is not [0, 0, 1]: {K[2, :]}"

    return True, "Intrinsics are valid"


def validate_pose(T: np.ndarray) -> Tuple[bool, str]:
    """
    Validates 4x4 Extrinsic transformation matrix (pose T_c2w).
    Checks rotation matrix orthogonality and translation reality.

    Parameters
    ----------
    T : np.ndarray

    Returns
    -------
    Tuple[bool, str]
    """
    if not isinstance(T, np.ndarray) or T.shape != (4, 4):
        return False, f"Expected 4x4 Pose matrix, got shape {T.shape if isinstance(T, np.ndarray) else type(T)}"

    # Check last row is [0, 0, 0, 1]
    if not np.allclose(T[3, :3], 0.0, atol=1e-5) or not np.allclose(T[3, 3], 1.0, atol=1e-5):
        return False, f"Last row of extrinsic pose is not [0, 0, 0, 1]: {T[3, :]}"

    # Extract rotation
    R = T[:3, :3]
    
    # Orthogonality: R.T @ R should be close to identity matrix
    identity_err = np.max(np.abs(R.T @ R - np.eye(3)))
    if identity_err > 1e-3:
        return False, f"Rotation matrix is not orthogonal. R.T @ R error: {identity_err:.6f}"

    # Determinant of R should be close to +1.0
    det = np.linalg.det(R)
    if np.abs(det - 1.0) > 1e-3:
        return False, f"Rotation matrix determinant has error: {det:.6f} (expected +1.0)"

    return True, "Camera Pose (extrinsic matrix) is valid"


def validate_tracking(tracking_state: str) -> Tuple[bool, str]:
    """
    Validates tracking state of ARKit.

    Parameters
    ----------
    tracking_state : str

    Returns
    -------
    Tuple[bool, str]
    """
    state_clean = tracking_state.strip().lower()
    if "normal" in state_clean:
        return True, "Tracking is optimal"
    
    return True, f"Warning: ARKit tracking state is '{tracking_state}', reconstruction might suffer"


def validate_frame(frame: RGBDFrame) -> Dict[str, Any]:
    """
    Runs complete validation check on a single RGBDFrame object.

    Parameters
    ----------
    frame : RGBDFrame

    Returns
    -------
    Dict[str, Any] : Results containing validity stats.
    """
    results = {}
    
    ok_rgb, msg_rgb = validate_rgb(frame.rgb)
    ok_depth, msg_depth = validate_depth(frame.depth)
    
    ok_k_rgb, msg_k_rgb = validate_intrinsics(frame.camera_rgb.intrinsics)
    ok_k_depth, msg_k_depth = validate_intrinsics(frame.camera_depth.intrinsics)
    
    ok_pose_rgb, msg_pose_rgb = validate_pose(frame.camera_rgb.pose)
    ok_pose_depth, msg_pose_depth = validate_pose(frame.camera_depth.pose)
    
    ok_track, msg_track = validate_tracking(frame.tracking_state)

    results["rgb"] = {"success": ok_rgb, "msg": msg_rgb}
    results["depth"] = {"success": ok_depth, "msg": msg_depth}
    results["intrinsics_rgb"] = {"success": ok_k_rgb, "msg": msg_k_rgb}
    results["intrinsics_depth"] = {"success": ok_k_depth, "msg": msg_k_depth}
    results["pose_rgb"] = {"success": ok_pose_rgb, "msg": msg_pose_rgb}
    results["pose_depth"] = {"success": ok_pose_depth, "msg": msg_pose_depth}
    results["tracking"] = {"success": ok_track, "msg": msg_track}

    overall_success = all([
        ok_rgb, ok_depth, ok_k_rgb, ok_k_depth, ok_pose_rgb, ok_pose_depth
    ])
    results["overall_success"] = overall_success

    return results


def print_diagnostics(frame: RGBDFrame) -> None:
    """
    Runs validation on a frame and outputs results to stdout.
    """
    results = validate_frame(frame)
    print(f"--- Frame {frame.index:04d} Diagnostics ---")
    print(f"Overall Status: {'PASSED' if results['overall_success'] else 'FAILED'}")
    print(f"  RGB Image:    {results['rgb']['msg']}")
    print(f"  Depth Map:    {results['depth']['msg']}")
    print(f"  K (RGB):      {results['intrinsics_rgb']['msg']}")
    print(f"  K (Depth):    {results['intrinsics_depth']['msg']}")
    print(f"  Pose (RGB):   {results['pose_rgb']['msg']}")
    print(f"  Pose (Depth): {results['pose_depth']['msg']}")
    print(f"  Tracking:     {results['tracking']['msg']}")
    print("-" * 35)
