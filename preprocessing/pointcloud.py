"""
pointcloud.py

Provides PointCloudBuilder class to construct colored, validated 3D point clouds
from ARKit RGB-D frames. Handles OpenCV-to-ARKit coordinate transformations.
"""

from typing import Tuple
import numpy as np
import open3d as o3d

from preprocessing.rgbd_types import RGBDFrame
from preprocessing.geometry import depth_to_camera, camera_to_world
from preprocessing.config import MIN_DEPTH_METERS, MAX_DEPTH_METERS


class PointCloudBuilder:
    """
    PointCloudBuilder handles depth map backprojection, color sampling, and
    transforming points from camera coordinates into world coordinate space.
    """

    def __init__(
        self,
        min_depth: float = MIN_DEPTH_METERS,
        max_depth: float = MAX_DEPTH_METERS,
    ) -> None:
        """
        Parameters
        ----------
        min_depth : float
            Minimum depth threshold in meters.
        max_depth : float
            Maximum depth threshold in meters.
        """
        self.min_depth = min_depth
        self.max_depth = max_depth

    def build(self, frame: RGBDFrame) -> o3d.geometry.PointCloud:
        """
        Generates an Open3D PointCloud from a given RGBDFrame.

        Mathematical Explanation:
            1. Vertices in Camera Space are computed using intrinsics:
               z = depth
               x = (u - cx) * z / fx
               y = (v - cy) * z / fy
            2. OpenCV camera convention (+X right, +Y down, +Z forward) is converted
               to ARKit camera convention (+X right, +Y up, -Z forward):
               [x_ark, y_ark, z_ark] = [x, -y, -z]
            3. World points are computed using the homogeneous camera pose T_c2w:
               P_world = T_c2w @ P_ark_homogeneous

        Parameters
        ----------
        frame : RGBDFrame

        Returns
        -------
        o3d.geometry.PointCloud
        """
        # 1. Backproject depth map to OpenCV camera space
        camera_points = depth_to_camera(frame.depth, frame.camera_depth.intrinsics)

        # 2. Filter invalid points (NaN, Inf, out of range depth)
        depth_val = camera_points[:, 2]
        valid_mask = (
            np.isfinite(depth_val) & 
            (depth_val >= self.min_depth) & 
            (depth_val <= self.max_depth)
        )

        # Depth discontinuity relative gradient filtering
        from preprocessing.config import DEPTH_DISCONTINUITY_THRESHOLD
        if DEPTH_DISCONTINUITY_THRESHOLD > 0:
            diff_r = np.abs(frame.depth[:, 1:] - frame.depth[:, :-1])
            diff_d = np.abs(frame.depth[1:, :] - frame.depth[:-1, :])
            thresh_r = frame.depth[:, 1:] * DEPTH_DISCONTINUITY_THRESHOLD
            thresh_d = frame.depth[1:, :] * DEPTH_DISCONTINUITY_THRESHOLD
            
            invalid = np.zeros_like(frame.depth, dtype=bool)
            invalid[:, 1:] |= (diff_r > thresh_r)
            invalid[:, :-1] |= (diff_r > thresh_r)
            invalid[1:, :] |= (diff_d > thresh_d)
            invalid[:-1, :] |= (diff_d > thresh_d)
            
            valid_mask &= (~invalid.flatten())

        valid_camera_points = camera_points[valid_mask]
        if valid_camera_points.shape[0] == 0:
            # Return empty PointCloud
            return o3d.geometry.PointCloud()

        # 3. Convert from OpenCV (+X right, +Y down, +Z forward) to ARKit (+X right, +Y up, -Z forward)
        ark_camera_points = valid_camera_points.copy()
        ark_camera_points[:, 1] = -ark_camera_points[:, 1]  # flip Y
        ark_camera_points[:, 2] = -ark_camera_points[:, 2]  # flip Z

        # 4. Transform from ARKit camera space into world coordinates
        world_points = camera_to_world(ark_camera_points, frame.camera_rgb.pose)

        # 5. Sample colors from the RGB image
        colors = self._sample_colors(frame, valid_mask)

        # 6. Construct Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        return pcd

    def _sample_colors(
        self,
        frame: RGBDFrame,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Samples colors from the RGB image corresponding to the depth map points.

        Parameters
        ----------
        frame : RGBDFrame
        valid_mask : np.ndarray (H * W,)
            Boolean mask indicating which depth pixels are valid.

        Returns
        -------
        np.ndarray (N, 3) : RGB colors scaled to [0, 1] range.
        """
        h_d, w_d = frame.depth.shape
        h_rgb, w_rgb = frame.rgb.shape[:2]

        # Calculate pixel map coordinates of depth map
        u_d, v_d = np.meshgrid(
            np.arange(w_d),
            np.arange(h_d),
            indexing="xy"
        )
        
        # Flatten
        u_flat = u_d.flatten()
        v_flat = v_d.flatten()

        # Apply mask
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]

        # Scale coordinates to the RGB resolution
        scale_x = w_rgb / w_d
        scale_y = h_rgb / h_d

        u_rgb = np.clip(
            np.round(u_valid * scale_x).astype(int),
            0,
            w_rgb - 1
        )
        v_rgb = np.clip(
            np.round(v_valid * scale_y).astype(int),
            0,
            h_rgb - 1
        )

        # Sample colors
        sampled_colors = frame.rgb[v_rgb, u_rgb]

        # Normalize to [0.0, 1.0]
        return sampled_colors.astype(np.float32) / 255.0
