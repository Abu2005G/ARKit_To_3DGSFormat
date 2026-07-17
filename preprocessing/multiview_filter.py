"""
multiview_filter.py

Provides MultiViewDepthFilter to perform extreme outlier/reflection removal
by enforcing multi-view depth and free-space projection consistency.
"""

import numpy as np
import open3d as o3d
from typing import List, Optional
from preprocessing.dataset import ARKitDataset


class MultiViewDepthFilter:
    """
    MultiViewDepthFilter checks the consistency of 3D points by projecting them
    back into the camera frames and matching their depths against LiDAR depth maps.
    Directly inspired by SR-LIVO's re-projection error and observation distance validation.
    """

    def __init__(
        self,
        dataset: ARKitDataset,
        depth_tolerance: float = 0.05,
        max_violation_ratio: float = 0.25,
        min_consistency_views: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        dataset : ARKitDataset
            The dataset containing depth maps and camera poses.
        depth_tolerance : float
            Tolerance for depth consistency (meters) or relative threshold.
        max_violation_ratio : float
            Max fraction of views where the point violates free space before being pruned.
        min_consistency_views : int
            Minimum number of views where the point must be consistent with the surface.
        """
        self.dataset = dataset
        self.depth_tolerance = depth_tolerance
        self.max_violation_ratio = max_violation_ratio
        self.min_consistency_views = min_consistency_views

    def filter_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        indices: Optional[List[int]] = None
    ) -> o3d.geometry.PointCloud:
        """
        Filters out points that violate free space or lack consistent surface coverage.

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
        indices : Optional[List[int]]
            Indices of frames to check. If None, checks all frames.

        Returns
        -------
        o3d.geometry.PointCloud : The curated, denoised point cloud.
        """
        if len(pcd.points) == 0:
            return pcd

        if indices is None:
            indices = list(range(len(self.dataset)))

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        num_points = len(points)

        # Allocate statistics counters
        # We track how many times a point fell inside a camera frustum (total_views),
        # how many times it was consistent with the depth map (consistent_views),
        # and how many times it violated free space (free_space_violations).
        total_views = np.zeros(num_points, dtype=np.int32)
        consistent_views = np.zeros(num_points, dtype=np.int32)
        free_space_violations = np.zeros(num_points, dtype=np.int32)

        Y = np.diag([1.0, -1.0, -1.0, 1.0])

        print(f"Running Multi-View Consistency Filter on {num_points} points over {len(indices)} frames...")

        for idx in indices:
            frame = self.dataset[idx]

            # 1. Compute world-to-camera matrix (OpenCV convention)
            T_c2w_ark = frame.camera_depth.pose
            T_c2w_o3d = T_c2w_ark @ Y
            T_w2c_o3d = np.linalg.inv(T_c2w_o3d)
            R_w2c = T_w2c_o3d[:3, :3]
            t_w2c = T_w2c_o3d[:3, 3]

            # 2. Transform all points to camera space
            pts_c = (R_w2c @ points.T).T + t_w2c

            # 3. Project to pixels
            depths_c = pts_c[:, 2]
            
            # Avoid divide-by-zero or backward projection
            valid_z = depths_c > 0.01
            if not np.any(valid_z):
                continue

            K = frame.camera_depth.intrinsics
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            h, w = frame.depth.shape

            # Project
            u = (fx * pts_c[:, 0] / (depths_c + 1e-8)) + cx
            v = (fy * pts_c[:, 1] / (depths_c + 1e-8)) + cy

            # Find points inside image boundaries
            in_bounds = (
                valid_z &
                (u >= 0) & (u < w - 0.5) &
                (v >= 0) & (v < h - 0.5)
            )
            
            if not np.any(in_bounds):
                continue

            # Convert to integer pixel coordinates
            u_idx = np.round(u[in_bounds]).astype(np.int32)
            v_idx = np.round(v[in_bounds]).astype(np.int32)
            d_proj = depths_c[in_bounds]

            # Read observed depth from depth map
            d_obs = frame.depth[v_idx, u_idx]

            # Valid depth measurements in LiDAR depth map
            valid_depth = (d_obs > 0.01) & np.isfinite(d_obs)
            
            # Update statistcs for points inside bounds with valid depth
            valid_points_indices = np.where(in_bounds)[0][valid_depth]
            if len(valid_points_indices) == 0:
                continue

            d_proj_val = d_proj[valid_depth]
            d_obs_val = d_obs[valid_depth]

            # Determine dynamic thresholds per point based on depth (larger tolerances further away)
            tolerances = np.maximum(self.depth_tolerance, 0.03 * d_obs_val)

            # Check logic:
            # - Free space violation: point is in front of the observed surface by more than tolerance
            violated = d_proj_val < (d_obs_val - tolerances)
            # - Surface consistency: point lies on/near the surface
            consistent = np.abs(d_proj_val - d_obs_val) <= tolerances

            # Update stats
            total_views[valid_points_indices] += 1
            free_space_violations[valid_points_indices[violated]] += 1
            consistent_views[valid_points_indices[consistent]] += 1

        # 4. Filter criteria
        # Calculate violation ratio (ratio of free-space violations to visible views)
        violation_ratio = np.zeros(num_points, dtype=np.float32)
        has_views = total_views > 0
        violation_ratio[has_views] = free_space_violations[has_views] / total_views[has_views]

        # Inlier conditions:
        # 1. Total violations ratio is within acceptable limits.
        # 2. Point has sufficient consistent surface observations (min_consistency_views).
        keep_mask = (
            (consistent_views >= self.min_consistency_views) &
            (violation_ratio <= self.max_violation_ratio)
        )

        inliers_count = np.sum(keep_mask)
        print(f"Multi-View Filter kept {inliers_count} / {num_points} points ({inliers_count/num_points*100:.1f}%).")
        
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(points[keep_mask])
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[keep_mask])

        return filtered_pcd
