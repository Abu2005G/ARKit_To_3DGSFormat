"""
fusion.py

Provides the FusionEngine class to aggregate and fuse multiple synchronized
RGB-D frames into a single consolidated, refined 3D point cloud.
"""

import numpy as np
import open3d as o3d
from typing import Optional, List

from preprocessing.dataset import ARKitDataset
from preprocessing.pointcloud import PointCloudBuilder
from preprocessing.refinement import voxel_downsample, statistical_outlier_removal
from preprocessing.config import VOXEL_SIZE, SOR_NEIGHBORS, SOR_STD_RATIO


class FusionEngine:
    """
    FusionEngine handles sequential accumulation of multi-frame point clouds,
    downsampling, and statistical outlier filtering.
    """

    def __init__(
        self,
        dataset: ARKitDataset,
        pointcloud_builder: Optional[PointCloudBuilder] = None,
    ) -> None:
        """
        Parameters
        ----------
        dataset : ARKitDataset
            Dataset instance containing RGB-D frames.
        pointcloud_builder : Optional[PointCloudBuilder]
            Builder instance to generate point clouds from frames.
            If None, a default PointCloudBuilder will be created.
        """
        self.dataset = dataset
        self.builder = pointcloud_builder or PointCloudBuilder()

    def fuse(
        self,
        indices: Optional[List[int]] = None,
        voxel_size: float = VOXEL_SIZE,
        run_sor: bool = True,
    ) -> o3d.geometry.PointCloud:
        """
        Fuses the selected frames into a single downsampled and filtered point cloud.

        Parameters
        ----------
        indices : Optional[List[int]]
            Indices of frames to fuse. If None, fuses all dataset frames.
        voxel_size : float
            Voxel size for intermediate downsampling.
        run_sor : bool
            Whether to run statistical outlier removal on the final point cloud.

        Returns
        -------
        o3d.geometry.PointCloud : The consolidated world-space point cloud.
        """
        if indices is None:
            indices = list(range(len(self.dataset)))

        accumulated_cloud = o3d.geometry.PointCloud()

        print(f"Fusing {len(indices)} frames...")

        for i, idx in enumerate(indices):
            # Load frame
            frame = self.dataset[idx]

            # Reconstruct frame point cloud
            frame_cloud = self.builder.build(frame)
            
            if len(frame_cloud.points) == 0:
                continue

            # Accumulate
            accumulated_cloud += frame_cloud

            # Voxel downsample occasionally to save RAM if accumulating many frames
            if i > 0 and i % 50 == 0 and voxel_size > 0:
                accumulated_cloud = voxel_downsample(accumulated_cloud, voxel_size)

        # Final downsampling
        if voxel_size > 0 and len(accumulated_cloud.points) > 0:
            accumulated_cloud = voxel_downsample(accumulated_cloud, voxel_size)

        # Final filtering
        if run_sor and len(accumulated_cloud.points) > 0:
            print("Applying Statistical Outlier Removal...")
            accumulated_cloud, _ = statistical_outlier_removal(
                accumulated_cloud,
                nb_neighbors=SOR_NEIGHBORS,
                std_ratio=SOR_STD_RATIO
            )

        print(f"Fusion complete. Final point cloud contains {len(accumulated_cloud.points)} points.")
        return accumulated_cloud
