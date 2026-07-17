"""
tsdf.py

Implements TSDFIntegrator class for performing Signed Distance Function (TSDF)
volumetric integration of multi-frame RGB-D datasets using Open3D's scalable voxel volume.
"""

import numpy as np
import open3d as o3d
from typing import List, Optional

from preprocessing.rgbd_types import RGBDFrame
from preprocessing.dataset import ARKitDataset
from preprocessing.config import TSDF_VOXEL_LENGTH, TSDF_SDF_TRUNC, MAX_DEPTH_METERS


class TSDFIntegrator:
    """
    TSDFIntegrator accumulates RGBDFrame inputs into a global scalable voxel volume
    using truncated signed distance functions.
    """

    def __init__(
        self,
        voxel_length: float = TSDF_VOXEL_LENGTH,
        sdf_trunc: float = TSDF_SDF_TRUNC,
        color_type: o3d.pipelines.integration.TSDFVolumeColorType = 
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    ) -> None:
        """
        Parameters
        ----------
        voxel_length : float
            Voxel grid resolution in meters (e.g. 0.01 corresponds to 1cm).
        sdf_trunc : float
            TSDF truncation distance (meters). Suggest 3-4x voxel length.
        color_type : o3d.pipelines.integration.TSDFVolumeColorType
            Method for storing color inside voxel grid.
        """
        # Issue 5: Check truncation ratio
        from preprocessing.config import TSDF_MIN_TRUNC_RATIO
        ratio = sdf_trunc / voxel_length if voxel_length > 0 else 0
        if ratio < TSDF_MIN_TRUNC_RATIO:
            old_trunc = sdf_trunc
            sdf_trunc = voxel_length * TSDF_MIN_TRUNC_RATIO
            print(f"  ⚠ TSDF sdf_trunc/voxel_length ratio was {ratio:.1f}x (below {TSDF_MIN_TRUNC_RATIO:.0f}x minimum).")
            print(f"    Auto-adjusted sdf_trunc: {old_trunc:.4f} → {sdf_trunc:.4f} m")

        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=color_type
        )

    def integrate_frame(self, frame: RGBDFrame) -> None:
        """
        Integrates a single RGBDFrame into the volume.

        Mathematical Explanation:
            Let T_c2w be the camera-to-world transform provided by ARKit (OpenGL convention: +Y up, -Z forward).
            Open3D expects OpenCV/Open3D convention (+Y down, +Z forward).
            We apply the flipping transform Y = diag(1, -1, -1):
                T_c2w_o3d = T_c2w @ Y
            To obtain the extrinsic camera parameter (world-to-camera) expected by Open3D:
                T_w2c_o3d = inv(T_c2w_o3d) = Y @ inv(T_c2w)

        Parameters
        ----------
        frame : RGBDFrame
        """
        # Convert NumPy RGB (UINT8) and Depth (Float32 in meters) to Open3D images
        import cv2
        h_d, w_d = frame.depth.shape
        rgb_resized = cv2.resize(frame.rgb, (w_d, h_d), interpolation=cv2.INTER_LINEAR)
        o3d_color = o3d.geometry.Image(rgb_resized)
        o3d_depth = o3d.geometry.Image(frame.depth.astype(np.float32))

        # Create Open3D RGBDImage (no scaling down needed since depth is already in meters)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0,
            depth_trunc=MAX_DEPTH_METERS,
            convert_rgb_to_intensity=False
        )

        # Set up camera intrinsics
        h_d, w_d = frame.depth.shape
        K = frame.camera_depth.intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=w_d,
            height=h_d,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )

        # Transform camera pose to Open3D extrinsic conventions
        T_c2w = frame.camera_depth.pose
        Y = np.diag([1.0, -1.0, -1.0, 1.0])
        # Open3D expects world-to-camera matrix (inv(pose))
        T_w2c = np.linalg.inv(T_c2w @ Y)

        # Integrate
        self.volume.integrate(rgbd_image, o3d_intrinsics, T_w2c)

    def integrate_dataset(
        self,
        dataset: ARKitDataset,
        indices: Optional[List[int]] = None
    ) -> None:
        """
        Sequentially integrates a list of frame indices from a dataset.

        Parameters
        ----------
        dataset : ARKitDataset
        indices : Optional[List[int]]
            Indices to integrate. If None, integrates all frames.
        """
        if indices is None:
            indices = list(range(len(dataset)))

        print(f"Integrating {len(indices)} frames into TSDF volume...")
        for count, idx in enumerate(indices):
            if count % 20 == 0:
                print(f"Integrating frame {count}/{len(indices)}...")
            frame = dataset[idx]
            self.integrate_frame(frame)
        print("TSDF Integration finished.")

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        Extracts a consolidated point cloud from the integrated TSDF volume.
        """
        return self.volume.extract_point_cloud()

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        Runs marching cubes to extract a TriangleMesh from the TSDF volume.
        """
        return self.volume.extract_triangle_mesh()
