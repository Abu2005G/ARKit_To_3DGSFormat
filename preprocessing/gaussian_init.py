"""
gaussian_init.py

Implements initialization routines for 3D Gaussian Splatting (3DGS).
Generates a COLMAP-compatible structure (sparse/0/cameras.txt, sparse/0/images.txt,
sparse/0/points3D.txt, points3D.ply) and prepares the images directory.
"""

import shutil
from pathlib import Path
import numpy as np
import open3d as o3d
from typing import List, Optional

from preprocessing.rgbd_types import RGBDFrame
from preprocessing.dataset import ARKitDataset
from preprocessing.io_utils import save_pointcloud


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to a scalar-first quaternion (qw, qx, qy, qz).
    Ensures mathematical accuracy and handles numerical stability.

    Parameters
    ----------
    R : np.ndarray (3, 3)

    Returns
    -------
    np.ndarray (4,) : [qw, qx, qy, qz]
    """
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    # Normalize to avoid unit float errors
    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    return q / np.linalg.norm(q)


class GaussianInitializer:
    """
    GaussianInitializer formats the preprocessing library results into
    GraphDECO 3DGS compatible Colmap dataset configurations.
    """

    def __init__(
        self,
        dataset: ARKitDataset,
        output_dir: Path,
    ) -> None:
        """
        Parameters
        ----------
        dataset : ARKitDataset
        output_dir : Path
            Root output folder for the 3DGS dataset.
        """
        self.dataset = dataset
        self.output_dir = Path(output_dir).resolve()
        
        self.sparse_dir = self.output_dir / "sparse" / "0"
        self.images_dir = self.output_dir / "images"

        # Initialize folders
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def export_dataset(
        self,
        fused_cloud: o3d.geometry.PointCloud,
        indices: Optional[List[int]] = None
    ) -> None:
        """
        Exports the dataset into 3DGS pipeline layout:
         1. Write cameras.txt
         2. Write images.txt and copy source images to output/images/
         3. Write points3D.txt and points3D.ply

        Parameters
        ----------
        fused_cloud : o3d.geometry.PointCloud
            Fused colored point cloud initialization.
        indices : Optional[List[int]]
            Indices of frames to include. Defaults to all.
        """
        if indices is None:
            indices = list(range(len(self.dataset)))

        # ----------------------------
        # 1. Write cameras.txt
        # ----------------------------
        # We assume all frames share intrinsics. We write a single camera model (PINHOLE)
        # using the first frame calibration parameters.
        first_frame = self.dataset[indices[0]]
        K = first_frame.camera_rgb.intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        w, h = first_frame.camera_rgb.width, first_frame.camera_rgb.height

        cameras_file = self.sparse_dir / "cameras.txt"
        with open(cameras_file, "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            # 1: CAMERA_ID, PINHOLE: model, W, H, fx, fy, cx, cy
            f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

        print(f"Saved {cameras_file}")

        # ----------------------------
        # 2. Write images.txt & Copy RGBs
        # ----------------------------
        # GraphDECO 3DGS uses sparse/0/images.txt mapped to relative image names.
        images_file = self.sparse_dir / "images.txt"
        
        with open(images_file, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

            for img_id, idx in enumerate(indices, start=1):
                frame = self.dataset[idx]
                
                # Image filename suffix
                frame_name = f"frame_{idx:06d}.png"
                src_image_path = self.dataset.image_dir / frame_name
                dst_image_path = self.images_dir / frame_name

                # Copy RGB image
                if src_image_path.exists():
                    shutil.copy2(src_image_path, dst_image_path)

                # Camera extrinsic matrix conversion: T_w2c_o3d = inv(T_c2w_ark @ Y)
                T_c2w_ark = frame.camera_rgb.pose
                Y = np.diag([1.0, -1.0, -1.0, 1.0])
                T_c2w_o3d = T_c2w_ark @ Y
                T_w2c_o3d = np.linalg.inv(T_c2w_o3d)

                R_w2c = T_w2c_o3d[:3, :3]
                t_w2c = T_w2c_o3d[:3, 3]

                q = rotation_to_quaternion(R_w2c)

                # Write line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                f.write(f"{img_id} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {t_w2c[0]:.8f} {t_w2c[1]:.8f} {t_w2c[2]:.8f} 1 {frame_name}\n")
                # Write line 2: Empty 2D features (zero padded)
                f.write("\n")

        print(f"Saved {images_file} and copied images to {self.images_dir}")

        # ----------------------------
        # 3. Write points3D.txt & points3D.ply
        # ----------------------------
        # Downsample point cloud if excessively dense to prevent 3DGS over-densification noise
        from preprocessing.config import GS_INIT_VOXEL_SIZE
        if len(fused_cloud.points) > 100000:
            print(f"Downsampling initialization point cloud from {len(fused_cloud.points)} points...")
            fused_cloud = fused_cloud.voxel_down_sample(voxel_size=GS_INIT_VOXEL_SIZE)
            print(f"Downsampled to {len(fused_cloud.points)} points for optimal 3DGS initialization.")

        # Ensure the point cloud has normals so the 3DGS PlyReader does not crash
        if not fused_cloud.has_normals():
            print("Estimating normals for initialization point cloud...")
            fused_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

        # Save points3D.ply directly into the sparse/0 folder
        points_ply_path = self.sparse_dir / "points3D.ply"
        save_pointcloud(fused_cloud, points_ply_path)
        print(f"Saved {points_ply_path}")

        # Also write the points3D.txt format
        pts = np.asarray(fused_cloud.points)
        cols = np.asarray(fused_cloud.colors)
        points_txt_path = self.sparse_dir / "points3D.txt"

        with open(points_txt_path, "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            for pid, (pt, col) in enumerate(zip(pts, cols), start=1):
                r = int(col[0] * 255)
                g = int(col[1] * 255)
                b = int(col[2] * 255)
                # POINT3D_ID X Y Z R G B ERROR TRACK
                f.write(f"{pid} {pt[0]:.8f} {pt[1]:.8f} {pt[2]:.8f} {r} {g} {b} 0.0\n")

        print(f"Saved {points_txt_path}")
        print("Gaussian Splatting initialization data generated successfully.")
