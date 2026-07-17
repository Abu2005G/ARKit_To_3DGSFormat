"""
export_single_frame.py

Isolated validation script to unproject and export one single frame's point cloud
and COLMAP camera files using only the corrected double-flip coordinate transformation.
Specifically allows bypassing the depth discontinuity noise filter to isolate the
rotation alignment fix in SuperSplat.
"""

import sys
import argparse
from pathlib import Path
import tempfile
import shutil
import numpy as np
import open3d as o3d

# Set up paths
sys.path.append(str(Path(__file__).parent.resolve()))

from preprocessing.io_utils import unzip_dataset
from preprocessing.dataset import ARKitDataset
from preprocessing.pointcloud import PointCloudBuilder
from preprocessing.geometry import depth_to_camera, camera_to_world
from preprocessing.gaussian_init import GaussianInitializer


def export_single(zip_path: Path, output_root: Path, run_noise_filter: bool = False):
    print("=" * 60)
    print("Exporting Single Frame for Transform Validation")
    print("=" * 60)

    # Temporary directory for extraction
    tmp_extract = Path(tempfile.gettempdir()) / "ark_single_frame_val"
    dataset_root = unzip_dataset(zip_path, tmp_extract)

    # Load dataset
    dataset = ARKitDataset(dataset_root)
    # Extract only Frame 0
    frame = dataset[0]
    
    # Destination directory
    project_output = output_root / "single_frame_val"
    project_output.mkdir(parents=True, exist_ok=True)
    
    # Unproject Point Cloud
    print("\n[Step 1] Backprojecting depth map to OpenCV camera space...")
    pts_cv = depth_to_camera(frame.depth, frame.camera_depth.intrinsics)
    
    # Filter only invalid (NaN/range) points
    depth_val = pts_cv[:, 2]
    valid_mask = np.isfinite(depth_val) & (depth_val >= 0.1) & (depth_val <= 5.0)
    
    # If noise filter is requested:
    if run_noise_filter:
        print("  Applying depth discontinuity filtering...")
        # Get threshold from config or default it
        thresh = 0.05
        diff_r = np.abs(frame.depth[:, 1:] - frame.depth[:, :-1])
        diff_d = np.abs(frame.depth[1:, :] - frame.depth[:-1, :])
        thresh_r = frame.depth[:, 1:] * thresh
        thresh_d = frame.depth[1:, :] * thresh
        
        invalid = np.zeros_like(frame.depth, dtype=bool)
        invalid[:, 1:] |= (diff_r > thresh_r)
        invalid[:, :-1] |= (diff_r > thresh_r)
        invalid[1:, :] |= (diff_d > thresh_d)
        invalid[:-1, :] |= (diff_d > thresh_d)
        valid_mask &= (~invalid.flatten())
        
    valid_pts_cv = pts_cv[valid_mask]
    
    # Apply Y, Z flip to ARKit camera space
    pts_ark = valid_pts_cv.copy()
    pts_ark[:, 1] = -pts_ark[:, 1]
    pts_ark[:, 2] = -pts_ark[:, 2]
    
    # Transform to world space using ARKit camera pose
    pts_world_cv = camera_to_world(pts_ark, frame.camera_rgb.pose)
    
    # Get colors
    builder = PointCloudBuilder()
    colors = builder._sample_colors(frame, valid_mask)
    
    # Build open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world_cv.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    
    # Format COLMAP Export
    print("\n[Step 2] Formatting and writing COLMAP files (cameras.txt, images.txt)...")
    gs_init = GaussianInitializer(dataset, project_output)
    
    # We will invoke export_dataset but only for Frame 0 (index [0])
    # Note: export_dataset internally uses our modified double-flip extrinsic matrix conversion:
    # T_c2w_o3d = Y @ T_c2w_ark @ Y
    gs_init.export_dataset(pcd, indices=[0])
    
    print("\n" + "=" * 60)
    print("SUCCESS: Single frame COLMAP files exported to:")
    print(f"  {project_output}")
    print("You can now load this sparse directory directly in SuperSplat.")
    print("=" * 60)
    
    # Cleanup extract dir
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated transform validation script")
    parser.add_argument("--zip", type=str, default="Capture-1.zip", help="Path to input ARKit ZIP dataset")
    parser.add_argument("--out", type=str, default="validation_outputs", help="Output root directory")
    parser.add_argument("--noise_filter", action="store_true", help="Apply depth discontinuity noise filtering")
    args = parser.parse_args()

    zip_path = Path(args.zip).expanduser()
    out_dir = Path(args.out).expanduser()
    
    if not zip_path.is_file():
        print(f"Error: ZIP file not found: {zip_path}")
        sys.exit(1)
        
    export_single(zip_path, out_dir, args.noise_filter)
