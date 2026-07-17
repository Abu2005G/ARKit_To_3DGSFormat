"""
config.py

Configuration parameters and settings for the ARKit LiDAR RGB-D Preprocessing Framework.
Provides default values and directory paths.
"""

from pathlib import Path

# ==========================================================
# Dataset Paths
# ==========================================================
DATASET_ROOT = Path("~/Documents/ARKit/MR-1").expanduser()

IMAGE_DIR = DATASET_ROOT / "images"
DEPTH_DIR = DATASET_ROOT / "depth"
METADATA_DIR = DATASET_ROOT / "metadata"

# ==========================================================
# Output Paths
# ==========================================================
DEFAULT_OUTPUT_DIR = Path("outputs").resolve()

# Use TSDF point cloud as initialization for 3DGS.
# False (default): Use raw fused cloud — preserves per-point color/position fidelity from direct depth backprojection.
# True: Use TSDF cloud — smoother but voxel-averaged, better for meshing than splat init.
USE_TSDF_PCD_FOR_3DGS = True

# ==========================================================
# Depth Filtering & Validation Parameters
# ==========================================================
MIN_DEPTH_METERS = 0.2
MAX_DEPTH_METERS = 3.5   # Clip at 3.5m — excludes glass reflections and background noise

# ==========================================================
# Point Cloud Refinement & Voxel Processing
# ==========================================================
# Voxel size for downsampling (meters)
VOXEL_SIZE = 0.005

# Statistical Outlier Removal (SOR) parameters — tighter = more aggressive
SOR_NEIGHBORS = 30
SOR_STD_RATIO = 1.0   # Reduced from 1.5 → removes points 1σ from mean (very aggressive)

# Radius Outlier Removal (ROR) parameters — second pass
ROR_NEIGHBORS = 7     # At least 7 neighbors required in given radius
ROR_RADIUS = 0.04    # Within 4cm sphere

# Third-pass ROR for extreme outlier removal
ROR2_NEIGHBORS = 5
ROR2_RADIUS = 0.06

# Normal Estimation
NORMAL_K_NEIGHBORS = 30
NORMAL_RADIUS = 0.1
NORMAL_ORIENT_K = 20  # k for orient_normals_consistent_tangent_plane (Issue 3)

# ==========================================================
# TSDF Volume Fusion Parameters
# ==========================================================
TSDF_VOXEL_LENGTH = 0.008  # Reduced from 0.01 → finer 8mm voxels = cleaner surfaces
TSDF_SDF_TRUNC = 0.032     # 4x voxel length (required minimum ratio)
TSDF_MIN_TRUNC_RATIO = 4.0  # Minimum sdf_trunc / voxel_length ratio (Issue 5)

# ==========================================================
# Mesh Reconstruction Parameters
# ==========================================================
POISSON_DEPTH = 9
BALL_PIVOTING_RADII = [0.005, 0.01, 0.02, 0.04]  # Fallback if adaptive fails
BPA_RADIUS_MULTIPLIERS = [1.5, 2.0, 3.0]  # Multipliers of avg NN distance (Issue 4)
BPA_MIN_COMPONENT_TRIANGLES = 20  # Min triangles to keep a connected component (Issue 4)

# Camera Coverage & Pose Quality (Issues 1 & 2)
# ==========================================================
COVERAGE_WARN_RATIO = 0.6   # Warn if camera bbox < 60% of scene on any axis
COVERAGE_OFFSET_RATIO = 0.3  # Warn if centroid offset > 30% of scene diagonal
POSE_JUMP_MULTIPLIER = 3.0   # Flag frames with displacement > 3x median

# ==========================================================
# Depth Discontinuity Filtering (Edge Flying Pixels Removal)
# ==========================================================
DEPTH_DISCONTINUITY_THRESHOLD = 0.02  # Tightened to 2% — more aggressive edge flying pixel removal

# ==========================================================
# 3DGS Initialization Downsampling
# ==========================================================
# Voxel size used when downsampling the init point cloud for 3DGS
# Smaller = more points = richer init but more Gaussian competition
GS_INIT_VOXEL_SIZE = 0.03  # 3cm — balanced between coverage and density

