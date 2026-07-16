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

# ==========================================================
# Depth Filtering & Validation Parameters
# ==========================================================
MIN_DEPTH_METERS = 0.1
MAX_DEPTH_METERS = 5.0

# ==========================================================
# Point Cloud Refinement & Voxel Processing
# ==========================================================
# Voxel size for downsampling (meters)
VOXEL_SIZE = 0.01

# Statistical Outlier Removal (SOR) parameters
SOR_NEIGHBORS = 30
SOR_STD_RATIO = 1.5

# Radius Outlier Removal (ROR) parameters
ROR_NEIGHBORS = 15
ROR_RADIUS = 0.05

# Normal Estimation
NORMAL_K_NEIGHBORS = 30
NORMAL_RADIUS = 0.1

# ==========================================================
# TSDF Volume Fusion Parameters
# ==========================================================
TSDF_VOXEL_LENGTH = 0.01  # Voxel size in meters
TSDF_SDF_TRUNC = 0.04     # Truncation distance in meters

# ==========================================================
# Mesh Reconstruction Parameters
# ==========================================================
POISSON_DEPTH = 9
BALL_PIVOTING_RADII = [0.005, 0.01, 0.02, 0.04]
