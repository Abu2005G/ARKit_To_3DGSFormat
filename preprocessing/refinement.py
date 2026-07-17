"""
refinement.py

Implements point cloud refinement operations, including downsampling, statistical
and radius outlier filtering, normal estimation, and RANSAC-based plane segmentation.
"""

from typing import Tuple, List
import open3d as o3d
import numpy as np


def voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    """
    Subsamples a point cloud using a voxel grid.
    Points within the same voxel cell are averaged (centroid).

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
    voxel_size : float
        Voxel size in meters.

    Returns
    -------
    o3d.geometry.PointCloud : Downsampled point cloud.
    """
    if voxel_size <= 0.0:
        raise ValueError(f"Voxel size must be positive, got {voxel_size}")
    return pcd.voxel_down_sample(voxel_size)


def statistical_outlier_removal(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Filters outliers by computing the average distance to its k-nearest neighbors.
    Points with distances larger than the mean plus a standard deviation ratio are removed.

    Mathematical Explanation:
        For each point, distance mean d_i is calculated.
        Points are outliers if:
        d_i > mean(d) + std_ratio * std(d)

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
    nb_neighbors : int
        Number of neighbors to analyze (k).
    std_ratio : float
        Standard deviation multiplier.

    Returns
    -------
    cleaned_pcd : o3d.geometry.PointCloud
    inlier_mask : np.ndarray (N,)
        Boolean array where True indicates the point is an inlier.
    """
    if len(pcd.points) == 0:
        return pcd, np.array([], dtype=bool)

    cleaned_pcd, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    inlier_mask = np.zeros(len(pcd.points), dtype=bool)
    inlier_mask[inlier_indices] = True

    return cleaned_pcd, inlier_mask


def radius_outlier_removal(
    pcd: o3d.geometry.PointCloud,
    nb_points: int = 15,
    radius: float = 0.05,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Filters outlier points that have fewer than the specified neighbor count
    within a sphere of the given radius.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
    nb_points : int
        Minimum neighbors required within the sphere.
    radius : float
        Radius of the neighborhood sphere (meters).

    Returns
    -------
    cleaned_pcd : o3d.geometry.PointCloud
    inlier_mask : np.ndarray (N,)
        Boolean array where True indicates the point is an inlier.
    """
    if len(pcd.points) == 0:
        return pcd, np.array([], dtype=bool)

    cleaned_pcd, inlier_indices = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius
    )

    inlier_mask = np.zeros(len(pcd.points), dtype=bool)
    inlier_mask[inlier_indices] = True

    return cleaned_pcd, inlier_mask


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    k_neighbors: int = 30,
    radius: float = 0.1,
    orient_k: int = None,
) -> None:
    """
    Estimates normals for each point by fitting a local plane to its neighbors.
    Mutates the input PointCloud by setting its normals.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
    k_neighbors : int
        Number of neighbors for KD-tree search.
    radius : float
        Maximum distance threshold.
    orient_k : int
        k for orient_normals_consistent_tangent_plane.
        If None, uses config NORMAL_ORIENT_K (default 20).
        Wider values (15-30) help with thin/concave geometry.
    """
    from preprocessing.config import NORMAL_ORIENT_K
    if orient_k is None:
        orient_k = NORMAL_ORIENT_K

    search_param = o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius,
        max_nn=k_neighbors
    )
    pcd.estimate_normals(search_param=search_param)
    
    # Orient normals consistently using tangent plane propagation
    pcd.orient_normals_consistent_tangent_plane(k=orient_k)


def segment_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> Tuple[np.ndarray, List[int]]:
    """
    Segments the dominant plane (e.g. wall/floor) in the point cloud using RANSAC.

    Mathematical Explanation:
        Finds plane equation ax + by + cz + d = 0 maximizing inliers.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
    distance_threshold : float
        Max distance from point to plane.
    ransac_n : int
        Number of random points to initialize the plane (usually 3).
    num_iterations : int
        Number of RANSAC trials.

    Returns
    -------
    plane_model : np.ndarray (4,)
        Coefficients (a, b, c, d) of the plane equation.
    inliers : List[int]
        Indices of inlier points.
    """
    if len(pcd.points) < ransac_n:
        raise ValueError("PointCloud has too few points for plane segmentation.")

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    return np.array(plane_model, dtype=np.float32), inliers
