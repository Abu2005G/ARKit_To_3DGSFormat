"""
mesh.py

Implements MeshReconstructor class supporting Poisson Surface Reconstruction and
Ball Pivoting Algorithm (BPA) to create 3D watertight meshes from 3D point clouds.
"""

from typing import List, Tuple, Optional
import open3d as o3d
import numpy as np

from preprocessing.refinement import estimate_normals
from preprocessing.config import POISSON_DEPTH, BALL_PIVOTING_RADII
from preprocessing.io_utils import save_mesh


class MeshReconstructor:
    """
    MeshReconstructor wraps Open3D surface reconstruction methods.
    Points shape and orientation (normals) must be present for these algorithms.
    """

    @staticmethod
    def reconstruct_poisson(
        pcd: o3d.geometry.PointCloud,
        depth: int = POISSON_DEPTH,
        width: float = 0.0,
        scale: float = 1.1,
        linear_fit: bool = False,
    ) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        """
        Reconstructs a TriangleMesh from a PointCloud using Poisson Surface Reconstruction.

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
        depth : int
            Octree depth. Higher values define finer details but consume more RAM/CPU.
        width : float
            Target voxel width. If > 0, depth is ignored.
        scale : float
            Scale of the bounding box.
        linear_fit : bool
            If true, fits the surface using linear interpolation.

        Returns
        -------
        mesh : o3d.geometry.TriangleMesh
        densities : np.ndarray
            Vertex density values, useful for filtering unreliable surface faces.
        """
        # Ensure point cloud has normals estimated
        if not pcd.has_normals():
            print("PointCloud does not have normals. Estimating normals...")
            estimate_normals(pcd)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=linear_fit
        )

        return mesh, np.asarray(densities)

    @staticmethod
    def reconstruct_ball_pivoting(
        pcd: o3d.geometry.PointCloud,
        radii: Optional[List[float]] = None,
    ) -> o3d.geometry.TriangleMesh:
        """
        Reconstructs a TriangleMesh from a PointCloud using the Ball Pivoting Algorithm (BPA).

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
        radii : Optional[List[float]]
            Radii of balls pivoting on point cloud surface.
            If None, uses default config values.

        Returns
        -------
        o3d.geometry.TriangleMesh
        """
        if radii is None:
            radii = BALL_PIVOTING_RADII

        # Ensure point cloud has normals estimated
        if not pcd.has_normals():
            print("PointCloud does not have normals. Estimating normals...")
            estimate_normals(pcd)

        o3d_radii = o3d.utility.DoubleVector(radii)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d_radii
        )
        return mesh

    @staticmethod
    def filter_low_density_vertices(
        mesh: o3d.geometry.TriangleMesh,
        densities: np.ndarray,
        quantile: float = 0.05
    ) -> o3d.geometry.TriangleMesh:
        """
        Filters out low-density vertices from Poisson Surface Reconstruction
        to clean up boundary/spurious faces. All vertices with density density below
        the specified quantile are removed.

        Parameters
        ----------
        mesh : o3d.geometry.TriangleMesh
        densities : np.ndarray
            Density arrays returned by Poisson reconstruction.
        quantile : float
            Quantile of densities threshold [0.0, 1.0].

        Returns
        -------
        o3d.geometry.TriangleMesh : Filtered mesh.
        """
        if len(densities) == 0:
            return mesh

        threshold = np.quantile(densities, quantile)
        vertices_to_remove = densities < threshold
        
        mesh.remove_vertices_by_mask(vertices_to_remove)
        return mesh

    @staticmethod
    def export(mesh: o3d.geometry.TriangleMesh, filename: str) -> None:
        """
        Saves a TriangleMesh to disk.

        Parameters
        ----------
        mesh : o3d.geometry.TriangleMesh
        filename : str
        """
        from pathlib import Path
        save_mesh(mesh, Path(filename))
