"""
visualization.py

Implements visualization utilities for 2D color images, depth maps, 3D camera
trajectories (wireframe frustums), point clouds, and mesh reconstructions.
"""

from typing import List, Union
import numpy as np
import open3d as o3d
import cv2


def show_rgb(rgb: np.ndarray, window_name: str = "RGB Image") -> None:
    """
    Displays an RGB image using OpenCV.

    Parameters
    ----------
    rgb : np.ndarray (H, W, 3)
    window_name : str
    """
    # OpenCV expects BGR colormap order
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_depth(depth: np.ndarray, window_name: str = "Depth Map") -> None:
    """
    Displays a depth map visualized with a colormap.

    Parameters
    ----------
    depth : np.ndarray (H, W)
    window_name : str
    """
    valid_mask = np.isfinite(depth) & (depth > 0)
    if not np.any(valid_mask):
        print("Empty or invalid depth image.")
        return

    # Normalize depth for visualization
    min_val = depth[valid_mask].min()
    max_val = depth[valid_mask].max()
    depth_norm = np.zeros_like(depth)
    depth_norm[valid_mask] = (depth[valid_mask] - min_val) / (max_val - min_val + 1e-5)

    # Scale to 0..255 and apply JET colormap
    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    color_map = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

    # Set invalid pixels to black
    color_map[~valid_mask] = 0

    cv2.imshow(window_name, color_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_rgb_depth(rgb: np.ndarray, depth: np.ndarray, window_name: str = "RGB-D Pair") -> None:
    """
    Displays RGB and colormapped Depth maps side-by-side.
    Resizes depth to match RGB resolution height if needed.

    Parameters
    ----------
    rgb : np.ndarray
    depth : np.ndarray
    window_name : str
    """
    h, w = rgb.shape[:2]
    
    # Scale depth visually
    valid_mask = np.isfinite(depth) & (depth > 0)
    depth_norm = np.zeros_like(depth)
    if np.any(valid_mask):
        min_v = depth[valid_mask].min()
        max_v = depth[valid_mask].max()
        depth_norm[valid_mask] = (depth[valid_mask] - min_v) / (max_v - min_v + 1e-5)

    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    depth_color[~valid_mask] = 0

    # Resize depth to RGB size
    depth_color_resized = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_NEAREST)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    stacked = np.hstack((bgr, depth_color_resized))

    # Resize for display if window dimensions are massive (e.g. 4K RGB)
    display_w = 1280
    display_h = int((h / (2 * w)) * display_w)
    stacked_resized = cv2.resize(stacked, (display_w, display_h))

    cv2.imshow(window_name, stacked_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_camera(
    camera_poses: List[np.ndarray],
    frustum_size: float = 0.1,
    window_name: str = "Camera Trajectory"
) -> None:
    """
    Renders camera wireframe frustums in 3D using Open3D.

    Parameters
    ----------
    camera_poses : List[np.ndarray]
        List of 4x4 camera pose matrices (camera-to-world).
    frustum_size : float
        Visual size of camera frustum wireframes in meters.
    window_name : str
    """
    geometries = []

    # Define unit frustum lines in camera space (+X right, -Y up, -Z forward in ARKit OpenGL layout)
    # 5 vertices defining the apex (0,0,0) and the 4 corners of visual plane.
    half_w = frustum_size * 0.5
    half_h = frustum_size * 0.375  # 4:3 aspect ratio representation
    d = -frustum_size

    verts_cam = np.array([
        [0.0, 0.0, 0.0],          # Apex (center of projection)
        [-half_w, half_h, d],     # Top-Left
        [half_w, half_h, d],      # Top-Right
        [half_w, -half_h, d],     # Bottom-Right
        [-half_w, -half_h, d]     # Bottom-Left
    ], dtype=np.float32)

    # Frustum wireframe lines connecting vertices
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4], # lines from apex to corners
        [1, 2], [2, 3], [3, 4], [4, 1]  # edge outline loop
    ]

    for pose in camera_poses:
        # Convert vertices to homogeneous coordinates
        verts_cam_h = np.hstack((verts_cam, np.ones((5, 1), dtype=np.float32)))
        verts_world = (pose @ verts_cam_h.T).T[:, :3]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(verts_world.astype(np.float64))
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Color trajectory lines (gradient from red to blue)
        colors = [[1.0, 0.0, 0.0] for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(line_set)

    # Add origin coordinate system frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frustum_size * 2, origin=[0, 0, 0])
    geometries.append(coord_frame)

    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def show_pointcloud(
    pcd: o3d.geometry.PointCloud,
    window_name: str = "Point Cloud Viewer"
) -> None:
    """
    Renders an Open3D PointCloud geometry object in a GUI window.
    """
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coord_frame], window_name=window_name)


def show_mesh(
    mesh: o3d.geometry.TriangleMesh,
    window_name: str = "Mesh Viewer"
) -> None:
    """
    Renders an Open3D TriangleMesh geometry object in a GUI window.
    """
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh, coord_frame], window_name=window_name)
