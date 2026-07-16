"""
io_utils.py

Input/Output utility functions for loading datasets (unzipping, loading rgb,
loading depth32f, json metadata) and writing 3D reconstruction outputs.
"""

from pathlib import Path
import json
import zipfile
import shutil
import numpy as np
from PIL import Image
import open3d as o3d


def unzip_dataset(zip_path: Path, extract_dir: Path) -> Path:
    """
    Unzips an ARKit zip file to the specified output folder.
    Recursively locates the dataset root directory (where images, depth,
    and metadata folders exist).

    Parameters
    ----------
    zip_path : Path
        Path to the input ZIP file.
    extract_dir : Path
        Directory where ZIP contents will be extracted.

    Returns
    -------
    Path : The local dataset root containing 'images', 'depth', 'metadata'.
    """
    zip_path = Path(zip_path).resolve()
    extract_dir = Path(extract_dir).resolve()

    if not zip_path.is_file():
        raise FileNotFoundError(f"Input ZIP file not found: {zip_path}")

    print(f"Extracting {zip_path} to {extract_dir}...")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=False)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Walk through the extracted files to find where 'images', 'depth', 'metadata' are.
    # Some zip exports contain a nested folder, e.g., 'dataset/images/...'
    # We find the folder that contains at least 'images' or 'metadata'.
    for path in extract_dir.rglob("images"):
        if path.is_dir():
            # Parent is the dataset root
            return path.parent

    # Fallback to extract_dir if no 'images' folder is found
    return extract_dir


def frame_name(index: int) -> str:
    """
    Returns the zero-padded string representation of a frame index.
    """
    return f"frame_{index:06d}"


def rgb_path(image_dir: Path, index: int) -> Path:
    """
    Returns the path to the RGB image.
    """
    return Path(image_dir) / f"{frame_name(index)}.png"


def depth_path(depth_dir: Path, index: int) -> Path:
    """
    Returns the path to the depth32f map.
    """
    return Path(depth_dir) / f"{frame_name(index)}.depth32f"


def metadata_path(metadata_dir: Path, index: int) -> Path:
    """
    Returns the path to the per-frame metadata JSON.
    """
    return Path(metadata_dir) / f"{frame_name(index)}.json"


def load_rgb(path: Path) -> np.ndarray:
    """
    Loads an RGB PNG image and returns it as a NumPy array.

    Parameters
    ----------
    path : Path

    Returns
    -------
    np.ndarray : HxW3 color array (uint8)
    """
    return np.array(Image.open(path))


def load_depth(path: Path, width: int = 256, height: int = 144) -> np.ndarray:
    """
    Loads raw ARKit depth data from a .depth32f file.
    Files are stored as raw float32 streams.

    Parameters
    ----------
    path : Path
    width : int, default 256
    height : int, default 144

    Returns
    -------
    np.ndarray : HxW depth map in meters (float32)
    """
    if not path.is_file():
        raise FileNotFoundError(f"Depth file not found: {path}")

    depth = np.fromfile(path, dtype=np.float32)
    
    # Validation of file size
    expected_elements = width * height
    if depth.size != expected_elements:
        raise ValueError(
            f"Expected {expected_elements} elements in depth map, "
            f"but got {depth.size} from {path}."
        )

    return depth.reshape(height, width)


def load_metadata(path: Path) -> dict:
    """
    Loads per-frame metadata stored in JSON.
    Converts intrinsics and transformation matrices to numpy arrays.

    Parameters
    ----------
    path : Path

    Returns
    -------
    dict
    """
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(path, "r") as f:
        meta = json.load(f)

    if "camera_intrinsics" in meta:
        meta["camera_intrinsics"] = np.array(meta["camera_intrinsics"], dtype=np.float32).reshape(3, 3)
    if "camera_transform" in meta:
        meta["camera_transform"] = np.array(meta["camera_transform"], dtype=np.float32).reshape(4, 4)

    return meta


def save_pointcloud(pointcloud: o3d.geometry.PointCloud, path: Path) -> None:
    """
    Saves an Open3D PointCloud structure to PLY or PCD format.

    Parameters
    ----------
    pointcloud : o3d.geometry.PointCloud
    path : Path
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_point_cloud(str(path), pointcloud)
    if not success:
        raise IOError(f"Failed to write point cloud to {path}")


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    """
    Saves an Open3D TriangleMesh to PLY or OBJ format.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
    path : Path
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_triangle_mesh(str(path), mesh)
    if not success:
        raise IOError(f"Failed to write mesh to {path}")
