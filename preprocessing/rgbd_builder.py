"""
rgbd_builder.py

Implements RGBDBuilder class to dynamically load RGB images, path depth32f depth maps,
and per-frame json metadata, coordinate alignment, and build RGBDFrame structures.
"""

from pathlib import Path
import numpy as np

from preprocessing.io_utils import (
    load_rgb,
    load_depth,
    load_metadata,
    rgb_path,
    depth_path,
    metadata_path,
)
from preprocessing.rgbd_types import RGBDFrame, Camera


class RGBDBuilder:
    """
    RGBDBuilder constructs validation-ready RGBDFrame objects or Camera properties
    from raw frame index inputs and dataset subdirectory references.
    """

    def __init__(
        self,
        image_dir: Path,
        depth_dir: Path,
        metadata_dir: Path
    ) -> None:
        """
        Parameters
        ----------
        image_dir : Path
            Subfolder containing image PNG files.
        depth_dir : Path
            Subfolder containing depth32f binary files.
        metadata_dir : Path
            Subfolder containing frame metadata JSON files.
        """
        self.image_dir = Path(image_dir)
        self.depth_dir = Path(depth_dir)
        self.metadata_dir = Path(metadata_dir)

    def build(self, index: int) -> RGBDFrame:
        """
        Constructs a complete RGBDFrame object from the raw frame on disk.

        Parameters
        ----------
        index : int
            Frame sequence index on disk (e.g. 0 matches 'frame_000000.png', etc.)

        Returns
        -------
        RGBDFrame
        """
        rgb_file = rgb_path(self.image_dir, index)
        depth_file = depth_path(self.depth_dir, index)
        meta_file = metadata_path(self.metadata_dir, index)

        rgb = load_rgb(rgb_file)
        meta = load_metadata(meta_file)

        # Get depth map dimensions from metadata
        depth_w = meta["depth_resolution"]["w"]
        depth_h = meta["depth_resolution"]["h"]

        depth = load_depth(depth_file, width=depth_w, height=depth_h)

        # Retrieve intrinsic mapping
        K_rgb = meta["camera_intrinsics"].copy()
        T_c2w = meta["camera_transform"].copy()

        # Capture RGB resolution
        image_w = meta["image_resolution"]["w"]
        image_h = meta["image_resolution"]["h"]

        # Compute scaling metrics for depth intrinsics because depth map is lower resolution
        scale_x = image_w / depth_w
        scale_y = image_h / depth_h

        K_depth = K_rgb.copy()
        K_depth[0, 0] /= scale_x  # fx_depth = fx_rgb / scale_x
        K_depth[1, 1] /= scale_y  # fy_depth = fy_rgb / scale_y
        K_depth[0, 2] /= scale_x  # cx_depth = cx_rgb / scale_x
        K_depth[1, 2] /= scale_y  # cy_depth = cy_rgb / scale_y

        # Build Camera structures
        camera_rgb = Camera(
            intrinsics=K_rgb,
            pose=T_c2w,
            width=image_w,
            height=image_h
        )

        camera_depth = Camera(
            intrinsics=K_depth,
            pose=T_c2w,
            width=depth_w,
            height=depth_h
        )

        return RGBDFrame(
            index=index,
            timestamp=meta["timestamp"],
            tracking_state=meta["tracking_state"],
            rgb=rgb,
            depth=depth,
            camera_rgb=camera_rgb,
            camera_depth=camera_depth
        )
