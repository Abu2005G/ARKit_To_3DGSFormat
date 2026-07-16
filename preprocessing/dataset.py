"""
dataset.py

Implements the ARKitDataset class, representing a collection of RGB-D frames
loaded from a directory with coordinate matching, checks and frame selections.
"""

import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from preprocessing.rgbd_types import RGBDFrame
from preprocessing.rgbd_builder import RGBDBuilder


class ARKitDataset:
    """
    ARKitDataset loads a sequence of RGB-D frames from an unzipped ARKit iPhone capture.
    It supports retrieving single frames, iterating, or slicing the dataset.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        frame_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        dataset_dir : Union[str, Path]
            Path to the folder containing 'images', 'depth', 'metadata'.
        frame_indices : Optional[List[int]]
            Specific frame indices to load. If None, automatically loads all frames found.
        """
        self.dataset_dir = Path(dataset_dir).resolve()
        self.image_dir = self.dataset_dir / "images"
        
        # Check standard layout first (separated folders)
        depth_subdir = self.dataset_dir / "depth"
        metadata_subdir = self.dataset_dir / "metadata"

        if self.image_dir.exists() and depth_subdir.exists() and metadata_subdir.exists():
            self.depth_dir = depth_subdir
            self.metadata_dir = metadata_subdir
        elif self.image_dir.exists():
            # Unified format: all frame files (.png, .depth32f, .json) inside images/
            self.depth_dir = self.image_dir
            self.metadata_dir = self.image_dir
        else:
            raise FileNotFoundError(
                f"ARKit dataset directory ('images') not found in: {self.dataset_dir}"
            )

        # Retrieve frame indices from files if not specified
        if frame_indices is None:
            self.indices = self._discover_frame_indices()
        else:
            self.indices = sorted(frame_indices)

        if not self.indices:
            raise ValueError(f"No frames found in dataset directory: {self.dataset_dir}")

        # Instantiate RGBDBuilder
        self.builder = RGBDBuilder(self.image_dir, self.depth_dir, self.metadata_dir)

    def _discover_frame_indices(self) -> List[int]:
        """
        Scans 'images' directory for frame files like 'frame_XXXXXX.png' and returns sorted indices.
        """
        indices = []
        pattern = re.compile(r"frame_(\d+)\.png")
        for file in self.image_dir.glob("frame_*.png"):
            match = pattern.match(file.name)
            if match:
                indices.append(int(match.group(1)))
        return sorted(indices)

    def __len__(self) -> int:
        """
        Returns the number of frames in the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> RGBDFrame:
        """
        Retrieves a single RGBDFrame by dataset index.

        Parameters
        ----------
        idx : int
            Index of the frame inside the current dataset sequence.

        Returns
        -------
        RGBDFrame
        """
        if idx < 0 or idx >= len(self.indices):
            raise IndexError("Dataset index out of bounds.")
        frame_idx = self.indices[idx]
        return self.builder.build(frame_idx)

    def get_frame_by_number(self, frame_num: int) -> RGBDFrame:
        """
        Retrieves a frame directly by its frame file suffix number (e.g. 5 for frame_000005).
        """
        if frame_num not in self.indices:
            raise ValueError(f"Frame number {frame_num} not found in this dataset.")
        return self.builder.build(frame_num)
