"""
test_io.py

Unit tests for dataset input/output operations, verifying depth loading and file conventions.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
import json

from preprocessing.io_utils import (
    load_depth,
    load_metadata,
    frame_name,
    rgb_path,
    depth_path,
    metadata_path,
)
from preprocessing.dataset import ARKitDataset


class TestIOUtils(unittest.TestCase):
    """
    Verifies frame filename logic, depth float32 reading, and JSON calibration loading.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_file_naming_conventions(self) -> None:
        self.assertEqual(frame_name(0), "frame_000000")
        self.assertEqual(frame_name(105), "frame_000105")

        img_dir = self.temp_path / "images"
        self.assertEqual(rgb_path(img_dir, 5), img_dir / "frame_000005.png")
        self.assertEqual(depth_path(img_dir, 42), img_dir / "frame_000042.depth32f")
        self.assertEqual(metadata_path(img_dir, 100), img_dir / "frame_000100.json")

    def test_load_depth(self) -> None:
        # Create a mock depth32f file (256x144 float32 sequence)
        w, h = 256, 144
        depth_data = np.linspace(0.5, 4.5, w * h, dtype=np.float32)
        mock_file = self.temp_path / "frame_000000.depth32f"
        depth_data.tofile(mock_file)

        # Load it and verify shape and content
        loaded = load_depth(mock_file, width=w, height=h)
        self.assertEqual(loaded.shape, (h, w))
        np.testing.assert_allclose(loaded, depth_data.reshape(h, w))

        # Test failure under invalid file dimensions
        with self.assertRaises(ValueError):
            load_depth(mock_file, width=10, height=10)

    def test_load_metadata(self) -> None:
        # Create mock metadata JSON
        mock_meta = {
            "exposure_offset": 0.05,
            "camera_intrinsics": [
                [1000.0, 0.0, 500.0],
                [0.0, 1000.0, 400.0],
                [0.0, 0.0, 1.0]
            ],
            "camera_transform": [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 10.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }
        
        mock_file = self.temp_path / "frame_000000.json"
        with open(mock_file, "w") as f:
            json.dump(mock_meta, f)

        # Load it and verify numeric conversions
        meta_loaded = load_metadata(mock_file)
        self.assertIsInstance(meta_loaded["camera_intrinsics"], np.ndarray)
        self.assertIsInstance(meta_loaded["camera_transform"], np.ndarray)
        
        self.assertEqual(meta_loaded["camera_intrinsics"].shape, (3, 3))
        self.assertEqual(meta_loaded["camera_transform"].shape, (4, 4))
        
        self.assertEqual(meta_loaded["camera_intrinsics"][0, 0], 1000.0)
        self.assertEqual(meta_loaded["camera_transform"][2, 3], 10.0)

    def test_dataset_layout_standard(self) -> None:
        # Create separate folders setup
        img_dir = self.temp_path / "images"
        depth_dir = self.temp_path / "depth"
        meta_dir = self.temp_path / "metadata"
        img_dir.mkdir()
        depth_dir.mkdir()
        meta_dir.mkdir()

        # Write dummy frame 0
        (img_dir / "frame_000000.png").touch()
        
        dataset = ARKitDataset(self.temp_path)
        self.assertEqual(dataset.image_dir, img_dir)
        self.assertEqual(dataset.depth_dir, depth_dir)
        self.assertEqual(dataset.metadata_dir, meta_dir)
        self.assertEqual(dataset.indices, [0])

    def test_dataset_layout_unified(self) -> None:
        # Create unified setup (only images/ exists)
        img_dir = self.temp_path / "images"
        img_dir.mkdir()

        # Write dummy frame 0
        (img_dir / "frame_000000.png").touch()

        dataset = ARKitDataset(self.temp_path)
        self.assertEqual(dataset.image_dir, img_dir)
        self.assertEqual(dataset.depth_dir, img_dir)
        self.assertEqual(dataset.metadata_dir, img_dir)
        self.assertEqual(dataset.indices, [0])


if __name__ == "__main__":
    unittest.main()
