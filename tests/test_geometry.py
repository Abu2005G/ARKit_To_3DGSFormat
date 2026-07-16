"""
test_geometry.py

Unit tests for checking geometry conversions (project, backproject, camera_to_world, etc.).
"""

import unittest
import numpy as np

from preprocessing.geometry import (
    pixel_to_camera,
    depth_to_camera,
    camera_to_world,
    world_to_camera,
    project,
    backproject,
)


class TestGeometry(unittest.TestCase):
    """
    Verifies that the coordinate system mappings and matrix operations are mathematically correct.
    """

    def setUp(self) -> None:
        # Standard intrinsic calibration matrix
        self.K = np.array([
            [500.0,   0.0, 320.0],
            [  0.0, 500.0, 240.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float32)

        # Standard identity and rigid camera-to-world transformations
        self.T_c2w_identity = np.eye(4, dtype=np.float32)
        
        # Camera translation and rotation (yaw by 90 deg around Y axis)
        c, s = 0.0, 1.0  # cos(90) and sin(90)
        self.T_c2w_rotated = np.array([
            [ c, 0.0,  s, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [-s, 0.0,  c, 3.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def test_pixel_to_camera(self) -> None:
        # Principal point should project along visual axis Z with depth distance
        pt = pixel_to_camera(320.0, 240.0, 2.0, self.K)
        np.testing.assert_allclose(pt, [0.0, 0.0, 2.0], atol=1e-5)

        # Offset pixel
        pt = pixel_to_camera(420.0, 140.0, 1.5, self.K)
        # X = (420 - 320) * 1.5 / 500 = 100 * 1.5 / 500 = 0.3
        # Y = (140 - 240) * 1.5 / 500 = -100 * 1.5 / 500 = -0.3
        np.testing.assert_allclose(pt, [0.3, -0.3, 1.5], atol=1e-5)

    def test_depth_to_camera(self) -> None:
        depth = np.ones((10, 10), dtype=np.float32) * 2.5
        pts_cam = depth_to_camera(depth, self.K)
        
        self.assertEqual(pts_cam.shape, (100, 3))
        # Check Z coordinate is constant
        np.testing.assert_allclose(pts_cam[:, 2], 2.5)

    def test_camera_to_world_identity(self) -> None:
        pts_cam = np.array([
            [1.0, 2.0, 3.0],
            [-1.5, 0.5, 4.0]
        ], dtype=np.float32)

        pts_world = camera_to_world(pts_cam, self.T_c2w_identity)
        np.testing.assert_allclose(pts_world, pts_cam)

    def test_camera_to_world_rotated(self) -> None:
        # Transformed coordinate: P_w = R_c2w @ P_c + t_c2w
        pts_cam = np.array([
            [0.0, 0.0, 1.0],  # along Z camera axis
        ], dtype=np.float32)

        pts_world = camera_to_world(pts_cam, self.T_c2w_rotated)
        # R_c2w @ [0,0,1]^T = [s, 0, c]^T = [1.0, 0.0, 0.0]
        # Then offset by translation [1, 2, 3] = [2.0, 2.0, 3.0]
        np.testing.assert_allclose(pts_world[0], [2.0, 2.0, 3.0], atol=1e-5)

    def test_world_to_camera_roundtrip(self) -> None:
        pts_cam = np.array([
            [0.5, -0.2, 1.8],
            [1.2, 3.0, 4.5]
        ], dtype=np.float32)

        # Forward
        pts_world = camera_to_world(pts_cam, self.T_c2w_rotated)
        # Inverse
        pts_cam_back = world_to_camera(pts_world, self.T_c2w_rotated)
        np.testing.assert_allclose(pts_cam_back, pts_cam, atol=1e-4)

    def test_projection_roundtrip(self) -> None:
        pts_cam = np.array([
            [0.5, -0.2, 2.0],
            [1.5, 1.0, 3.0]
        ], dtype=np.float32)

        pixels, depths = project(pts_cam, self.K)
        
        # Test backprojecting the pixels using the depths
        pt0 = pixel_to_camera(pixels[0, 0], pixels[0, 1], depths[0], self.K)
        pt1 = pixel_to_camera(pixels[1, 0], pixels[1, 1], depths[1], self.K)

        np.testing.assert_allclose(pt0, pts_cam[0], atol=1e-4)
        np.testing.assert_allclose(pt1, pts_cam[1], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
