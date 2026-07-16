"""
utils.py

Provides utility functions and classes for computing frame counts, tracking processing
execution times, checking system memory usage, and verifying GPU hardware acceleration.
"""

from pathlib import Path
from typing import Dict, Any, Union
import time
import os
import resource


def count_frames(image_dir: Union[str, Path]) -> int:
    """
    Returns the number of PNG images in the specified directory.
    """
    return len(list(Path(image_dir).glob("frame_*.png")))


def get_memory_usage_mb() -> float:
    """
    Returns the current resident set size (RSS) memory usage of this process
    in Megabytes (MB). Uses standard library resource routines.
    """
    # ru_maxrss returns kilobytes on Linux
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss_kb / 1024.0


def check_gpu_info() -> Dict[str, Any]:
    """
    Checks for GPU hardware support via PyTorch or Open3D.
    """
    gpu_info = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": "CPU-only",
        "open3d_gpu": False
    }
    
    # Check PyTorch CUDA availability
    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        gpu_info["device_count"] = torch.cuda.device_count()
        if gpu_info["cuda_available"]:
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # Check Open3D compiled GPU support
    try:
        import open3d as o3d
        gpu_info["open3d_gpu"] = o3d._build_config["ENABLE_CUDA"]
    except (ImportError, KeyError, AttributeError):
        pass

    return gpu_info


class Timer:
    """
    Timer class for profiling processing durations of pipeline blocks.
    To be used as a context manager or manually.
    """

    def __init__(self, description: str = "Block") -> None:
        self.description = description
        self.start_time = 0.0
        self.end_time = 0.0
        self.duration = 0.0

    def __enter__(self) -> 'Timer':
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"[Profiling] {self.description} took {self.duration:.4f} seconds.")
