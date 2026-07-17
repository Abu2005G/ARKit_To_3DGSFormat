"""
main.py

Entry point and user interface for the ARKit LiDAR RGB-D Preprocessing Framework.
Prompts the user for execution mode, dataset zip, and outputs, unzips files,
and sequences reconstruction pipelines.
"""

import sys
import shutil
import tempfile
import traceback
from pathlib import Path
import numpy as np

from preprocessing.config import DEFAULT_OUTPUT_DIR, VOXEL_SIZE, BPA_MIN_COMPONENT_TRIANGLES
from preprocessing.io_utils import unzip_dataset
from preprocessing.dataset import ARKitDataset
from preprocessing.validation import (
    print_diagnostics, validate_frame,
    compute_coverage_diagnostic, detect_pose_jumps,
)
from preprocessing.pointcloud import PointCloudBuilder
from preprocessing.refinement import estimate_normals
from preprocessing.fusion import FusionEngine
from preprocessing.tsdf import TSDFIntegrator
from preprocessing.mesh import MeshReconstructor
from preprocessing.gaussian_init import GaussianInitializer
from preprocessing.utils import check_gpu_info, get_memory_usage_mb, Timer, count_frames


def farthest_point_sample_poses(dataset, all_indices, n_target):
    """
    Selects n_target frames from all_indices using farthest-point sampling
    over camera pose positions. This ensures maximal spatial coverage.
    """
    if n_target >= len(all_indices):
        return all_indices

    # Extract all camera centers
    centers = []
    for idx in all_indices:
        frame = dataset[idx]
        T_c2w = frame.camera_rgb.pose
        centers.append(T_c2w[:3, 3].copy())
    centers = np.array(centers)

    # Start with the first frame
    selected = [0]
    min_distances = np.full(len(centers), np.inf)

    for _ in range(n_target - 1):
        last_center = centers[selected[-1]]
        dists = np.linalg.norm(centers - last_center, axis=1)
        min_distances = np.minimum(min_distances, dists)
        # Select the point farthest from all already-selected points
        next_idx = np.argmax(min_distances)
        selected.append(next_idx)

    # Map back to original frame indices
    result = sorted([all_indices[i] for i in selected])
    return result


def run_pipeline(
    zip_path: Path,
    output_root: Path,
    project_name: str,
    reconstruction_mode: str,
    frame_limit_mode: str,
    n_frames: int = -1,
    sampling_mode: str = "sequential",
) -> None:
    """
    Runs the ARKit reconstruction pipeline based on user configurations.
    """
    print("\n" + "=" * 60)
    print(f"Starting ARKit Reconstruction Pipeline: {project_name}")
    print("=" * 60)

    # Telemetry
    gpu_info = check_gpu_info()
    print(f"System State: CUDA={gpu_info['cuda_available']}, GPU Build={gpu_info['open3d_gpu']}, Memory={get_memory_usage_mb():.1f} MB")
    if gpu_info['cuda_available']:
        print(f"GPU Hardware: {gpu_info['device_name']}")

    # Create temporary directory for extraction
    tmp_extract = Path(tempfile.gettempdir()) / f"ark_3dgs_tmp_{project_name}"
    
    with Timer("ZIP Dataset Extraction"):
        dataset_root = unzip_dataset(zip_path, tmp_extract)

    print(f"Dataset root identified at: {dataset_root}")
    
    # Load dataset
    dataset = ARKitDataset(dataset_root)
    total_frames = len(dataset)
    print(f"Found {total_frames} total frames in ARKit dataset.")

    # Determine frame slice
    if frame_limit_mode == "1":
        indices = [0]
        print("Selecting: Frame 0 only.")
    elif frame_limit_mode == "2":
        limit = n_frames if n_frames > 0 else 5
        all_indices = list(range(total_frames))
        if sampling_mode == "farthest_point" and limit < total_frames:
            indices = farthest_point_sample_poses(dataset, all_indices, limit)
            print(f"Selecting: {len(indices)} frames (farthest-point sampling).")
        else:
            indices = list(range(min(limit, total_frames)))
            print(f"Selecting: First {len(indices)} frames.")
    else:
        indices = list(range(total_frames))
        print(f"Selecting: Entire dataset ({len(indices)} frames).")

    # Destination directory
    project_output = output_root / project_name
    project_output.mkdir(parents=True, exist_ok=True)
    print(f"Results will be written to: {project_output}")

    # Validate first frame diagnostics
    print("\n[Validation] Auditing Frame 0 calibration assets...")
    print_diagnostics(dataset[0])

    # Issue 2: Detect pose trajectory jumps
    if len(indices) >= 2:
        detect_pose_jumps(dataset, indices)

    # Instantiate point cloud builder & fusion engine
    pc_builder = PointCloudBuilder()
    fusion_engine = FusionEngine(dataset, pc_builder)

    # Objects compiled
    fused_pcd = None
    tsdf_pcd = None
    tsdf_mesh = None
    recon_mesh = None
    bpa_mesh = None

    # Summary stats collectors
    summary = {
        "point_count": 0,
        "camera_count": len(indices),
        "tsdf_mesh_components": None,
        "poisson_mesh_components": None,
        "bpa_mesh_components": None,
        "coverage": None,
    }

    reconstruction_mode_lower = reconstruction_mode.lower().strip()

    # Apply Pipeline Steps
    try:
        # Step A: Point Cloud (Requires compilation in Gaussian and Mesh mode too)
        if reconstruction_mode_lower in ["point cloud", "mesh", "gaussian", "full pipeline"]:
            with Timer("Frame Fusion & Outlier Removal"):
                fused_pcd = fusion_engine.fuse(indices=indices, voxel_size=VOXEL_SIZE, run_sor=True)
            
            pcd_ply_out = project_output / "fused_point_cloud.ply"
            pcd_pcd_out = project_output / "fused_point_cloud.pcd"

            if len(fused_pcd.points) > 0:
                # Save PLY and PCD
                from preprocessing.io_utils import save_pointcloud
                save_pointcloud(fused_pcd, pcd_ply_out)
                save_pointcloud(fused_pcd, pcd_pcd_out)
                
                # Print stats
                bbox = fused_pcd.get_axis_aligned_bounding_box()
                print(f"Fused Bounding Box Min: {bbox.get_min_bound()}")
                print(f"Fused Bounding Box Max: {bbox.get_max_bound()}")
                summary["point_count"] = len(fused_pcd.points)
            else:
                print("Warning: Fused point cloud is empty!")

        # Step B: TSDF Fusion
        if reconstruction_mode_lower in ["tsdf", "full pipeline"]:
            integrator = TSDFIntegrator()
            with Timer("TSDF Volumetric Integration"):
                integrator.integrate_dataset(dataset, indices=indices)
            
            # Extract point cloud from TSDF
            tsdf_pcd = integrator.extract_point_cloud()
            if len(tsdf_pcd.points) > 0:
                tsdf_pcd_out = project_output / "tsdf_point_cloud.ply"
                from preprocessing.io_utils import save_pointcloud
                save_pointcloud(tsdf_pcd, tsdf_pcd_out)
            
            # Extract Marching Cubes mesh
            tsdf_mesh = integrator.extract_mesh()
            if len(tsdf_mesh.triangles) > 0:
                with Timer("Cleaning TSDF Mesh"):
                    tsdf_mesh = MeshReconstructor.clean_mesh(tsdf_mesh)
                    tsdf_mesh = MeshReconstructor.remove_small_connected_components(tsdf_mesh, min_triangles=BPA_MIN_COMPONENT_TRIANGLES)
                tsdf_mesh_ply = project_output / "tsdf_mesh_marching_cubes.ply"
                tsdf_mesh_obj = project_output / "tsdf_mesh_marching_cubes.obj"
                from preprocessing.io_utils import save_mesh
                save_mesh(tsdf_mesh, tsdf_mesh_ply)
                save_mesh(tsdf_mesh, tsdf_mesh_obj)
                summary["tsdf_mesh_components"] = MeshReconstructor.component_count(tsdf_mesh)
                print(f"TSDF Marching Cubes mesh saved with {len(tsdf_mesh.triangles)} face parameters.")

        # Step C: Mesh Reconstruction
        if reconstruction_mode_lower in ["mesh", "full pipeline"]:
            if fused_pcd is not None and len(fused_pcd.points) > 0:
                # Compute normals
                with Timer("Normal Estimation"):
                    estimate_normals(fused_pcd)

                # Reconstruct Mesh (Poisson Surface Reconstruction)
                with Timer("Poisson Mesh Reconstruction"):
                    poisson_mesh, densities = MeshReconstructor.reconstruct_poisson(fused_pcd)
                    # Filter boundary faces
                    filtered_mesh = MeshReconstructor.filter_low_density_vertices(poisson_mesh, densities)
                    # Crop post-filtering to avoid boundary mismatch errors
                    bbox = fused_pcd.get_axis_aligned_bounding_box()
                    cropped_mesh = filtered_mesh.crop(bbox)
                    # Clean and remove spurious noise blobs
                    cleaned_mesh = MeshReconstructor.clean_mesh(cropped_mesh)
                    recon_mesh = MeshReconstructor.remove_small_connected_components(cleaned_mesh, min_triangles=BPA_MIN_COMPONENT_TRIANGLES)

                mesh_ply = project_output / "reconstructed_mesh_poisson.ply"
                mesh_obj = project_output / "reconstructed_mesh_poisson.obj"
                from preprocessing.io_utils import save_mesh
                save_mesh(recon_mesh, mesh_ply)
                save_mesh(recon_mesh, mesh_obj)
                summary["poisson_mesh_components"] = MeshReconstructor.component_count(recon_mesh)
                print(f"Poisson mesh reconstruction saved with {len(recon_mesh.triangles)} triangles.")

                # BPA Reconstruction option (adaptive radii)
                with Timer("Ball Pivoting Mesh Reconstruction"):
                    raw_bpa_mesh = MeshReconstructor.reconstruct_ball_pivoting(fused_pcd)
                    cleaned_bpa_mesh = MeshReconstructor.clean_mesh(raw_bpa_mesh)
                    bpa_mesh = MeshReconstructor.remove_small_connected_components(cleaned_bpa_mesh, min_triangles=BPA_MIN_COMPONENT_TRIANGLES)
                mesh_bpa_ply = project_output / "reconstructed_mesh_bpa.ply"
                save_mesh(bpa_mesh, mesh_bpa_ply)
                summary["bpa_mesh_components"] = MeshReconstructor.component_count(bpa_mesh)
                print(f"Ball Pivoting mesh saved with {len(bpa_mesh.triangles)} triangles.")

        # Step D: Gaussian Splatting compatible export
        if reconstruction_mode_lower in ["gaussian", "full pipeline"]:
            from preprocessing.config import USE_TSDF_PCD_FOR_3DGS
            pcd_to_export = fused_pcd
            if USE_TSDF_PCD_FOR_3DGS and tsdf_pcd is not None and len(tsdf_pcd.points) > 0:
                pcd_to_export = tsdf_pcd
                print("Using TSDF Point Cloud as initialization for 3DGS.")

            if pcd_to_export is not None and len(pcd_to_export.points) > 0:
                # Apply Multi-View Depth Consistency Filter (SR-LIVO style extreme denoising)
                with Timer("Multi-View Depth Consistency Filtering"):
                    from preprocessing.multiview_filter import MultiViewDepthFilter
                    mv_filter = MultiViewDepthFilter(dataset)
                    pcd_to_export = mv_filter.filter_point_cloud(pcd_to_export, indices)

                with Timer("3DGS COLMAP Export Formatting"):
                    gs_init = GaussianInitializer(dataset, project_output)
                    gs_init.export_dataset(pcd_to_export, indices)

                # Issue 1: Coverage diagnostic after GS export
                if fused_pcd is not None and len(fused_pcd.points) > 0:
                    camera_centers = []
                    for idx in indices:
                        frame = dataset[idx]
                        T_c2w = frame.camera_rgb.pose
                        Y = np.diag([1.0, -1.0, -1.0, 1.0])
                        T_c2w_cv = T_c2w @ Y
                        camera_centers.append(T_c2w_cv[:3, 3].copy())
                    camera_centers = np.array(camera_centers)
                    pcd_bbox = fused_pcd.get_axis_aligned_bounding_box()
                    coverage_result = compute_coverage_diagnostic(
                        camera_centers,
                        pcd_bbox.get_min_bound(),
                        pcd_bbox.get_max_bound(),
                    )
                    summary["coverage"] = coverage_result
            else:
                print("Cannot build Gaussian Initialization: Target point cloud is empty.")

        # ============================================================
        # Pipeline Summary Table
        # ============================================================
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"  Point Cloud:       {summary['point_count']:,} points")
        print(f"  Camera Count:      {summary['camera_count']}")

        if summary["coverage"] is not None:
            cov = summary["coverage"]["coverage_ratios"]
            print(f"  Camera Coverage:   X={cov[0]*100:.0f}%  Y={cov[1]*100:.0f}%  Z={cov[2]*100:.0f}%")
            print(f"  Centroid Offset:   {summary['coverage']['offset_ratio']*100:.1f}% of scene diagonal")

        for label, key in [("TSDF Mesh", "tsdf_mesh_components"),
                           ("Poisson Mesh", "poisson_mesh_components"),
                           ("BPA Mesh", "bpa_mesh_components")]:
            cc = summary[key]
            if cc is not None:
                print(f"  {label}:  {cc['total_tris']:,} tris, {cc['total_components']} components (largest: {cc['largest_component_tris']:,})")
        print("=" * 60)

        print(f"\nPipeline execution completed successfully.")
        print(f"Final RAM Peak: {get_memory_usage_mb():.1f} MB")
        
    except Exception as e:
        print(f"\n[Error] Pipeline failed: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        raise e
    finally:
        # Clean up temporary unzipped folder to avoid disk leakage
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract)


def main() -> None:
    """
    Main interface loop for gathering inputs and launching pipeline.
    """
    import argparse
    
    # Check if arguments are supplied via command line
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="ARKit Reconstruction Pipeline CLI")
        parser.add_argument("--zip", type=str, required=True, help="Path to input ARKit ZIP dataset")
        parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
        parser.add_argument("--name", type=str, required=True, help="Project name")
        parser.add_argument("--mode", type=str, default="Full Pipeline", 
                            choices=["Point Cloud", "TSDF", "Mesh", "Gaussian", "Full Pipeline"],
                            help="Reconstruction Mode")
        parser.add_argument("--range", type=str, default="3", choices=["1", "2", "3"],
                            help="Frame Processing Range: 1 (debug), 2 (subset), 3 (all)")
        parser.add_argument("--n_frames", type=int, default=-1, help="N frames (if range is 2)")
        parser.add_argument("--sampling", type=str, default="sequential",
                            choices=["sequential", "farthest_point"],
                            help="Frame sampling strategy: sequential (default) or farthest_point")

        args = parser.parse_args()
        zip_path = Path(args.zip).expanduser()
        output_dir = Path(args.out).expanduser()
        project_name = args.name
        reconstruction_mode = args.mode
        frame_limit_mode = args.range
        n_frames = args.n_frames
        sampling_mode = args.sampling
        
        if not zip_path.is_file():
            print(f"Error: ZIP file not found: {zip_path}")
            sys.exit(1)

        run_pipeline(
            zip_path=zip_path,
            output_root=output_dir,
            project_name=project_name,
            reconstruction_mode=reconstruction_mode,
            frame_limit_mode=frame_limit_mode,
            n_frames=n_frames,
            sampling_mode=sampling_mode,
        )
        return

    print("\n============================================================")
    print("      ARKit LiDAR RGB-D Preprocessing Framework Launcher      ")
    print("============================================================\n")

    # 1. Ask for input ZIP dataset
    while True:
        zip_str = input("Enter Path to input ARKit ZIP dataset: ").strip()
        if not zip_str:
            print("Path cannot be empty.")
            continue
        zip_path = Path(zip_str).expanduser()
        if not zip_path.is_file():
            print(f"File not found: {zip_path}. Please re-enter.")
            continue
        break

    # 2. Ask for Output Directory
    output_str = input(f"Enter Output directory (default '{DEFAULT_OUTPUT_DIR}'): ").strip()
    if output_str:
        output_dir = Path(output_str).expanduser()
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    # 3. Ask for Project Name
    while True:
        project_name = input("Enter Project name: ").strip()
        if not project_name:
            print("Project name cannot be empty.")
            continue
        break

    # 4. Reconstruction Mode Selection
    print("\nSelect Reconstruction Mode:")
    print("  1. Point Cloud (Mesh vertices, colors, normals, and outlier filtering)")
    print("  2. TSDF (Scalable TSDF fusion & Marching Cubes mesh)")
    print("  3. Mesh (Poisson Surface Reconstruction + BPA)")
    print("  4. Gaussian (GraphDECO 3DGS-compatible Colmap data initialization)")
    print("  5. Full Pipeline (Runs all options above)")
    
    modes_map = {
        "1": "Point Cloud",
        "2": "TSDF",
        "3": "Mesh",
        "4": "Gaussian",
        "5": "Full Pipeline"
    }

    while True:
        mode_choice = input("Enter choice (1-5): ").strip()
        if mode_choice in modes_map:
            reconstruction_mode = modes_map[mode_choice]
            break
        print("Invalid choice, enter 1, 2, 3, 4 or 5.")

    # 5. Debug Frame limits
    print("\nSelect Frame Processing Range:")
    print("  1. First frame only (debug)")
    print("  2. First N frames (subset)")
    print("  3. Entire dataset")
    
    while True:
        range_choice = input("Enter choice (1-3): ").strip()
        if range_choice in ["1", "2", "3"]:
            frame_limit_mode = range_choice
            break
        print("Invalid choice, enter 1, 2 or 3.")

    n_frames = -1
    sampling_mode = "sequential"

    if frame_limit_mode == "2":
        while True:
            val = input("Enter value for N frames: ").strip()
            try:
                n_frames = int(val)
                if n_frames > 0:
                    break
                print("Must be positive.")
            except ValueError:
                print("Must be an integer.")

        samp = input("Sampling strategy? (1=sequential, 2=farthest_point) [default: 1]: ").strip()
        if samp == "2":
            sampling_mode = "farthest_point"

    # Run the pipeline
    try:
        run_pipeline(
            zip_path=zip_path,
            output_root=output_dir,
            project_name=project_name,
            reconstruction_mode=reconstruction_mode,
            frame_limit_mode=frame_limit_mode,
            n_frames=n_frames,
            sampling_mode=sampling_mode,
        )
    except Exception as e:
        print(f"\n[Execution Failure] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
