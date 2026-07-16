# ARKit LiDAR 3D Reconstruction Preprocessing Framework

A production-quality, modular Python framework designed to transform raw ARKit RGB-D capture datasets (exported from LiDAR-enabled Apple devices) into various 3D representations. 

This library supports exporting watertight meshes, camera trajectories, TSDF volumes, and **3D Gaussian Splatting (3DGS) COLMAP initializations**.

---

## 🚀 Why this is better than COLMAP for ARKit data

[COLMAP](https://colmap.github.io/) is the gold standard for Structure-from-Motion (SfM), but applying it to raw sequences recorded on an iPad or iPhone is highly sub-optimal. This framework provides distinct advantages:

1. **No Scale Ambiguity (True Metrical Scale)**
   * *COLMAP*: Performs SfM by matching visual features, which lacks absolute scale estimation. Outputs require manual scaling or alignment to fit real-world coordinates.
   * *This Framework*: Directly leverages iPhone LiDAR depth readings and calibrated visual-inertial odometry (VIO) poses which are measured in **true physical meters**.

2. **No Fragile Feature Matching / SfM Failures**
   * *COLMAP*: Often fails to register frames or optimize bundle adjustment on textureless surfaces (reflective screens, blank white walls, flat floors) or repetitive patterns.
   * *This Framework*: Uses hardware IMU + VIO camera-to-world transforms. Camera tracking works reliably regardless of visual texture, avoiding skipped or disconnected frames.

3. **Deterministic & Fast Execution**
   * *COLMAP*: Multi-phase SfM (sift feature extraction, exhaustive matching, triangulation, bundle adjustment) can take minutes or hours for several hundred frames.
   * *This Framework*: Sequential projection, voxel downsampling, and math-based alignment run in **seconds**, saving significant overhead.

4. **Hardware-Aligned Depth Fusions**
   * *COLMAP*: Computes depth maps using Multi-View Stereo (MVS) which is computationally expensive and prone to boundary artifacts.
   * *This Framework*: Performs direct, hardware-supported TSDF (Truncated Signed Distance Function) voxel integration to build clean, watertight surface models.

---

## 🛠️ Packages & Dependencies

The library is designed to run in a conda environment containing the following key packages:
* **`open3d`** (>=0.16.0): Used for volumetric TSDF integration, point cloud manipulation (Statistical Outlier Removal, KD-Tree normal estimations), and mesh reconstruction.
* **`opencv-python`**: Used for rapid image resizing (upsampling/downsampling depth maps & fitting RGB frames) and color map conversions.
* **`numpy`**: Powering vectorized grid modifications, projection transformations, and coordinate adjustments.
* **`pillow` (PIL)**: Standard image reading and handling.
* **`torch`** (Optional): Checked at startup to log CUDA GPU configuration hardware.

---

## 📐 How it Works (Under the Hood)

```
        Raw ZIP Export (PNG Images + Binary .depth32f + Metadata JSONs)
                                      |
                                      v
                             [ZIP Extraction]
                                      |
                                      v
                             [ARKitDataset Wrapper]
                                      |
      -----------------------------------------------------------------
     |                                                                 |
     v                                                                 v
[RGBDBuilder Profile]                                       [Validation Checks]
   - Intrinsics Scale matching                                 - Poses orthogonality
   - OpenGL (+Y up, -Z forward) -> OpenCV (+Y down, +Z fwd)     - Frame dimensions integrity
     Coordinate Flipping & Coordinate Alignments               - Tracking flags status check
     |                                                                 |
     v                                                                 v
[PointCloudBuilder & Fusion]                                [TSDFIntegrator Engine]
   - Backproject depth to world coordinates                    - scalable voxel integration
   - Combined RGB sampling                                     - Marching Cubes mesh extraction
   - Statistical & Radius Outlier removal (SOR/ROR)                     |
     |                                                                  v
     |                                                      [MeshReconstructor Setup]
     v                                                         - Poisson watertight reconstruction
[GaussianInitializer Formatting]                               - Ball Pivoting Algorithm (BPA)
   - Convert poses to scalar-first quaternions                          |
   - Colmap structure (cameras.txt, images.txt,                         v
     points3D.txt/ply) for 3DGS pipeline                     Export Outputs (PLY, OBJ, 3DGS)
```

---

## 💻 Usage Instructions

### 1. Environment Setup
Activate your environment:
```bash
conda activate rgbd
```

### 2. Run Interactive Mode
Launch the pipeline with the walkthrough setup:
```bash
python -m preprocessing.main
```
You will be prompted for:
1. Archive location path (`.zip`)
2. Path for saving result directories
3. Project Name
4. Reconstruction Mode (1-5)
5. Frame Processing slice bounds

### 3. Run Non-Interactive CLI (Command-Line Mode)
Integrate or batch process datasets directly:
```bash
python -m preprocessing.main \
  --zip /home/user/Capture.zip \
  --out /home/user/outputs \
  --name my_study_room \
  --mode "Full Pipeline" \
  --range 3
```

#### Argument Details:
*   `--zip`: Absolute path to source dataset ZIP file.
*   `--out`: Parent output directory.
*   `--name`: Folder subdirectory name for generated files.
*   `--mode`: Selection of `"Point Cloud"`, `"TSDF"`, `"Mesh"`, `"Gaussian"`, or `"Full Pipeline"`.
*   `--range`: Frame parsing bounds: `"1"` (first frame only), `"2"` (subset bounds), or `"3"` (all frames).
*   `--n_frames`: Number of frames to slice (only used if `--range` is `"2"`).

### 4. Running Unit Tests
Validate transformations and decoders:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
