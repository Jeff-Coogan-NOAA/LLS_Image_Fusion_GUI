# COLMAP Export Tool for Gaussian Splatting

This tool converts your underwater vehicle image navigation data into COLMAP format, which is the standard input format for Gaussian Splatting and many 3D reconstruction tools.

## Quick Start

1. **Edit Configuration** in `export_to_colmap.py`:
   ```python
   IMAGE_LIST_CSV = r"C:\path\to\your\image_list.csv"
   OUTPUT_DIR = r"C:\path\to\output\colmap_workspace"
   ```

2. **Run the script**:
   ```bash
   python export_to_colmap.py
   ```

3. **Copy your images** to the output directory:
   ```
   colmap_workspace/
       cameras.bin      ← Camera intrinsics
       images.bin       ← Camera poses
       points3D.bin     ← Empty (for now)
       images/          ← Put your image files here
   ```

## What Gets Created

### cameras.bin
Contains camera **intrinsic parameters** (lens characteristics):
- **Focal length**: 3801.37 pixels
- **Principal point**: Image center (2048, 1504)
- **Distortion coefficients**: k1, k2 (radial), p1, p2 (tangential)
- **Camera model**: OPENCV (model_id = 3)

### images.bin
Contains camera **extrinsic parameters** (position & orientation) for each image:
- **Position**: [East, North, Up] in meters (ENU coordinates)
- **Orientation**: Quaternion [w, x, y, z] representing camera rotation
- **Computed from**:
  - IMU position (AUV_Easting, AUV_Northing, AUV_Depth)
  - IMU orientation (AUV_Pitch, AUV_Roll, AUV_Heading)
  - Lever arm offsets (physical camera-to-IMU offset)
  - Angular calibration offsets

### points3D.bin
Empty file for now. Can be populated by:
- COLMAP feature matching & triangulation
- Importing your existing point cloud data
- Gaussian Splatting initialization

## Understanding the Coordinate Transformations

### 1. IMU to Camera Position
```
IMU Position (from CSV) → Lever Arm Transform → Camera Position (world)
```
The lever arm accounts for the physical offset between the IMU and camera on the vehicle.

### 2. Camera Orientation
```
Vehicle Attitude (pitch/roll/heading) → Rotation Matrix → Quaternion
```
The script converts Euler angles to a rotation matrix, then to a quaternion for COLMAP.

### 3. Coordinate Systems
- **Vehicle Body Frame**: X=forward, Y=right, Z=up
- **World Frame (ENU)**: X=East, Y=North, Z=Up
- **COLMAP Camera**: +Z=optical axis (forward), +X=right, +Y=down

## Calibration Parameters

### Lever Arms (meters)
Physical offset from IMU to camera in vehicle body frame:
```python
LEVER_ARM_X = 0.125813   # Forward
LEVER_ARM_Y = 0.945584   # Right
LEVER_ARM_Z = -0.213513  # Up (negative = camera below IMU)
```

### Angular Offsets (degrees)
Calibration corrections applied to IMU measurements:
```python
PITCH_OFFSET = 0.010     # Pitch correction
ROLL_OFFSET = 0.010      # Roll correction
HEADING_OFFSET = 0.000   # Heading correction
```

### Camera Intrinsics
```python
FOCAL_LENGTH_PX = 3801.37053  # Focal length
IMAGE_WIDTH = 4096
IMAGE_HEIGHT = 3008
K1 = 0.0113579                # Radial distortion
K2 = -0.0143928
P1 = 0.0042688                # Tangential distortion
P2 = -0.000244194
```

## Using with Gaussian Splatting

### Option 1: Direct Import (if supported)
Some Gaussian Splatting implementations can directly read COLMAP binary files.

### Option 2: Convert to Text Format
COLMAP also supports text format (cameras.txt, images.txt, points3D.txt):
```bash
colmap model_converter \
    --input_path colmap_workspace \
    --output_path colmap_workspace \
    --output_type TXT
```

### Option 3: Popular Training Frameworks

**Nerfstudio** (supports COLMAP directly):
```bash
ns-train splatfacto \
    --data colmap_workspace \
    --pipeline.datamanager.camera-optimizer.mode off
```

**Inria Gaussian Splatting** (expects COLMAP structure):
```bash
python train.py \
    --source_path colmap_workspace \
    --model_path output/model
```

**gsplat** (PyTorch implementation):
```bash
python train.py \
    --data_dir colmap_workspace \
    --result_dir output
```

## Troubleshooting

### Issue: "Image list CSV not found"
Update the `IMAGE_LIST_CSV` path at the top of `export_to_colmap.py`.

### Issue: "Missing required columns"
Your CSV must contain these columns:
- file_name
- AUV_Pitch, AUV_Roll, AUV_Heading
- AUV_Easting, AUV_Northing, AUV_Depth

### Issue: Gaussian Splatting training fails
1. Verify images are in `colmap_workspace/images/` directory
2. Check that filenames in CSV match actual image filenames
3. Ensure all camera poses are reasonable (no NaN values)

### Issue: Coordinate system mismatch
If your reconstruction looks inverted or rotated:
- Check lever arm signs (Z should be negative if camera below IMU)
- Verify angular offset signs
- Confirm that AUV_Depth is positive downward

## Advanced: Populating points3D.bin

If you want to include initial 3D points (from your existing point clouds):

```python
def add_point_cloud_to_colmap(las_file, output_path):
    """Add points from LAZ file to points3D.bin"""
    import laspy
    
    las = laspy.read(las_file)
    num_points = len(las.x)
    
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<Q', num_points))
        
        for i in range(num_points):
            point_id = i + 1
            x, y, z = las.x[i], las.y[i], las.z[i]
            r, g, b = las.red[i] // 256, las.green[i] // 256, las.blue[i] // 256
            error = 1.0
            track_length = 0
            
            f.write(struct.pack('<Q', point_id))
            f.write(struct.pack('<ddd', x, y, z))
            f.write(struct.pack('<BBB', r, g, b))
            f.write(struct.pack('<d', error))
            f.write(struct.pack('<Q', track_length))
```

## Comparison: Your Current Workflow vs COLMAP

| Aspect | Current (Custom) | COLMAP Format |
|--------|------------------|---------------|
| **Coordinate System** | ENU (custom) | ENU (standard) |
| **Rotation Format** | Euler angles | Quaternions |
| **File Format** | LAZ/XYZ point clouds | Binary workspace |
| **Compatibility** | Custom tools only | Industry standard |
| **Gaussian Splatting** | Needs adapter | Native support |
| **Feature Matching** | Not available | COLMAP provides |

## Benefits of COLMAP Format

1. **Standard Format**: Works with most 3D reconstruction tools
2. **Gaussian Splatting**: Direct compatibility with training frameworks
3. **Feature Matching**: Can run COLMAP to improve poses (optional)
4. **Visualization**: Use COLMAP GUI to inspect camera poses
5. **Interoperability**: Easy to share data with other researchers

## References

- [COLMAP Documentation](https://colmap.github.io/)
- [COLMAP Binary Format Specification](https://colmap.github.io/format.html)
- [Gaussian Splatting (Original Paper)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Nerfstudio (Training Framework)](https://docs.nerf.studio/)

---

**Questions or Issues?** 
Check the comments in `export_to_colmap.py` for detailed explanations of each function.
