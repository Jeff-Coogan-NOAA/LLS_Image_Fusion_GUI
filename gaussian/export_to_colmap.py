"""
Export Image Navigation Data to COLMAP Format

This script converts underwater vehicle image navigation data (from CSV) into
COLMAP workspace format for use with Gaussian Splatting and other 3D reconstruction tools.

COLMAP Format Overview:
-----------------------
COLMAP uses three binary files to describe camera parameters and poses:
  - cameras.bin:   Camera intrinsic parameters (focal length, distortion, etc.)
  - images.bin:    Camera extrinsic parameters (position, rotation) for each image
  - points3D.bin:  3D point cloud (can be empty for initial reconstruction)

Coordinate Systems:
-------------------
  Vehicle Body Frame:  X=forward, Y=right, Z=up
  ENU (World) Frame:   X=East, Y=North, Z=Up
  COLMAP Convention:   Camera looks down +Z axis, +X right, +Y down
"""

import os
import struct
import numpy as np
import pandas as pd
from typing import Tuple



# ============================================================================
# CONFIGURATION - Edit these paths and parameters
# ============================================================================

# Hard-coded paths (modify these for your data)
IMAGE_LIST_CSV = r"I:\Image_LLS_PRC\DIVE012_SN402\processing\image\image_file_list.csv"
OUTPUT_DIR = r"I:\Image_LLS_PRC\DIVE012_SN402\processing\image\colmap_workspace"

# Camera intrinsic parameters (from your existing calibration)
FOCAL_LENGTH_PX = 3801.37053  # Focal length in pixels
IMAGE_WIDTH = 4096            # Image width in pixels
IMAGE_HEIGHT = 3008           # Image height in pixels

# Distortion coefficients (radial: k1, k2; tangential: p1, p2)
K1 = 0.0113579
K2 = -0.0143928
P1 = 0.0042688
P2 = -0.000244194

# Lever arm offsets from IMU to camera in vehicle body frame (meters)
# These define where the camera is relative to the IMU/navigation reference
LEVER_ARM_X = 0.125813   # Forward offset
LEVER_ARM_Y = 0.945584   # Right offset
LEVER_ARM_Z = -0.213513  # Up offset

# Angular offsets (degrees) - calibration corrections
PITCH_OFFSET = 0.010
ROLL_OFFSET = 0.010
HEADING_OFFSET = 0.000


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def imu_to_camera_enu(
    imu_east: float,
    imu_north: float,
    imu_up: float,
    lever_arm_x: float,
    lever_arm_y: float,
    lever_arm_z: float,
    pitch: float,
    roll: float,
    heading: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Calculate camera ENU position from IMU ENU position using lever arms.
    
    The lever arm represents the offset from IMU to camera in the vehicle's body frame:
    - X: Forward (positive = camera is forward of IMU)
    - Y: Right (positive = camera is right of IMU) 
    - Z: Up (positive = camera is above IMU)
    
    Parameters:
    -----------
    imu_east, imu_north, imu_up : float
        IMU position in ENU coordinates (meters)
    lever_arm_x, lever_arm_y, lever_arm_z : float
        Lever arm offset from IMU to camera in vehicle body frame (meters)
        X: forward, Y: right, Z: up
    pitch : float
        IMU pitch angle in degrees (positive = nose up)
    roll : float
        IMU roll angle in degrees (positive = right wing down)
    heading : float
        IMU heading angle in degrees (0 = North, 90 = East)
        
    Returns:
    --------
    Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ((camera_east, camera_north, camera_up), (shift_east, shift_north, shift_up))
        Camera ENU position and the shift applied in each direction
    """
    
    # Convert angles from degrees to radians
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    heading_rad = np.radians(heading)
    
    # Lever arm vector in vehicle body frame
    # Body frame: X=forward, Y=right, Z=up
    lever_arm_body = np.array([lever_arm_x, lever_arm_y, lever_arm_z])
    
    # Create rotation matrices to transform from body frame to ENU frame
    # Roll rotation (around Y-axis in camera frame - left/right bank)
    R_roll = np.array([
        [np.cos(roll_rad), 0, np.sin(roll_rad)],
        [0, 1, 0],
        [-np.sin(roll_rad), 0, np.cos(roll_rad)]
    ])
    
    # Pitch rotation (around X-axis in camera frame - nose up/down)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    # Heading rotation (around body Z-axis)
    R_heading = np.array([
        [np.cos(heading_rad), np.sin(heading_rad), 0],
        [-np.sin(heading_rad), np.cos(heading_rad), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: Body to ENU
    # Both ENU and body frame now use Z=up, so no coordinate flip needed
    
    # Apply rotations in order: roll, pitch, heading
    R_body_to_enu = R_heading @ R_roll @ R_pitch
    
    # Transform lever arm from body frame to ENU frame
    lever_arm_enu = R_body_to_enu @ lever_arm_body
    
    # Calculate camera position in ENU coordinates
    camera_east = imu_east + lever_arm_enu[0]
    camera_north = imu_north + lever_arm_enu[1] 
    camera_up = imu_up + lever_arm_enu[2]
    
    # The shift applied is the lever arm transformed to ENU
    shift_east = lever_arm_enu[0]
    shift_north = lever_arm_enu[1]
    shift_up = lever_arm_enu[2]
    
    return (
        (camera_east, camera_north, camera_up),
        (shift_east, shift_north, shift_up)
    )

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].
    
    Uses the Shepperd's method for numerical stability.
    
    Parameters:
    -----------
    R : np.ndarray (3, 3)
        Rotation matrix
        
    Returns:
    --------
    np.ndarray (4,)
        Quaternion in [w, x, y, z] format (COLMAP convention)
    """
    # Trace of rotation matrix
    trace = np.trace(R)
    
    if trace > 0:
        # w is the largest component
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        # x is the largest component
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        # y is the largest component
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        # z is the largest component
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    # Normalize quaternion
    quat = np.array([w, x, y, z])
    quat = quat / np.linalg.norm(quat)
    
    return quat


def build_camera_rotation_matrix(pitch: float, roll: float, heading: float) -> np.ndarray:
    """
    Build rotation matrix from camera orientation angles.
    
    This creates the rotation matrix that transforms from world (ENU) coordinates
    to camera coordinates. The resulting matrix describes camera orientation.
    
    Parameters:
    -----------
    pitch : float
        Pitch angle in degrees (positive = nose up)
    roll : float
        Roll angle in degrees (positive = right wing down)
    heading : float
        Heading angle in degrees (0 = North, 90 = East)
        
    Returns:
    --------
    np.ndarray (3, 3)
        Rotation matrix from world to camera frame
    """
    # Convert angles to radians
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    heading_rad = np.radians(heading)
    
    # Adjust pitch for downward-looking camera
    # Camera optical axis points downward when vehicle is level
    pitch_rad = np.radians(-90) + pitch_rad
    
    # Individual rotation matrices (in camera reference frame)
    # Camera frame: X=right, Y=forward (optical axis), Z=up
    
    # Pitch rotation (around X-axis: nose up/down)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    # Roll rotation (around Y-axis: left/right bank)
    R_roll = np.array([
        [np.cos(roll_rad), 0, np.sin(roll_rad)],
        [0, 1, 0],
        [-np.sin(roll_rad), 0, np.cos(roll_rad)]
    ])
    
    # Heading rotation (around Z-axis: left/right turn)
    R_heading = np.array([
        [np.cos(heading_rad), np.sin(heading_rad), 0],
        [-np.sin(heading_rad), np.cos(heading_rad), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: camera to world
    R_camera_to_world = R_heading @ R_roll @ R_pitch
    
    # COLMAP expects world to camera rotation, so we need the inverse (transpose)
    R_world_to_camera = R_camera_to_world.T
    
    return R_world_to_camera


def compute_camera_pose(
    image_row: pd.Series,
    lever_arm_x: float,
    lever_arm_y: float,
    lever_arm_z: float,
    pitch_offset: float,
    roll_offset: float,
    heading_offset: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute camera position and orientation from navigation data.
    
    Parameters:
    -----------
    image_row : pd.Series
        Row from image dataframe with navigation data
    lever_arm_x, lever_arm_y, lever_arm_z : float
        Lever arm offsets (meters) in vehicle body frame
    pitch_offset, roll_offset, heading_offset : float
        Angular correction offsets (degrees)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (position, quaternion) where:
        - position: [x, y, z] camera position in ENU (meters)
        - quaternion: [w, x, y, z] camera orientation
    """
    # Extract navigation data from CSV row
    pitch = -image_row['pitch']  # Negated based on your convention
    roll = image_row['roll']
    heading = image_row['heading']
    imu_east = image_row['easting']
    imu_north = image_row['northing']
    imu_up = image_row['depth']  # Note: Depth might need negation depending on convention
    
    # Calculate camera position from IMU position using lever arms
    # This accounts for the physical offset between IMU and camera
    camera_pos, _ = imu_to_camera_enu(
        imu_east, imu_north, imu_up,
        lever_arm_x, lever_arm_y, lever_arm_z,
        pitch, roll, heading
    )
    
    # Apply angular offsets (calibration corrections)
    pitch_corrected = pitch + pitch_offset
    roll_corrected = roll + roll_offset
    heading_corrected = heading + heading_offset
    
    # Build rotation matrix from world to camera
    R_world_to_camera = build_camera_rotation_matrix(
        pitch_corrected, roll_corrected, heading_corrected
    )
    
    # Convert rotation matrix to quaternion
    quat = rotation_matrix_to_quaternion(R_world_to_camera)
    
    # Camera position as numpy array [East, North, Up]
    position = np.array([camera_pos[0], camera_pos[1], camera_pos[2]])
    
    return position, quat


# ============================================================================
# COLMAP BINARY FILE WRITERS
# ============================================================================

def write_cameras_bin(output_path: str):
    """
    Write COLMAP cameras.bin file with camera intrinsic parameters.
    
    COLMAP Camera Models:
    ---------------------
    We use model_id = 3 (OPENCV) which supports:
      - fx, fy: focal lengths in pixels
      - cx, cy: principal point coordinates
      - k1, k2: radial distortion coefficients
      - p1, p2: tangential distortion coefficients
    
    Binary Format (per camera):
    ---------------------------
      camera_id (uint64)
      model_id (int32)
      width (uint64)
      height (uint64)
      params (double[]) - variable length based on model
    
    Parameters:
    -----------
    output_path : str
        Path to write cameras.bin file
    """
    # Camera ID (we use 1 for single camera system)
    camera_id = 1
    
    # Camera model: 3 = OPENCV model with 8 parameters
    # (fx, fy, cx, cy, k1, k2, p1, p2)
    model_id = 3
    
    # Image dimensions
    width = IMAGE_WIDTH
    height = IMAGE_HEIGHT
    
    # Camera intrinsic parameters
    # Principal point at image center
    cx = width / 2.0
    cy = height / 2.0
    fx = fy = FOCAL_LENGTH_PX
    
    # Camera parameters array (8 values for OPENCV model)
    params = [fx, fy, cx, cy, K1, K2, P1, P2]
    
    # Write binary file
    with open(output_path, 'wb') as f:
        # Number of cameras (we have 1)
        f.write(struct.pack('<Q', 1))  # uint64: 1 camera
        
        # Camera entry
        f.write(struct.pack('<Q', camera_id))        # uint64: camera ID
        f.write(struct.pack('<i', model_id))         # int32: model ID
        f.write(struct.pack('<Q', width))            # uint64: width
        f.write(struct.pack('<Q', height))           # uint64: height
        
        # Write 8 parameter values (all as double)
        for param in params:
            f.write(struct.pack('<d', param))
    
    print(f"  Wrote cameras.bin: 1 camera (OPENCV model)")
    print(f"    Focal length: {FOCAL_LENGTH_PX:.2f} px")
    print(f"    Image size: {width} x {height}")
    print(f"    Distortion: k1={K1:.6f}, k2={K2:.6f}, p1={P1:.6f}, p2={P2:.6f}")


def write_images_bin(output_path: str, df_images: pd.DataFrame, poses: list):
    """
    Write COLMAP images.bin file with camera extrinsic parameters.
    
    Binary Format (per image):
    --------------------------
      image_id (uint64)
      qw, qx, qy, qz (double[4]) - rotation quaternion
      tx, ty, tz (double[3]) - camera position
      camera_id (uint64)
      name (char[]) - null-terminated filename string
      num_points2D (uint64)
      [x, y, point3D_id (double, double, uint64)] repeated num_points2D times
    
    For initial export, we set num_points2D = 0 (no 2D-3D correspondences yet).
    
    Parameters:
    -----------
    output_path : str
        Path to write images.bin file
    df_images : pd.DataFrame
        Dataframe with image information
    poses : list
        List of (position, quaternion) tuples for each image
    """
    num_images = len(df_images)
    camera_id = 1  # All images use the same camera
    
    with open(output_path, 'wb') as f:
        # Number of images
        f.write(struct.pack('<Q', num_images))
        
        for idx, ((_, img_row), (position, quat)) in enumerate(zip(df_images.iterrows(), poses), 1):
            image_id = idx  # Sequential image IDs starting from 1
            
            # Image filename
            filename = img_row['filename']
            
            # Write image entry
            f.write(struct.pack('<Q', image_id))        # uint64: image ID
            
            # Quaternion (w, x, y, z)
            f.write(struct.pack('<d', quat[0]))         # qw
            f.write(struct.pack('<d', quat[1]))         # qx
            f.write(struct.pack('<d', quat[2]))         # qy
            f.write(struct.pack('<d', quat[3]))         # qz
            
            # Translation (camera position in world frame)
            f.write(struct.pack('<d', position[0]))     # tx (East)
            f.write(struct.pack('<d', position[1]))     # ty (North)
            f.write(struct.pack('<d', position[2]))     # tz (Up)
            
            # Camera ID (same for all images)
            f.write(struct.pack('<Q', camera_id))
            
            # Image filename (null-terminated string)
            f.write(filename.encode('utf-8') + b'\x00')
            
            # Number of 2D points (0 for initial export)
            f.write(struct.pack('<Q', 0))
            
            # No 2D point data to write
    
    print(f"  Wrote images.bin: {num_images} images")


def write_points3D_bin(output_path: str):
    """
    Write COLMAP points3D.bin file.
    
    For initial export, we create an empty point cloud. This can be populated
    later by COLMAP during feature matching and triangulation, or by importing
    your existing point cloud data.
    
    Binary Format (per point):
    --------------------------
      point3D_id (uint64)
      x, y, z (double[3]) - 3D position
      r, g, b (uint8[3]) - color
      error (double) - reconstruction error
      track_length (uint64) - number of observations
      [image_id, point2D_idx (uint64, uint64)] repeated track_length times
    
    Parameters:
    -----------
    output_path : str
        Path to write points3D.bin file
    """
    with open(output_path, 'wb') as f:
        # Number of 3D points (0 for empty point cloud)
        f.write(struct.pack('<Q', 0))
    
    print(f"  Wrote points3D.bin: empty (0 points)")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """
    Main processing function:
    1. Read image navigation data from CSV
    2. Compute camera poses (position + orientation)
    3. Export to COLMAP binary format
    """
    print("=" * 70)
    print("COLMAP Export Tool for Underwater Image Navigation Data")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Validate inputs and create output directory
    # -------------------------------------------------------------------------
    print(f"Input CSV: {IMAGE_LIST_CSV}")
    
    if not os.path.exists(IMAGE_LIST_CSV):
        print(f"ERROR: Image list CSV not found: {IMAGE_LIST_CSV}")
        print("Please update the IMAGE_LIST_CSV path in the script.")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Read image navigation data
    # -------------------------------------------------------------------------
    print("Reading image navigation data...")
    df_images = pd.read_csv(IMAGE_LIST_CSV)
    num_images = len(df_images)
    print(f"  Found {num_images} images in CSV")
    
    # Validate required columns
    required_cols = [
        'filename', 'pitch', 'roll', 'heading',
        'easting', 'northing', 'depth'
    ]
    missing_cols = [col for col in required_cols if col not in df_images.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Compute camera poses for all images
    # -------------------------------------------------------------------------
    print("Computing camera poses...")
    poses = []
    
    for idx, (_, img_row) in enumerate(df_images.iterrows(), 1):
        # Compute position and orientation for this image
        position, quat = compute_camera_pose(
            img_row,
            LEVER_ARM_X, LEVER_ARM_Y, LEVER_ARM_Z,
            PITCH_OFFSET, ROLL_OFFSET, HEADING_OFFSET
        )
        poses.append((position, quat))
        
        # Progress indicator
        if idx % 100 == 0 or idx == num_images:
            print(f"  Processed {idx}/{num_images} images...", end='\r')
    
    print(f"  Processed {num_images}/{num_images} images - COMPLETE")
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Write COLMAP binary files
    # -------------------------------------------------------------------------
    print("Writing COLMAP binary files...")
    
    cameras_path = os.path.join(OUTPUT_DIR, 'cameras.bin')
    images_path = os.path.join(OUTPUT_DIR, 'images.bin')
    points3D_path = os.path.join(OUTPUT_DIR, 'points3D.bin')
    
    write_cameras_bin(cameras_path)
    write_images_bin(images_path, df_images, poses)
    write_points3D_bin(points3D_path)
    
    print()
    print("=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print()
    print("COLMAP workspace created at:")
    print(f"  {OUTPUT_DIR}")
    print()
    print("Files created:")
    print(f"  - cameras.bin   (camera intrinsics)")
    print(f"  - images.bin    (camera poses for {num_images} images)")
    print(f"  - points3D.bin  (empty - to be populated by reconstruction)")
    print()
    print("Next steps:")
    print("  1. Copy your images to: " + os.path.join(OUTPUT_DIR, 'images/'))
    print("  2. Use COLMAP for feature extraction and matching (optional)")
    print("  3. Import this workspace into Gaussian Splatting trainer")
    print()


if __name__ == "__main__":
    main()
