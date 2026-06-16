"""
Utility functions for coordinate transformations and camera projections.
"""
import numpy as np
import cv2
from typing import Tuple


def pixels_to_world_coordinates(
    distance_off_bottom: float,
    pitch: float,
    roll: float, 
    heading: float,
    image_path: str,
    camera_east: float = 0.0,
    camera_north: float = 0.0,
    focal_length_px: float = 3801.37053,
    k1: float = 0.0113579,
    k2: float = -0.0143928,
    p1: float = 0.0042688,
    p2: float = -0.000244194,
    image_width: int = 4096,
    image_height: int = 3008,
    downsample: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert image pixels to world coordinates in meters assuming a flat 2D plane.
    
    Parameters:
    -----------
    distance_off_bottom : float
        Distance from camera to seafloor in meters
    pitch : float
        Camera pitch angle in degrees (rotation about X-axis: positive = nose up)
    roll : float  
        Camera roll angle in degrees (rotation about Y-axis: positive = right bank)
    heading : float
        Camera heading angle in degrees (rotation about Z-axis: 0 = North, 90 = East)
    image_path : str
        Path to the image file (for validation/reference)
    camera_east, camera_north : float
        Camera position in ENU coordinates (meters)
    focal_length_px : float
        Focal length in pixels (default: 3801.37053)
    k1, k2, p1, p2 : float
        Radial and tangential distortion coefficients
    image_width, image_height : int
        Image dimensions in pixels
    downsample : int
        Downsampling factor (1=no downsampling, 2=half size, 10=1/10 size, etc.)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        x_world, y_world arrays in meters for each pixel in absolute ENU coordinates
    """
    
    # make correction to pitch for downward looking camera
    pitch = -90 + pitch
    
    # Apply downsampling to image dimensions and camera parameters
    image_width_ds = image_width // downsample
    image_height_ds = image_height // downsample
    focal_length_ds = focal_length_px / downsample
    
    # Convert angles from degrees to radians
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll) 
    heading_rad = np.radians(heading)
    
    # Camera intrinsic matrix (using downsampled parameters)
    cx = image_width_ds / 2.0   # Principal point x
    cy = image_height_ds / 2.0  # Principal point y
    fx = fy = focal_length_ds  # Focal length in pixels
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy], 
        [0, 0, 1]
    ])
    
    # Distortion coefficients
    dist_coeffs = np.array([k1, k2, p1, p2, 0])
    
    # Create pixel coordinate grids (using downsampled dimensions)
    u_coords, v_coords = np.meshgrid(
        np.arange(image_width_ds, dtype=np.float32),
        np.arange(image_height_ds, dtype=np.float32)
    )
    
    # Stack coordinates for undistortion
    pixel_coords = np.stack([u_coords.ravel(), v_coords.ravel()], axis=1)
    pixel_coords = pixel_coords.reshape(-1, 1, 2)
    
    # Undistort pixel coordinates
    undistorted_coords = cv2.undistortPoints(
        pixel_coords, camera_matrix, dist_coeffs, P=camera_matrix
    )
    undistorted_coords = undistorted_coords.reshape(-1, 2)
    
    # Convert to normalized camera coordinates
    u_undist = undistorted_coords[:, 0]
    v_undist = undistorted_coords[:, 1]
    
    # Convert to normalized coordinates (subtract principal point, divide by focal length)
    x_norm = (u_undist - cx) / fx
    y_norm = (v_undist - cy) / fy
    
    # Create rays in camera coordinate system
    # Camera coordinates: X=right, Y=forward, Z=up
    # For a ray pointing through pixel (x_norm, y_norm), the ray direction in camera frame is:
    # X = x_norm (rightward displacement from optical axis)
    # Y = 1 (forward along optical axis - this should be the primary direction)  
    # Z = -y_norm (upward displacement from optical axis, negative because image Y is downward)
    rays_camera = np.column_stack([x_norm, np.ones(len(x_norm)), -y_norm])
    
    # Create rotation matrices for camera orientation
    # All rotations are in camera reference frame: X=right, Y=forward, Z=up
    
    # Pitch rotation (around X-axis in camera frame - nose up/down)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    # Roll rotation (around Y-axis in camera frame - left/right bank)
    R_roll = np.array([
        [np.cos(roll_rad), 0, np.sin(roll_rad)],
        [0, 1, 0],
        [-np.sin(roll_rad), 0, np.cos(roll_rad)]
    ])
    
    # Heading rotation (around Z-axis in camera frame - left/right turn)
    # Standard navigation convention: 0° = North, 90° = East
    # This rotates the camera's forward direction to align with the heading
    R_heading = np.array([
        [np.cos(heading_rad), np.sin(heading_rad), 0],
        [-np.sin(heading_rad), np.cos(heading_rad), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: pitch, then roll, then heading (all in camera frame)
    # Order matters: we apply rotations in sequence
    R_camera_to_world = R_heading @ R_roll @ R_pitch
    
    # Transform rays to world coordinate system
    rays_world = (R_camera_to_world @ rays_camera.T).T
    
    # Extract ray components in world coordinates (ENU)
    ray_x = rays_world[:, 0]  # East component
    ray_y = rays_world[:, 1]  # North component  
    ray_z = rays_world[:, 2]  # Up component
    
    # Calculate intersection with seafloor plane (Z = 0)
    # Camera is at height +distance_off_bottom above the seafloor (seafloor at Z=0)
    # Ray equation: P = camera_position + t * ray_direction
    # Camera position: (camera_east, camera_north, distance_off_bottom)  
    # Seafloor plane: Z = 0
    # Solve: distance_off_bottom + t * ray_z = 0
    # Therefore: t = -distance_off_bottom / ray_z
    
    # Avoid division by zero for rays parallel to seafloor
    valid_rays = np.abs(ray_z) > 1e-10
    t = np.full(len(ray_z), np.nan)
    t[valid_rays] = -distance_off_bottom / ray_z[valid_rays]
    
    # Only keep rays that intersect the seafloor (positive t for downward-pointing rays)
    # For a camera above seafloor, we want rays with negative Z component (pointing down)
    # and positive t values (intersection in front of camera)
    valid_intersection = valid_rays & (t > 0) & (ray_z < 0)
    t[~valid_intersection] = np.nan
    
    # Calculate world coordinates relative to camera position
    x_relative = t * ray_x
    y_relative = t * ray_y
    
    # Add camera position to get absolute ENU coordinates
    x_world = x_relative + camera_east
    y_world = y_relative + camera_north
    
    # Reshape back to downsampled image dimensions
    x_world = x_world.reshape(image_height_ds, image_width_ds)
    y_world = y_world.reshape(image_height_ds, image_width_ds)
    
    return x_world, y_world


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


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth in meters.
    
    Parameters:
    -----------
    lat1, lon1 : float
        Latitude and longitude of first point in degrees
    lat2, lon2 : float
        Latitude and longitude of second point in degrees
        
    Returns:
    --------
    float
        Distance in meters
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in meters
    r = 6371000
    
    return c * r
