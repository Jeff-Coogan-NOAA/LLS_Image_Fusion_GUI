"""
Point cloud generation from images and LLS data.
"""
import os
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree
from typing import Optional, Callable
from utils import pixels_to_world_coordinates, imu_to_camera_enu


def process_image_to_pointcloud(
    image_row: pd.Series,
    lls_data: pd.DataFrame,
    lls_datetime: pd.Series,
    image_dir: str,
    output_dir: str,
    lever_arm_x: float = 0.1044,
    lever_arm_y: float = 0.6246,
    lever_arm_z: float = 0.0826,
    pitch_offset: float = 0.010,
    roll_offset: float = 0.010,
    heading_offset: float = 0.000,
    downsample: int = 3,
    max_distance: float = 0.03,
    time_window_before: float = 3.0,
    time_window_after: float = 6.0,
    progress_callback: Optional[Callable[[str], None]] = None,
    output_format: str = 'xyz'
) -> bool:
    """
    Process a single image to generate a point cloud fused with LLS data.
    
    Parameters:
    -----------
    image_row : pd.Series
        Row from image dataframe containing navigation data
    lls_data : pd.DataFrame
        LLS data with columns [E, N, Z, time_us, intensity, quality]
    lls_datetime : pd.Series
        Pandas datetime series corresponding to lls_data timestamps
    image_dir : str
        Directory containing input images
    output_dir : str
        Directory to save output point clouds
    lever_arm_x, lever_arm_y, lever_arm_z : float
        Lever arm offsets from IMU to camera in vehicle body frame
    pitch_offset, roll_offset, heading_offset : float
        Angular offsets in degrees
    downsample : int
        Downsampling factor for image processing
    max_distance : float
        Maximum distance in meters for LLS point matching
    time_window_before, time_window_after : float
        Time window in seconds before/after image for LLS data selection
    progress_callback : Optional[Callable[[str], None]]
        Callback function for progress updates
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    try:
        image_filename = image_row['file_name']
        image_filepath = os.path.join(image_dir, image_filename)
        
        if not os.path.exists(image_filepath):
            log(f"    Warning: Image not found: {image_filepath}")
            return False
        
        # Extract navigation data
        distance_off_bottom = image_row['AUV_Altitude']
        pitch = -image_row['AUV_Pitch']
        roll = image_row['AUV_Roll']
        heading = image_row['AUV_Heading']
        imu_east = image_row['AUV_Easting']
        imu_north = image_row['AUV_Northing']
        imu_up = image_row['AUV_Depth']
        
        # Get image time and filter LLS data
        img_time = pd.to_datetime(image_row['Date_Time'])

        # Ensure timezone-naive for comparison
        if hasattr(img_time, 'tz') and img_time.tz is not None:
            img_time = img_time.tz_localize(None)
        if hasattr(lls_datetime, 'dt') and lls_datetime.dt.tz is not None:
            lls_datetime = lls_datetime.dt.tz_localize(None)

        # Find closest LLS point
        time_differences = np.abs((lls_datetime - img_time).dt.total_seconds())

        # Diagnostic logging: report min difference and count within 1 second
        if len(time_differences) == 0:
            log(f"    LLS datetime series is empty for this segment, skipping.")
            return False

        # Compute basic diagnostics
        try:
            min_diff = float(np.nanmin(time_differences))
            count_within_1s = int(np.sum(time_differences <= 1.0))
        except Exception:
            min_diff = float('nan')
            count_within_1s = 0

        log(f"    LLS time diagnostics: min_diff={min_diff:.6f}s, count_within_1s={count_within_1s}, total_lls_points={len(time_differences)}")

        if np.all(time_differences > 1):
            # Provide nearest LLS time and its difference for debugging
            try:
                nearest_idx = int(np.nanargmin(time_differences))
                nearest_lls_time = lls_datetime.iloc[nearest_idx] if hasattr(lls_datetime, 'iloc') else lls_datetime[nearest_idx]
                nearest_diff = float(time_differences[nearest_idx])
                log(f"    Nearest LLS time: {nearest_lls_time} (diff {nearest_diff:.6f}s). No points within 1s, skipping.")
            except Exception:
                log(f"    No valid nearest LLS time could be determined. No points within 1s, skipping.")
            return False
        
        # Calculate camera position
        camera_pos, shift = imu_to_camera_enu(
            imu_east, imu_north, imu_up,
            lever_arm_x, lever_arm_y, lever_arm_z,
            pitch, roll, heading
        )
        
        # Get world coordinates for pixels
        x_coords, y_coords = pixels_to_world_coordinates(
            distance_off_bottom=distance_off_bottom + shift[2],
            pitch=pitch + pitch_offset,
            roll=roll + roll_offset,
            heading=heading + heading_offset,
            image_path=image_filepath,
            camera_east=camera_pos[0],
            camera_north=camera_pos[1],
            downsample=downsample
        )
        
        # Filter LLS data by time window
        start_time = img_time - pd.Timedelta(seconds=time_window_before)
        end_time = img_time + pd.Timedelta(seconds=time_window_after)
        time_mask = (lls_datetime >= start_time) & (lls_datetime <= end_time)
        
        LLS_Easting = lls_data.loc[time_mask, lls_data.columns[0]].values
        LLS_Northing = lls_data.loc[time_mask, lls_data.columns[1]].values
        LLS_Elevation = lls_data.loc[time_mask, lls_data.columns[2]].values
        
        if len(LLS_Easting) == 0:
            log(f"    No LLS data in time window, skipping.")
            return False

        log(f"    LLS spatial: {len(LLS_Easting)} points in window | "
            f"E=[{LLS_Easting.min():.3f}, {LLS_Easting.max():.3f}] "
            f"N=[{LLS_Northing.min():.3f}, {LLS_Northing.max():.3f}]")

        # Build KD-tree for nearest neighbor search
        lls_points_2d = np.column_stack([LLS_Easting, LLS_Northing])
        tree = cKDTree(lls_points_2d)
        
        # Load and process image
        image = cv2.imread(image_filepath)
        if image is None:
            log(f"    Warning: Could not load image")
            return False
        
        if downsample > 1:
            new_width = image.shape[1] // downsample
            new_height = image.shape[0] // downsample
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Flatten coordinate arrays and get RGB values
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        rgb_values = image_rgb.reshape(-1, 3)
        
        # Query nearest LLS points
        pixel_coords_2d = np.column_stack([x_flat, y_flat])
        valid_pixel_mask = ~(np.isnan(x_flat) | np.isnan(y_flat))
        
        z_coords = np.full(len(x_flat), np.nan)
        distances = np.full(len(x_flat), np.nan)
        
        if np.any(valid_pixel_mask):
            valid_coords = pixel_coords_2d[valid_pixel_mask]
            dist, indices = tree.query(valid_coords)

            # Spatial diagnostics: log pixel coordinate range and actual distances
            px_e = x_flat[valid_pixel_mask]
            px_n = y_flat[valid_pixel_mask]
            log(f"    Pixel spatial: {valid_pixel_mask.sum()} valid pixels | "
                f"E=[{px_e.min():.3f}, {px_e.max():.3f}] "
                f"N=[{px_n.min():.3f}, {px_n.max():.3f}]")
            log(f"    KD-tree distances: min={dist.min():.4f}m  median={np.median(dist):.4f}m  "
                f"max={dist.max():.4f}m  | threshold={max_distance}m  "
                f"passing={int((dist <= max_distance).sum())}/{len(dist)}")

            distances[valid_pixel_mask] = dist
            z_coords[valid_pixel_mask] = LLS_Elevation[indices]
            
            # Filter by distance
            valid_dist_mask = dist <= max_distance
            final_mask = np.zeros(len(x_flat), dtype=bool)
            final_mask[valid_pixel_mask] = valid_dist_mask
            valid_points_mask = final_mask
        else:
            valid_points_mask = np.zeros(len(x_flat), dtype=bool)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        valid_data = np.column_stack([
            x_flat[valid_points_mask],
            y_flat[valid_points_mask],
            z_coords[valid_points_mask],
            rgb_values[valid_points_mask]
        ])

        # Default to XYZ text output if no format specified
        fmt = output_format.lower() if 'output_format' in locals() else 'xyz'

        if fmt in ('laz', 'las'):
            try:
                import laspy

                # Build LAS header
                header = laspy.LasHeader(version="1.2", point_format=3)
                # Set reasonable scales/offsets
                header.scales = [0.001, 0.001, 0.001]
                header.offsets = [0.0, 0.0, 0.0]

                las = laspy.LasData(header)
                las.x = valid_data[:, 0]
                las.y = valid_data[:, 1]
                las.z = valid_data[:, 2]

                # RGB must be uint16 in LAS; upscale 0-255 to 0-65535
                rgb_uint8 = valid_data[:, 3:].astype(np.uint8)
                red = (rgb_uint8[:, 0].astype(np.uint16) << 8)
                green = (rgb_uint8[:, 1].astype(np.uint16) << 8)
                blue = (rgb_uint8[:, 2].astype(np.uint16) << 8)

                try:
                    las.red = red
                    las.green = green
                    las.blue = blue
                except Exception:
                    # Some point formats may not support color; ignore if set fails
                    pass

                out_ext = fmt
                pointcloud_filename = image_filename.replace('.jpg', f'_pointcloud.{out_ext}').replace('.png', f'_pointcloud.{out_ext}')
                pointcloud_filepath = os.path.join(output_dir, pointcloud_filename)
                las.write(pointcloud_filepath)
                log(f"    Saved: {pointcloud_filename} ({len(valid_data)} points)")
            except Exception as e:
                log(f"    LAS/LAZ write failed ({e}), falling back to XYZ text output")
                # fallback to XYZ
                pointcloud_filename = image_filename.replace('.jpg', '_pointcloud.xyz').replace('.png', '_pointcloud.xyz')
                pointcloud_filepath = os.path.join(output_dir, pointcloud_filename)
                np.savetxt(pointcloud_filepath, valid_data, fmt='%.6f %.6f %.6f %d %d %d')
                log(f"    Saved: {pointcloud_filename} ({len(valid_data)} points)")
        else:
            # Save as XYZ text
            pointcloud_filename = image_filename.replace('.jpg', '_pointcloud.xyz').replace('.png', '_pointcloud.xyz')
            pointcloud_filepath = os.path.join(output_dir, pointcloud_filename)
            np.savetxt(pointcloud_filepath, valid_data, fmt='%.6f %.6f %.6f %d %d %d')
            log(f"    Saved: {pointcloud_filename} ({len(valid_data)} points)")
        
        return True
        
    except Exception as e:
        log(f"    Error processing image: {e}")
        return False


def process_images_to_pointclouds(
    image_list_csv: str,
    lls_list_csv: str,
    lls_dir: str,
    image_dir: str,
    output_dir: str,
    selected_images: Optional[list] = None,
    lever_arm_x: float = 0.1044,
    lever_arm_y: float = 0.6246,
    lever_arm_z: float = 0.0826,
    pitch_offset: float = 0.010,
    roll_offset: float = 0.010,
    heading_offset: float = 0.000,
    downsample: int = 3,
    max_distance: float = 0.03,
    progress_callback: Optional[Callable[[str], None]] = None
    , output_format: str = 'xyz'
) -> dict:
    """
    Process multiple images to generate point clouds.
    
    Parameters:
    -----------
    image_list_csv : str
        Path to CSV file containing image list with navigation data
    lls_list_csv : str
        Path to CSV file containing LLS file list
    lls_dir : str
        Directory containing LLS files
    image_dir : str
        Directory containing images
    output_dir : str
        Directory to save output point clouds
    selected_images : Optional[list]
        List of specific image filenames to process (None = process all)
    lever_arm_x, lever_arm_y, lever_arm_z : float
        Lever arm offsets
    pitch_offset, roll_offset, heading_offset : float
        Angular offsets in degrees
    downsample : int
        Downsampling factor
    max_distance : float
        Maximum distance for LLS matching
    progress_callback : Optional[Callable[[str], None]]
        Callback for progress updates
        
    Returns:
    --------
    dict
        Summary statistics: {'total': int, 'success': int, 'failed': int}
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    df_image_list = pd.read_csv(image_list_csv)
    df_LLS_list = pd.read_csv(lls_list_csv)
    
    stats = {'total': 0, 'success': 0, 'failed': 0}
    
    for idx, row in df_LLS_list.iterrows():
        # Support new LLS list CSV schema with columns like:
        # v2_filename,start_time_unix_us,end_time_unix_us,start_datetime,end_datetime,...
        # Fallback to legacy columns if present.
        v2_filename = row.get('v2_filename') or row.get('FileName') or ''

        # Determine start/end datetimes (prefer unix us if provided)
        start_us = row.get('start_time_unix_us') if 'start_time_unix_us' in row.index else None
        end_us = row.get('end_time_unix_us') if 'end_time_unix_us' in row.index else None

        if pd.notnull(start_us):
            TimeStart = pd.to_datetime(int(start_us), unit='us')
        elif 'start_datetime' in row.index and pd.notnull(row.get('start_datetime')):
            TimeStart = pd.to_datetime(row.get('start_datetime'))
        else:
            TimeStart = row.get('TimeStart') if 'TimeStart' in row.index else None
            TimeStart = pd.to_datetime(TimeStart) if pd.notnull(TimeStart) else None

        if pd.notnull(end_us):
            TimeEnd = pd.to_datetime(int(end_us), unit='us')
        elif 'end_datetime' in row.index and pd.notnull(row.get('end_datetime')):
            TimeEnd = pd.to_datetime(row.get('end_datetime'))
        else:
            TimeEnd = row.get('TimeEnd') if 'TimeEnd' in row.index else None
            TimeEnd = pd.to_datetime(TimeEnd) if pd.notnull(TimeEnd) else None

        # Try to locate an L2 CSV corresponding to the v2 filename (legacy behavior),
        # otherwise attempt to read the provided V2 file (LAZ/LAS/CSV).
        LLS_filename = os.path.basename(v2_filename) if v2_filename else ''
        # Candidate L2 CSV name: replace leading 'LLS_' with 'L2_LLS_' and use .csv
        if LLS_filename:
            l2_candidate = os.path.splitext(LLS_filename)[0].replace("LLS_", "L2_LLS_") + '.csv'
            L2_LLS_filepath = os.path.join(lls_dir, l2_candidate)
            L2_LLS_filename = os.path.basename(l2_candidate)
        else:
            L2_LLS_filepath = None
            L2_LLS_filename = ''

        data = None

        if L2_LLS_filepath and os.path.exists(L2_LLS_filepath):
            try:
                data = pd.read_csv(L2_LLS_filepath)
            except Exception as e:
                log(f"Failed to read L2 CSV {L2_LLS_filepath}: {e}")

        if data is None:
            # Try the v2 file itself inside lls_dir or as an absolute path
            lls_path = v2_filename if os.path.isabs(v2_filename) else os.path.join(lls_dir, v2_filename)
            if not os.path.exists(lls_path):
                # if still not found, try basename in lls_dir
                lls_path = os.path.join(lls_dir, LLS_filename)

            if os.path.exists(lls_path):
                if lls_path.lower().endswith(('.laz', '.las')):
                    try:
                        import laspy

                        las = laspy.read(lls_path)
                        # In this LAZ format las.x = Easting, las.y = Northing
                        # (confirmed by coordinate range diagnostics — swap to match
                        #  the [Easting, Northing, Z] column convention used downstream)
                        easting = np.array(las.x)
                        northing = np.array(las.y)
                        z = np.array(las.z)

                        # Try to obtain time values from common LAS/LAZ fields.
                        # Preferred: an explicit microsecond field named 'time_us'.
                        times_us = None
                        try:
                            # laspy exposes point dimensions; check for 'time_us'
                            if hasattr(las, 'point_format') and hasattr(las.point_format, 'dimension_names'):
                                if 'time_us' in las.point_format.dimension_names:
                                    times_us = np.array(las['time_us']).astype(np.int64)
                        except Exception:
                            times_us = None

                        # Fallback: direct attribute
                        if times_us is None and hasattr(las, 'time_us'):
                            try:
                                times_us = np.array(las.time_us).astype(np.int64)
                            except Exception:
                                times_us = None

                        # Fallback: gps_time is common and is in seconds -> convert to microseconds
                        if times_us is None and hasattr(las, 'gps_time'):
                            try:
                                times = np.array(las.gps_time)
                                times_us = (times * 1e6).astype(np.int64)
                            except Exception:
                                times_us = None

                        # Fallback: generic 'time' dimension (often seconds)
                        if times_us is None:
                            try:
                                times = np.array(las['time'])
                                times_us = (times * 1e6).astype(np.int64)
                            except Exception:
                                times_us = None

                        # Final fallback: zero timestamps
                        if times_us is None:
                            times_us = np.zeros(len(easting), dtype=np.int64)

                        intensity = np.array(las.intensity) if hasattr(las, 'intensity') else np.zeros(len(easting), dtype=np.int32)
                        quality = np.zeros(len(easting), dtype=np.float32)

                        data = pd.DataFrame({
                            0: easting,
                            1: northing,
                            2: z,
                            3: times_us,
                            4: intensity,
                            5: quality
                        })
                    except Exception as e:
                        log(f"Failed to read LAZ/LAS {lls_path}: {e}")
                        data = None
                elif lls_path.lower().endswith('.csv'):
                    try:
                        data = pd.read_csv(lls_path)
                    except Exception as e:
                        log(f"Failed to read CSV {lls_path}: {e}")
                        data = None

        if data is None:
            log(f"LLS file not found or unreadable for entry: {v2_filename}")
            continue

        # Convert LLS time column to datetimes.
        # Prefer an explicit 'time_us' column (microsecond unix time) when present.
        try:
            if isinstance(data, pd.DataFrame):
                if 'time_us' in data.columns:
                    LLS_datetime = pd.to_datetime(data['time_us'].astype('int64'), unit='us')
                elif 'time' in data.columns:
                    # ambiguous: try microseconds first, then fallback to seconds/parsing
                    try:
                        LLS_datetime = pd.to_datetime(data['time'].astype('int64'), unit='us')
                    except Exception:
                        LLS_datetime = pd.to_datetime(data['time'])
                elif 'gps_time' in data.columns:
                    try:
                        LLS_datetime = pd.to_datetime(data['gps_time'].astype('int64'), unit='s')
                    except Exception:
                        LLS_datetime = pd.to_datetime(data['gps_time'])
                else:
                    # default: column index 3 (legacy datasets)
                    try:
                        LLS_datetime = pd.to_datetime(data.iloc[:, 3], unit='us')
                    except Exception:
                        LLS_datetime = pd.to_datetime(data.iloc[:, 3])
            else:
                # non-DataFrame (constructed from LAS) - assume column index 3 contains microsecond unix time
                try:
                    LLS_datetime = pd.to_datetime(data.iloc[:, 3], unit='us')
                except Exception:
                    LLS_datetime = pd.to_datetime(data.iloc[:, 3])
        except Exception:
            # final fallback: attempt generic parsing of column 3
            LLS_datetime = pd.to_datetime(data.iloc[:, 3])

        # If TimeStart/TimeEnd are available, filter images by that interval
        if TimeStart is not None and TimeEnd is not None:
            df_image_subset = df_image_list[
                (pd.to_datetime(df_image_list['Date_Time']) >= TimeStart) &
                (pd.to_datetime(df_image_list['Date_Time']) <= TimeEnd)
            ]
        else:
            # If no interval provided, use entire image list
            df_image_subset = df_image_list
        
        # Filter by selected images if provided
        if selected_images is not None:
            df_image_subset = df_image_subset[
                df_image_subset['file_name'].isin(selected_images)
            ]
        
        log(f"Processing LLS segment: {L2_LLS_filename}")
        log(f"  Found {len(df_image_subset)} images in time range")
        
        for loop_idx, (img_idx, img_row) in enumerate(df_image_subset.iterrows(), 1):
            stats['total'] += 1
            log(f"  Processing image {loop_idx}/{len(df_image_subset)}: {img_row['file_name']}")
            
            success = process_image_to_pointcloud(
                img_row, data, LLS_datetime, image_dir, output_dir,
                lever_arm_x, lever_arm_y, lever_arm_z,
                pitch_offset, roll_offset, heading_offset,
                downsample, max_distance,
                progress_callback=progress_callback
                , output_format=output_format
            )
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
    
    log(f"\nProcessing complete: {stats['success']} successful, {stats['failed']} failed out of {stats['total']} total")
    return stats
