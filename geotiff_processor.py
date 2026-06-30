"""
GeoTIFF generation from georeferenced images.
"""
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib
# Use a non-interactive backend to avoid starting GUI windows from background threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from PIL import Image
from typing import Optional, Callable
from utils import pixels_to_world_coordinates, imu_to_camera_enu

# ## old process_image_to_geotiff function using matplotlib scatter plot for georeferencing
# def process_image_to_geotiff(
#     image_row: pd.Series,
#     image_dir: str,
#     output_dir: str,
#     lever_arm_x: float = 0.1044,
#     lever_arm_y: float = 0.6246,
#     lever_arm_z: float = 0.0826,
#     pitch_offset: float = 0.010,
#     roll_offset: float = 0.010,
#     heading_offset: float = 0.000,
#     utm_zone: int = 16,
#     utm_hemisphere: str = 'N',
#     dpi: int = 500,
#     progress_callback: Optional[Callable[[str], None]] = None
# ) -> bool:
#     """
#     Process a single image to generate a georeferenced GeoTIFF.
    
#     Parameters:
#     -----------
#     image_row : pd.Series
#         Row from image dataframe containing navigation data
#     image_dir : str
#         Directory containing input images
#     output_dir : str
#         Directory to save output GeoTIFFs
#     lever_arm_x, lever_arm_y, lever_arm_z : float
#         Lever arm offsets from IMU to camera in vehicle body frame
#     pitch_offset, roll_offset, heading_offset : float
#         Angular offsets in degrees
#     utm_zone : int
#         UTM zone number (e.g., 16 for Zone 16)
#     utm_hemisphere : str
#         'N' for Northern hemisphere, 'S' for Southern hemisphere
#     dpi : int
#         Resolution for rendered image (dots per inch)
#     progress_callback : Optional[Callable[[str], None]]
#         Callback function for progress updates
        
#     Returns:
#     --------
#     bool
#         True if successful, False otherwise
#     """
#     def log(msg: str):
#         if progress_callback:
#             progress_callback(msg)
#         else:
#             print(msg)
    
#     try:
#         image_filename = image_row['file_name']
#         image_filepath = os.path.join(image_dir, image_filename)
        
#         if not os.path.exists(image_filepath):
#             log(f"    Warning: Image not found: {image_filepath}")
#             return False
        
#         # Extract navigation data
#         distance_off_bottom = image_row['AUV_Altitude']
#         pitch = -image_row['AUV_Pitch']
#         roll = image_row['AUV_Roll']
#         heading = image_row['AUV_Heading']
#         imu_east = image_row['AUV_Easting']
#         imu_north = image_row['AUV_Northing']
#         imu_up = image_row['AUV_Depth']
        
#         # Calculate camera position
#         camera_pos, shift = imu_to_camera_enu(
#             imu_east, imu_north, imu_up,
#             lever_arm_x, lever_arm_y, lever_arm_z,
#             pitch, roll, heading
#         )
        
#         # Get world coordinates for pixels
#         x_coords, y_coords = pixels_to_world_coordinates(
#             distance_off_bottom=distance_off_bottom + shift[2],
#             pitch=pitch + pitch_offset,
#             roll=roll + roll_offset,
#             heading=heading + heading_offset,
#             image_path=image_filepath,
#             camera_east=camera_pos[0],
#             camera_north=camera_pos[1]
#         )
        
#         # Load image
#         image = cv2.imread(image_filepath)
#         if image is None:
#             log(f"    Warning: Could not load image")
#             return False
        
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # Create plot for georeferencing
#         fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
#         # Sample the image at regular intervals
#         step = 5
#         y_indices, x_indices = np.meshgrid(
#             np.arange(0, image_rgb.shape[0], step),
#             np.arange(0, image_rgb.shape[1], step),
#             indexing='ij'
#         )
        
#         # Get corresponding world coordinates
#         x_world_sample = x_coords[y_indices, x_indices]
#         y_world_sample = y_coords[y_indices, x_indices]
#         rgb_sample = image_rgb[y_indices, x_indices]
        
#         # Flatten arrays
#         x_flat = x_world_sample.flatten()
#         y_flat = y_world_sample.flatten()
#         rgb_flat = rgb_sample.reshape(-1, 3) / 255.0
        
#         # Remove invalid coordinates
#         valid_mask = ~(np.isnan(x_flat) | np.isnan(y_flat))
#         x_valid = x_flat[valid_mask]
#         y_valid = y_flat[valid_mask]
#         rgb_valid = rgb_flat[valid_mask]
        
#         # Plot the georeferenced image
#         ax.scatter(x_valid, y_valid, c=rgb_valid, s=8)
#         ax.set_aspect('equal')
#         ax.axis('off')
#         ax.set_position([0, 0, 1, 1])
        
#         # Create output directory
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Save plot as PNG
#         plot_filename = f"georef_image_{os.path.basename(image_filename).replace('.jpg', '.png')}"
#         plot_filepath = os.path.join(output_dir, plot_filename)
#         plt.savefig(plot_filepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
#         plt.close(fig)
        
#         # Get bounds for GeoTIFF
#         x_min, x_max = np.nanmin(x_coords), np.nanmax(x_coords)
#         y_min, y_max = np.nanmin(y_coords), np.nanmax(y_coords)
        
#         # Load the saved PNG
#         png_img = Image.open(plot_filepath)
        
#         # Convert to RGB
#         if png_img.mode == 'RGBA':
#             background = Image.new('RGB', png_img.size, (255, 255, 255))
#             background.paste(png_img, mask=png_img.split()[3])
#             png_img = background
#         elif png_img.mode != 'RGB':
#             png_img = png_img.convert('RGB')
        
#         png_array = np.array(png_img)
        
#         # Create mask for white pixels
#         white_threshold = 250
#         white_mask = np.all(png_array >= white_threshold, axis=2)
        
#         # Get image dimensions
#         height, width = png_array.shape[:2]
        
#         # Calculate resolution
#         x_resolution = (x_max - x_min) / width
#         y_resolution = (y_max - y_min) / height
        
#         # Create affine transformation
#         transform = Affine.translation(x_min, y_max) * Affine.scale(x_resolution, -y_resolution)
        
#         # Define CRS
#         if utm_hemisphere.upper() == 'N':
#             epsg_code = 32600 + utm_zone
#         else:
#             epsg_code = 32700 + utm_zone
#         crs = CRS.from_epsg(epsg_code)
        
#         # Save as GeoTIFF
#         geotiff_filename = f"georef_{os.path.basename(image_filename).replace('.jpg', '.tif')}"
#         geotiff_filepath = os.path.join(output_dir, geotiff_filename)
        
#         meta = {
#             'driver': 'GTiff',
#             'dtype': 'uint8',
#             'width': width,
#             'height': height,
#             'count': 3,
#             'crs': crs,
#             'transform': transform,
#             'compress': 'lzw',
#             'tiled': True,
#             'blockxsize': 512,
#             'blockysize': 512,
#             'photometric': 'RGB',
#             'nodata': 255
#         }
        
#         with rasterio.open(geotiff_filepath, 'w', **meta) as dst:
#             for band in range(3):
#                 band_data = png_array[:, :, band].copy()
#                 band_data[white_mask] = 255
#                 dst.write(band_data, band + 1)
#             dst.update_tags(SOFTWARE='Python/Rasterio')
        
#         log(f"    Saved: {geotiff_filename}")
        
#         # Clean up intermediate PNG
#         os.remove(plot_filepath)
        
#         return True
        
#     except Exception as e:
#         log(f"    Error processing image: {e}")
#         import traceback
#         traceback.print_exc()
#         return False


def process_images_to_geotiffs(
    image_list_csv: str,
    image_dir: str,
    output_dir: str,
    selected_images: Optional[list] = None,
    lever_arm_x: float = 0.1044,
    lever_arm_y: float = 0.6246,
    lever_arm_z: float = 0.0826,
    pitch_offset: float = 0.010,
    roll_offset: float = 0.010,
    heading_offset: float = 0.000,
    utm_zone: int = 16,
    utm_hemisphere: str = 'N',
    dpi: int = 500,
    progress_callback: Optional[Callable[[str], None]] = None
) -> dict:
    """
    Process multiple images to generate GeoTIFFs.
    
    Parameters:
    -----------
    image_list_csv : str
        Path to CSV file containing image list with navigation data
    image_dir : str
        Directory containing images
    output_dir : str
        Directory to save output GeoTIFFs
    selected_images : Optional[list]
        List of specific image filenames to process (None = process all)
    lever_arm_x, lever_arm_y, lever_arm_z : float
        Lever arm offsets
    pitch_offset, roll_offset, heading_offset : float
        Angular offsets in degrees
    utm_zone : int
        UTM zone number
    utm_hemisphere : str
        'N' or 'S'
    dpi : int
        Output resolution
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
    
    # Filter by selected images if provided
    if selected_images is not None:
        df_image_list = df_image_list[df_image_list['file_name'].isin(selected_images)]
    
    stats = {'total': len(df_image_list), 'success': 0, 'failed': 0}
    
    log(f"Processing {stats['total']} images to GeoTIFFs...")
    
    for loop_idx, (idx, img_row) in enumerate(df_image_list.iterrows(), 1):
        log(f"Processing image {loop_idx}/{stats['total']}: {img_row['file_name']}")
        
        success = process_image_to_geotiff(
            img_row, image_dir, output_dir,
            lever_arm_x, lever_arm_y, lever_arm_z,
            pitch_offset, roll_offset, heading_offset,
            utm_zone, utm_hemisphere,
            progress_callback=progress_callback
        )
        
        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1
    
    log(f"\nProcessing complete: {stats['success']} successful, {stats['failed']} failed out of {stats['total']} total")
    return stats

import os
import numpy as np
import pandas as pd
import cv2
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from typing import Optional, Callable
from utils import pixels_to_world_coordinates, imu_to_camera_enu

def process_image_to_geotiff(
    image_row: pd.Series,
    image_dir: str,
    output_dir: str,
    lever_arm_x: float = 0.1044,
    lever_arm_y: float = 0.6246,
    lever_arm_z: float = 0.0826,
    pitch_offset: float = 0.010,
    roll_offset: float = 0.010,
    heading_offset: float = 0.000,
    utm_zone: int = 16,
    utm_hemisphere: str = 'N',
    progress_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Process a single image to generate a georeferenced GeoTIFF using 
    OpenCV Homography for true perspective orthorectification.
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
        
        # 1. Extract navigation data & camera position
        distance_off_bottom = image_row['AUV_Altitude']
        pitch = -image_row['AUV_Pitch']
        roll = image_row['AUV_Roll']
        heading = image_row['AUV_Heading']
        
        camera_pos, shift = imu_to_camera_enu(
            image_row['AUV_Easting'], image_row['AUV_Northing'], image_row['AUV_Depth'],
            lever_arm_x, lever_arm_y, lever_arm_z,
            pitch, roll, heading
        )
        
        # 2. Get world coordinates for pixels
        x_coords, y_coords = pixels_to_world_coordinates(
            distance_off_bottom=distance_off_bottom + shift[2],
            pitch=pitch + pitch_offset,
            roll=roll + roll_offset,
            heading=heading + heading_offset,
            image_path=image_filepath,
            camera_east=camera_pos[0],
            camera_north=camera_pos[1]
        )
        
        # 3. Load image
        image = cv2.imread(image_filepath)
        if image is None:
            log(f"    Warning: Could not load image")
            return False
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # 4. Define source and destination corners for Homography
        # Grab the 4 corners of the image in pixel space
        src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        
        # Grab the corresponding real-world coordinates for those corners
        world_corners = np.float32([
            [x_coords[0, 0], y_coords[0, 0]],       # Top-Left
            [x_coords[0, -1], y_coords[0, -1]],     # Top-Right
            [x_coords[-1, -1], y_coords[-1, -1]],   # Bottom-Right
            [x_coords[-1, 0], y_coords[-1, 0]]      # Bottom-Left
        ])
        
        # Determine the physical bounding box
        x_min, x_max = np.nanmin(x_coords), np.nanmax(x_coords)
        y_min, y_max = np.nanmin(y_coords), np.nanmax(y_coords)
        
        # Calculate optimal output resolution (meters per pixel)
        # We base this on the original image width to preserve detail
        resolution = (x_max - x_min) / w
        out_w = int(w)
        out_h = int((y_max - y_min) / resolution)
        
        # Map the real-world corners into the new flat 2D output pixel space
        dst_pts = np.float32([
            [(pt[0] - x_min) / resolution, (y_max - pt[1]) / resolution]
            for pt in world_corners
        ])
        
        # 5. Warp the Image (Orthorectification)
        # Calculate the transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Add an Alpha channel to the original image (255 = solid)
        # This will safely track our valid data pixels through the warp
        alpha = np.ones((h, w), dtype=np.uint8) * 255
        image_rgba = np.dstack((image_rgb, alpha))
        
        # Warp the image. Empty areas created by the skew will default to 0 (transparent)
        warped_rgba = cv2.warpPerspective(
            image_rgba, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        # 6. Prepare for Rasterio
        # Create affine transformation
        transform = Affine.translation(x_min, y_max) * Affine.scale(resolution, -resolution)
        
        # Define CRS
        epsg_code = 32600 + utm_zone if utm_hemisphere.upper() == 'N' else 32700 + utm_zone
        crs = CRS.from_epsg(epsg_code)
        
        # Setup output paths
        os.makedirs(output_dir, exist_ok=True)
        geotiff_filename = f"georef_{os.path.basename(image_filename).replace('.jpg', '.tif')}"
        geotiff_filepath = os.path.join(output_dir, geotiff_filename)
        
        # Use a NodData value of 0. Because we used an Alpha channel during the warp,
        # we can safely force the empty black space to 0 across all bands.
        nodata_val = 0
        warped_rgb = warped_rgba[:, :, :3]
        invalid_mask = warped_rgba[:, :, 3] == 0
        warped_rgb[invalid_mask] = nodata_val

        meta = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'width': out_w,
            'height': out_h,
            'count': 3,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'photometric': 'RGB',
            'nodata': nodata_val
        }
        
        # 7. Write the GeoTIFF
        with rasterio.open(geotiff_filepath, 'w', **meta) as dst:
            for band in range(3):
                dst.write(warped_rgb[:, :, band], band + 1)
            dst.update_tags(SOFTWARE='Python/Rasterio/OpenCV')
        
        log(f"    Saved: {geotiff_filename}")
        return True
        
    except Exception as e:
        log(f"    Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return False