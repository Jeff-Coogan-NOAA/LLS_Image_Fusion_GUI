# LLS & Image Fusion Processor GUI

A graphical user interface application for processing underwater imagery and LiDAR (LLS) data to generate georeferenced GeoTIFFs and 3D point clouds.

## Overview

This application combines the functionality of the `Fuse_LLS_and_Image.py` and `Create_Geotiff.py` scripts into a single, easy-to-use GUI interface. It provides flexible image selection options and batch processing capabilities.

## Features

- **Dual Output Modes:**
  - Generate georeferenced GeoTIFF images
  - Create 3D point clouds fused with LLS elevation data

- **Flexible Image Selection:**
  - Process all images in a dataset
  - Select specific images from a list file
  - Filter images by geographic location (lat/lon + radius)

- **Configurable Parameters:**
  - Camera lever arms and angular offsets
  - UTM zone and hemisphere settings
  - Point cloud downsampling and matching parameters
  - GeoTIFF output resolution (DPI)

- **User-Friendly Interface:**
  - Organized tabs for different settings
  - Real-time processing log
  - Progress tracking
  - Threaded processing to keep UI responsive

## File Structure

```
LLSvsImage_GUI/
├── gui_app.py                 # Main GUI application
├── utils.py                   # Core coordinate transformation functions
├── pointcloud_processor.py    # Point cloud generation module
├── geotiff_processor.py       # GeoTIFF generation module
└── README.md                  # This file
```

## Installation

### Requirements

```bash
pip install numpy pandas opencv-python scipy matplotlib rasterio pillow
```

### Dependencies

- **numpy**: Numerical computations
- **pandas**: Data handling and CSV processing
- **opencv-python**: Image processing and camera undistortion
- **scipy**: Spatial algorithms (KD-tree for nearest neighbor search)
- **matplotlib**: Image rendering for GeoTIFFs
- **rasterio**: GeoTIFF file creation
- **pillow**: Image file handling
- **tkinter**: GUI framework (usually included with Python)

## Usage

### Starting the Application

```bash
python gui_app.py
```

### Configuration Steps

#### 1. Input/Output Tab

- **Image List CSV**: CSV file containing image metadata and navigation data
  - Required columns: `file_name`, `Date_Time`, `AUV_Altitude`, `AUV_Pitch`, `AUV_Roll`, `AUV_Heading`, `AUV_Easting`, `AUV_Northing`, `AUV_Depth`
  - Optional columns for location filtering: `AUV_Latitude`, `AUV_Longitude`

- **LLS List CSV**: CSV file listing LLS data files and time ranges
  - Required columns: `FileName`, `TimeStart`, `TimeEnd`
  - Only needed for point cloud generation

- **LLS Directory**: Directory containing processed LLS files (L2_LLS_*.csv format)
  - Only needed for point cloud generation

- **Image Directory**: Directory containing the input images

- **Output Directory**: Directory where output files will be saved

#### 2. Processing Options Tab

**Output Types:**
- ☑ Create GeoTIFF: Generate georeferenced GeoTIFF files
- ☑ Create Point Cloud: Generate 3D point clouds with RGB colors

**Point Cloud Parameters:**
- **Downsample Factor**: Reduce image resolution by this factor (1=full, 3=1/3 size)
- **Max Distance (m)**: Maximum distance to match image pixels with LLS points (default: 0.03m)

**GeoTIFF Parameters:**
- **UTM Zone**: UTM zone number (e.g., 16)
- **Hemisphere**: N (Northern) or S (Southern)
- **DPI**: Output image resolution (default: 500)

#### 3. Image Selection Tab

Choose one of three selection modes:

- **Process All Images**: Process every image in the image list CSV

- **Select Specific Images from File**: Provide a text file with one image filename per line
  ```
  image_001.jpg
  image_005.jpg
  image_010.jpg
  ```

- **Filter by Location**: Process only images within a specified radius of a lat/lon point
  - **Latitude**: Center point latitude (decimal degrees)
  - **Longitude**: Center point longitude (decimal degrees)
  - **Radius (m)**: Search radius in meters

#### 4. Camera Parameters Tab

**Lever Arms (Body Frame):**
- **X (forward)**: Forward offset from IMU to camera (meters)
- **Y (right)**: Right offset from IMU to camera (meters)
- **Z (up)**: Upward offset from IMU to camera (meters)

**Angular Offsets:**
- **Pitch Offset**: Pitch calibration offset (degrees)
- **Roll Offset**: Roll calibration offset (degrees)
- **Heading Offset**: Heading calibration offset (degrees)

### Running Processing

1. Configure all required settings in the tabs
2. Click **Start Processing** button
3. Monitor progress in the log window
4. Output files will be saved to the specified output directory:
   - GeoTIFFs: `<output_dir>/GeoTIFFs/`
   - Point clouds: `<output_dir>/pointclouds/`

## Function-Based Architecture

The application is designed with a modular, function-based architecture for easy maintenance and reusability:

### Core Functions (utils.py)

- `pixels_to_world_coordinates()`: Convert image pixels to ENU world coordinates
- `imu_to_camera_enu()`: Transform IMU position to camera position using lever arms
- `haversine_distance()`: Calculate distance between lat/lon points

### Processing Functions

**pointcloud_processor.py:**
- `process_image_to_pointcloud()`: Process single image to point cloud
- `process_images_to_pointclouds()`: Batch process multiple images

**geotiff_processor.py:**
- `process_image_to_geotiff()`: Process single image to GeoTIFF
- `process_images_to_geotiffs()`: Batch process multiple images

### Reusability

These functions can be easily imported and used in other scripts:

```python
from utils import pixels_to_world_coordinates, imu_to_camera_enu
from pointcloud_processor import process_image_to_pointcloud
from geotiff_processor import process_image_to_geotiff

# Use in your own code
x_coords, y_coords = pixels_to_world_coordinates(
    distance_off_bottom=5.0,
    pitch=2.0,
    roll=0.5,
    heading=45.0,
    image_path="path/to/image.jpg",
    camera_east=1000.0,
    camera_north=2000.0
)
```

## Output Formats

### GeoTIFF Files

- **Format**: RGB GeoTIFF (3-band, uint8)
- **Compression**: LZW
- **Coordinate System**: UTM (configurable zone)
- **Nodata Value**: 255 (white pixels masked as background)
- **Naming**: `georef_<original_filename>.tif`

### Point Cloud Files

- **Format**: ASCII XYZ with RGB
- **Columns**: `X Y Z R G B`
- **Units**: X/Y/Z in meters (UTM), RGB in 0-255
- **Naming**: `<original_filename>_pointcloud.xyz`

## Coordinate Systems

- **Input Navigation**: ENU (East-North-Up) in UTM coordinates
- **Camera Body Frame**: X=forward, Y=right, Z=up
- **Image Coordinates**: Origin at top-left, X=right, Y=down
- **Output**: UTM projected coordinates

## Troubleshooting

### Common Issues

**"LLS file not found"**
- Verify LLS directory path
- Check that files follow the naming convention: `L2_LLS_*.csv`

**"No LLS points within time window"**
- Check that LLS and image timestamps are synchronized
- Verify LLS time is in microseconds (not nanoseconds or seconds)

**"Image not found"**
- Verify image directory path
- Check that filenames in CSV match actual image files

**Location filtering returns no images**
- Ensure CSV contains `AUV_Latitude` and `AUV_Longitude` columns
- Check lat/lon coordinates and radius values

### Performance Tips

- Use downsampling (factor of 3-5) for faster point cloud generation
- Process images in batches for large datasets
- GeoTIFF generation is generally faster than point cloud generation
- Close other applications to free up memory for large image processing

## Credits

Based on the original scripts:
- `Fuse_LLS_and_Image.py`: Point cloud fusion
- `Create_Geotiff.py`: GeoTIFF generation

## License

This software is provided as-is for research and operational use.
