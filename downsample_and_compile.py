"""
Script to downsample and compile GeoTIFF files from multiple subdirectories.

This script:
1. Iterates through subdirectories in a specified parent folder
2. Finds all .tif files (excluding those with "Bathy" in the name)
3. Downsamples each image by 10x
4. Merges all downsampled images from each subdirectory into a single GeoTIFF
5. Saves the result and moves to the next subdirectory
"""

import os
import glob
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.transform import Affine
from datetime import datetime


def normalize_raster_orientation(src):
    """
    Normalize a raster to ensure positive pixel width and height.
    If the raster has negative pixel width (is "flipped") or negative pixel height
    (is "upside down"), flip it and update the transform.
    
    Args:
        src: An open rasterio dataset
    
    Returns:
        tuple: (normalized_data, normalized_transform) or original if already normalized
    """
    transform = src.transform
    data = src.read()
    
    # Check if pixel width is negative (transform.a is the x-scale/pixel width)
    if transform.a < 0:
        print(f"    ⚠️  Detected flipped raster (negative pixel width), normalizing...")
        # Flip the data horizontally
        data = data[:, :, ::-1]  # Flip along the width axis
        
        # Create new transform with positive pixel width
        # When we flip, we need to adjust the origin (c) and invert the x-scale (a)
        transform = Affine(
            -transform.a,  # Make pixel width positive
            transform.b,
            transform.c + transform.a * src.width,  # Adjust x origin
            transform.d,
            transform.e,
            transform.f
        )
    
    # Check if pixel height is POSITIVE (transform.e is the y-scale/pixel height)
    # Note: In standard GeoTIFFs, pixel height is NEGATIVE by convention (origin at top-left)
    # We only need to flip if it's positive (truly "upside down")
    if transform.e > 0:
        print(f"    ⚠️  Detected upside down raster (positive pixel height), normalizing...")
        # Flip the data vertically
        data = data[:, ::-1, :]  # Flip along the height axis
        
        # Create new transform with negative pixel height (standard)
        # When we flip vertically, we need to adjust the origin (f) and invert the y-scale (e)
        transform = Affine(
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            -transform.e,  # Make pixel height negative (standard)
            transform.f + transform.e * src.height  # Adjust y origin
        )
    
    return data, transform


def downsample_geotiff(input_path, downsample_factor=10):
    """
    Downsample a GeoTIFF file by a specified factor.
    
    Args:
        input_path (str): Path to the input GeoTIFF file
        downsample_factor (int): Factor by which to downsample (default: 10)
    
    Returns:
        tuple: (downsampled_array, transform, profile) or (None, None, None) if error
    """
    try:
        with rasterio.open(input_path) as src:
            # First, check if the raster needs to be normalized (flipped)
            data_full, normalized_transform = normalize_raster_orientation(src)
            
            # Calculate new dimensions (divide by downsample factor)
            new_height = src.height // downsample_factor
            new_width = src.width // downsample_factor
            
            # Downsample the (potentially normalized) data
            downsampled_data = data_full[:, 
                                         ::downsample_factor,
                                         ::downsample_factor]
            
            # Ensure we get exactly the right dimensions
            downsampled_data = downsampled_data[:, :new_height, :new_width]
            
            # Update the transform to reflect the new pixel size
            # The transform describes how to map pixel coordinates to geographic coordinates
            transform = normalized_transform * normalized_transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )
            
            # Copy the profile (metadata) and update dimensions
            profile = src.profile.copy()
            profile.update({
                'height': new_height,
                'width': new_width,
                'transform': transform
            })
            
            return downsampled_data, transform, profile
            
    except Exception as e:
        print(f"  ❌ Error downsampling {input_path}: {e}")
        return None, None, None


def process_subdirectory(subdir_path, output_dir, downsample_factor=10):
    """
    Process all valid .tif files in a subdirectory:
    - Downsample each file
    - Merge them into a single GeoTIFF
    - Save the result
    
    Args:
        subdir_path (Path): Path to the subdirectory to process
        output_dir (Path): Directory where output files should be saved
        downsample_factor (int): Factor by which to downsample images
    """
    # Get subdirectory name for output file naming
    subdir_name = subdir_path.name
    print(f"\n{'='*60}")
    print(f"Processing subdirectory: {subdir_name}")
    print(f"{'='*60}")
    
    # Find all .tif files in the subdirectory
    tif_files = list(subdir_path.glob("*.tif"))
    
    # Filter out files with "Bathy" in the name (case-insensitive)
    valid_files = [f for f in tif_files if "bathy" not in f.name.lower()]
    
    print(f"Found {len(tif_files)} total .tif files")
    print(f"Found {len(valid_files)} files without 'Bathy' in the name")
    
    if not valid_files:
        print("⚠️  No valid files to process in this directory")
        return
    
    # Create a temporary directory for downsampled files
    temp_dir = output_dir / "temp_downsampled"
    temp_dir.mkdir(exist_ok=True)
    
    downsampled_files = []
    
    # Process each file: downsample and save temporarily
    for i, tif_file in enumerate(valid_files, 1):
        print(f"\n  [{i}/{len(valid_files)}] Processing: {tif_file.name}")
        
        # Downsample the file
        data, transform, profile = downsample_geotiff(str(tif_file), downsample_factor)
        
        if data is None:
            continue
        
        # Create temporary file path for downsampled image
        temp_file = temp_dir / f"temp_{tif_file.name}"
        
        # Write the downsampled image to a temporary file
        # This allows us to use rasterio.merge later
        try:
            with rasterio.open(temp_file, 'w', **profile) as dst:
                dst.write(data)
            downsampled_files.append(temp_file)
            print(f"  ✓ Downsampled from {profile['width']*downsample_factor}x{profile['height']*downsample_factor} "
                  f"to {profile['width']}x{profile['height']}")
        except Exception as e:
            print(f"  ❌ Error writing temporary file: {e}")
    
    # If we have downsampled files, merge them into a single GeoTIFF
    if downsampled_files:
        print(f"\n  Merging {len(downsampled_files)} downsampled images...")
        
        try:
            # Open all downsampled files
            src_files_to_merge = []
            for file in downsampled_files:
                src = rasterio.open(file)
                src_files_to_merge.append(src)
            
            # Merge all images into one
            # The merge function automatically handles overlapping areas
            mosaic, out_transform = merge(src_files_to_merge)
            
            # Get metadata from the first file and update it
            out_meta = src_files_to_merge[0].meta.copy()
            out_meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform
            })
            
            # Close all source files
            for src in src_files_to_merge:
                src.close()
            
            # Create output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{subdir_name}_mosaic_10x_downsampled_{timestamp}.tif"
            output_path = output_dir / output_filename
            
            # Write the merged mosaic to file
            with rasterio.open(output_path, 'w', **out_meta) as dest:
                dest.write(mosaic)
            
            print(f"  ✓ Successfully saved mosaic to: {output_path}")
            print(f"  📊 Final dimensions: {mosaic.shape[2]}x{mosaic.shape[1]} pixels")
            
        except Exception as e:
            print(f"  ❌ Error merging files: {e}")
    
    # Clean up temporary files
    print("\n  Cleaning up temporary files...")
    for temp_file in downsampled_files:
        try:
            temp_file.unlink()
        except Exception as e:
            print(f"  ⚠️  Could not delete {temp_file}: {e}")
    
    # Remove temporary directory if empty
    try:
        temp_dir.rmdir()
    except:
        pass


def main():
    """
    Main function to process all subdirectories in the parent folder.
    """
    # Define the parent directory containing all subdirectories
    parent_dir = Path(r"V:\OR2601\SAS_RAW\DIVE012_SN401\sas_raw")
    
    # Define where to save output files
    output_dir = Path(r"V:\OR2601\SAS_PRC\DIVE012\RTSAS_mosaics")
    output_dir.mkdir(exist_ok=True)
    
    # Downsample factor (10x means the output will be 1/10th the size in each dimension)
    downsample_factor = 5
    
    print("="*60)
    print("GeoTIFF Downsampling and Mosaic Compilation Script")
    print("="*60)
    print(f"Parent directory: {parent_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Downsample factor: {downsample_factor}x")
    
    # Check if parent directory exists
    if not parent_dir.exists():
        print(f"\n❌ ERROR: Parent directory does not exist: {parent_dir}")
        return
    
    # Get all subdirectories
    subdirectories = [d for d in parent_dir.iterdir() if d.is_dir()]
    
    if not subdirectories:
        print("\n⚠️  No subdirectories found to process")
        return
    
    print(f"\nFound {len(subdirectories)} subdirectories to process")
    
    # Process each subdirectory
    for i, subdir in enumerate(subdirectories, 1):
        print(f"\n\n{'#'*60}")
        print(f"Subdirectory {i}/{len(subdirectories)}")
        print(f"{'#'*60}")
        
        process_subdirectory(subdir, output_dir, downsample_factor)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Output files saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()