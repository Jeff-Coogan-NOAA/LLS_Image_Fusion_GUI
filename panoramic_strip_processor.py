"""
Panoramic strip mosaic generation from individual georeferenced GeoTIFFs.

Alignment is purely coordinate-based using each GeoTIFF's embedded CRS and
affine transform (as produced by geotiff_processor.py).  No feature matching
or image warping is performed — the georeferencing is assumed to be accurate
enough that the merge is seamless.
"""

import os
import glob
import traceback
from typing import Callable, List, Optional

import numpy as np
import rasterio
from rasterio.merge import merge


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_panoramic_strips(
    geotiff_dir: str,
    output_dir: str,
    images_per_strip: int = 100,
    resolution_m: float = 0.005,
    strip_prefix: str = "strip",
    nodata_value: int = 255,
    reverse_order: bool = False,
    merge_method: str = "first",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Merge individual georeferenced GeoTIFFs into panoramic strip mosaics.

    Parameters
    ----------
    geotiff_dir : str
        Directory containing per-image GeoTIFF files (.tif / .tiff).
        Files are sorted by filename, so the sort order must match the
        along-track order of the dive.
    output_dir : str
        Directory where strip GeoTIFFs will be saved.
    images_per_strip : int
        Number of individual GeoTIFFs to include in each strip.
    resolution_m : float
        Output pixel size in metres (e.g. 0.005 = 5 mm, 0.01 = 1 cm,
        0.002 = 2 mm).
    strip_prefix : str
        Filename prefix for output files (strip_001.tif, strip_002.tif, …).
    nodata_value : int
        Pixel value treated as background / no-data in source files.
        Defaults to 255 (white background used by geotiff_processor).
    reverse_order : bool
        If True, process GeoTIFFs in reversed filename order so that
        overlap priority is effectively flipped (useful when you want
        later files to occlude earlier ones). Default ``False``.
    merge_method : str
        Merge strategy passed to ``rasterio.merge.merge``. Common
        values are ``'first'`` (keep first non-nodata pixel) and
        ``'last'`` (keep last non-nodata pixel). Default ``'first'``.
    progress_callback : Optional[Callable[[str], None]]
        Optional function called with each progress message.

    Returns
    -------
    dict
        ``{'total_strips': int, 'success': int, 'failed': int,
           'total_geotiffs': int}``
    """

    def log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    # ------------------------------------------------------------------
    # Discover GeoTIFFs
    # ------------------------------------------------------------------
    tif_files = sorted(
        glob.glob(os.path.join(geotiff_dir, "*.tif"))
        + glob.glob(os.path.join(geotiff_dir, "*.tiff"))
    )

    if not tif_files:
        log("  No GeoTIFF files (.tif / .tiff) found in the specified directory.")
        return {"total_strips": 0, "success": 0, "failed": 0, "total_geotiffs": 0}

    log(f"  Found {len(tif_files)} GeoTIFF file(s) in {geotiff_dir}")
    if reverse_order:
        tif_files = list(reversed(tif_files))
        log("  Processing GeoTIFFs in reverse filename order (reverse_order=True).")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Partition into strips
    # ------------------------------------------------------------------
    strips: List[List[str]] = [
        tif_files[i : i + images_per_strip]
        for i in range(0, len(tif_files), images_per_strip)
    ]

    total_strips = len(strips)
    log(f"  Partitioned into {total_strips} strip(s) of up to {images_per_strip} image(s) each.")
    log(f"  Output resolution: {resolution_m * 100:.1f} cm/pixel  ({resolution_m} m/pixel)")

    stats = {
        "total_strips": total_strips,
        "success": 0,
        "failed": 0,
        "total_geotiffs": len(tif_files),
    }

    for strip_idx, strip_files in enumerate(strips, 1):
        strip_name = f"{strip_prefix}_{strip_idx:03d}.tif"
        strip_path = os.path.join(output_dir, strip_name)

        log(
            f"\n  Strip {strip_idx}/{total_strips}: "
            f"merging {len(strip_files)} GeoTIFF(s) → {strip_name}"
        )

        try:
            _merge_strip(strip_files, strip_path, resolution_m, nodata_value, merge_method, log)
            log(f"    Saved: {strip_path}")
            stats["success"] += 1
        except Exception as exc:
            log(f"    ERROR on strip {strip_idx}: {exc}")
            log(traceback.format_exc())
            stats["failed"] += 1

    log(
        f"\n  Panoramic strip summary: {stats['success']} succeeded, "
        f"{stats['failed']} failed out of {stats['total_strips']} strip(s)."
    )
    return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_crs_for_merge(
    datasets: List[rasterio.DatasetReader],
    log: Callable[[str], None],
) -> None:
    """Validate that all datasets have compatible CRS for merging.
    
    Raises RuntimeError if CRS issues are detected.
    """
    if not datasets:
        return
    
    # Check that all datasets have the same CRS
    reference_crs = datasets[0].crs
    log(f"    Reference CRS: {reference_crs}")
    
    for ds in datasets[1:]:
        if ds.crs != reference_crs:
            raise RuntimeError(
                f"CRS mismatch detected: {os.path.basename(ds.name)} uses {ds.crs}, "
                f"but first file uses {reference_crs}. All GeoTIFFs must share the same CRS."
            )
    
    # Warn if using geographic coordinates (lat/lon)
    if reference_crs.is_geographic:
        log(
            f"    WARNING: GeoTIFFs use geographic CRS ({reference_crs}). "
            f"Resolution parameter will be interpreted as degrees, not meters. "
            f"Consider reprojecting to a projected CRS (e.g., UTM) for accurate metric resolution."
        )


def _calculate_merge_dimensions(
    datasets: List[rasterio.DatasetReader],
    resolution: float,
) -> dict:
    """Calculate expected output dimensions for a merge operation.
    
    Returns dict with 'width', 'height', 'width_m', 'height_m'.
    """
    from rasterio.merge import merge
    
    # Get bounds of all datasets in their common CRS
    min_x, min_y, max_x, max_y = datasets[0].bounds
    
    # Track individual bounds for diagnostics
    all_bounds = []
    for ds in datasets:
        bounds = ds.bounds
        all_bounds.append({
            'file': os.path.basename(ds.name),
            'bounds': bounds,
            'width': bounds.right - bounds.left,
            'height': bounds.top - bounds.bottom,
        })
        min_x = min(min_x, bounds.left)
        min_y = min(min_y, bounds.bottom)
        max_x = max(max_x, bounds.right)
        max_y = max(max_y, bounds.top)
    
    # Calculate dimensions at the target resolution
    width_units = max_x - min_x
    height_units = max_y - min_y
    
    width_pixels = int(np.ceil(width_units / resolution))
    height_pixels = int(np.ceil(height_units / resolution))
    
    return {
        'width': width_pixels,
        'height': height_pixels,
        'width_m': width_units,
        'height_m': height_units,
        'bounds': (min_x, min_y, max_x, max_y),
        'individual_bounds': all_bounds,
    }


def _merge_strip(
    tif_paths: List[str],
    output_path: str,
    resolution_m: float,
    nodata_value: int,
    merge_method: str,
    log: Callable[[str], None],
) -> None:
    """Open *tif_paths*, mosaic them with rasterio.merge, and write *output_path*.

    Alignment is purely geospatial — each source file's CRS and affine
    transform determine where its pixels land in the merged output.
    The merge method controls overlap priority (e.g. 'first' or 'last').
    """
    opened: List[rasterio.DatasetReader] = []
    valid_datasets: List[rasterio.DatasetReader] = []

    try:
        for p in tif_paths:
            try:
                ds = rasterio.open(p)
                opened.append(ds)
                if ds.crs is None:
                    log(f"    Skipping {os.path.basename(p)}: no CRS embedded.")
                    continue
                valid_datasets.append(ds)
            except Exception as exc:
                log(f"    Could not open {os.path.basename(p)}: {exc}")

        if not valid_datasets:
            raise RuntimeError("No valid georeferenced GeoTIFFs could be opened for this strip.")

        # Validate CRS consistency and type
        _validate_crs_for_merge(valid_datasets, log)

        log(f"    Merging {len(valid_datasets)} valid dataset(s)…")

        # Calculate and validate output dimensions before attempting merge
        expected_dims = _calculate_merge_dimensions(valid_datasets, resolution_m)
        log(f"    Expected output: {expected_dims['width']:,} x {expected_dims['height']:,} pixels "
            f"({expected_dims['width_m']:.1f} x {expected_dims['height_m']:.1f} m)")
        
        # Analyze individual file dimensions to detect outliers
        bounds_info = expected_dims['individual_bounds']
        widths = [b['width'] for b in bounds_info]
        heights = [b['height'] for b in bounds_info]
        
        median_width = np.median(widths)
        median_height = np.median(heights)
        
        # Flag files with dimensions > 10x median (likely bad georeferencing)
        outliers = []
        for b in bounds_info:
            if b['width'] > 10 * median_width or b['height'] > 10 * median_height:
                outliers.append(b)
        
        if outliers:
            log(f"    WARNING: {len(outliers)} file(s) have unusually large dimensions (>10x median):")
            log(f"    Median dimensions: {median_width:.1f} x {median_height:.1f} m")
            for outlier in outliers[:5]:  # Show first 5 outliers
                log(f"      - {outlier['file']}: {outlier['width']:.1f} x {outlier['height']:.1f} m")
            if len(outliers) > 5:
                log(f"      ... and {len(outliers) - 5} more")
            log("    These files likely have incorrect georeferencing and should be excluded or fixed.")
        
        # Safety check: prevent absurdly large outputs
        max_pixels = 100_000_000  # 100 megapixels per band
        total_pixels = expected_dims['width'] * expected_dims['height']
        if total_pixels > max_pixels:
            raise RuntimeError(
                f"Output would be {total_pixels:,} pixels ({total_pixels / 1e6:.1f} MP), "
                f"exceeding safety limit of {max_pixels:,} pixels ({max_pixels / 1e6:.1f} MP). "
                f"This may indicate incorrect georeferencing. Check that all GeoTIFFs use a "
                f"projected CRS (e.g., UTM) and have correct affine transforms."
            )

        # rasterio.merge handles all reprojection / resampling internally.
        # res=(resolution_m, resolution_m) controls the output pixel size.
        mosaic, out_transform = merge(
            valid_datasets,
            method=merge_method,
            nodata=nodata_value,
            res=(resolution_m, resolution_m),
        )

        out_meta = valid_datasets[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "nodata": nodata_value,
                "photometric": "RGB",
            }
        )

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(mosaic)
            dst.update_tags(
                SOFTWARE="Python/Rasterio",
                STRIP_IMAGE_COUNT=str(len(valid_datasets)),
                RESOLUTION_M=str(resolution_m),
                MERGE_METHOD=merge_method,
            )

    finally:
        for ds in opened:
            try:
                ds.close()
            except Exception:
                pass
