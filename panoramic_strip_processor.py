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

        log(f"    Merging {len(valid_datasets)} valid dataset(s)…")

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
