"""
LAZ point cloud colorization from images using voxel-based RGB mapping.

Pipeline:
  1. Read the input LAZ/LAS file.
  2. For each image, project pixels into world Easting/Northing coordinates
     using the existing pixel-to-world conversion functions.
  3. Accumulate pixel RGB values into a 2-D voxel grid in EN space
     (the grid cell side length is the configurable voxel_size, default 0.01 m).
     4. Assign voxel colours to the original LAZ points via one of four methods:
         nearest  – pixel whose EN position is closest to the voxel centre
                        (search radius = voxel_size)
         mean     – arithmetic mean of all pixel RGB values in the voxel
         median   – per-channel median of all pixel RGB values in the voxel
         std      – per-channel standard deviation of pixel RGB values in the voxel
  5. Write the coloured point cloud as a new LAZ file.

Note: The coordinate convention inherited from the project's LAZ files stores
Northing in las.x and Easting in las.y.  The functions here honour that
convention so that the coloured output file is spatially consistent with the
original data.
"""

import os
import numpy as np
import pandas as pd
import cv2
from typing import Optional, Callable

from utils import pixels_to_world_coordinates, imu_to_camera_enu


def colorize_laz_from_images(
    laz_input_path: str,
    image_list_csv: str,
    image_dir: str,
    output_path: str,
    voxel_size: float = 0.01,
    rgb_method: str = 'nearest',
    lever_arm_x: float = 0.1044,
    lever_arm_y: float = 0.6246,
    lever_arm_z: float = 0.0826,
    pitch_offset: float = 0.010,
    roll_offset: float = 0.010,
    heading_offset: float = 0.000,
    downsample: int = 3,
    selected_images: Optional[list] = None,
    lls_list_csv: Optional[str] = None,
    time_buffer: float = 5.0,
    chunk_step: float = 30.0,
    time_window: float = 10.0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Colorize a LAZ/LAS point cloud by projecting image pixels into world EN
    coordinates and aggregating pixel RGB values into a 2-D voxel grid.

    Parameters
    ----------
    laz_input_path : str
        Path to the input LAZ or LAS file.
    image_list_csv : str
        Path to the CSV listing images with navigation columns:
        file_name, AUV_Pitch, AUV_Roll, AUV_Heading,
        AUV_Easting, AUV_Northing, AUV_Depth, AUV_Altitude.
    image_dir : str
        Directory containing the image files referenced by image_list_csv.
    output_path : str
        Desired path for the coloured output LAZ file.
    voxel_size : float
        Side length of each 2-D voxel in the EN plane, in metres.
        Default 0.01 m (1 cm).
    rgb_method : str
        How to derive a single RGB value for each voxel:
        'nearest' – pixel closest to the voxel centre (within voxel_size);
        'mean'    – arithmetic mean of all pixel RGB values in the voxel;
        'median'  – per-channel median of all pixel RGB values in the voxel.
        'std'     – per-channel standard deviation of pixel RGB values in the voxel
    lever_arm_x/y/z : float
        Body-frame lever arm offsets (metres) from IMU to camera.
    pitch_offset, roll_offset, heading_offset : float
        Angular correction offsets in degrees.
    downsample : int
        Image downsampling factor applied before pixel projection.
        Higher values reduce memory use and processing time.
    selected_images : list or None
        Restrict processing to these filenames.  None = process all rows.
    lls_list_csv : str or None
        Path to the LLS list CSV (columns: v2_filename, start_time_unix_us,
        end_time_unix_us, start_datetime, end_datetime, …).  When provided,
        the LAZ filename is looked up in v2_filename and images are pre-filtered
        to those whose Date_Time falls within
        [start_datetime − time_buffer, end_datetime + time_buffer].
    time_buffer : float
        Seconds of padding added on each side of the LLS time window when
        filtering images.  Default 5 s.
    progress_callback : callable or None
        Optional function accepting a single str for log output.

    Returns
    -------
    dict
        Summary with keys: total_images, processed_images,
        colored_points, total_points.
    """

    def log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    stats = {
        'total_images': 0,
        'processed_images': 0,
        'colored_points': 0,
        'total_points': 0,
    }

    method = rgb_method.lower().strip()
    if method not in ('nearest', 'mean', 'median', 'std'):
        raise ValueError(
            f"rgb_method must be one of: 'nearest', 'mean', 'median', 'std'; got '{rgb_method}'"
        )

    # ------------------------------------------------------------------
    # 1. Read input LAZ / LAS
    # ------------------------------------------------------------------
    try:
        import laspy
    except ImportError:
        raise ImportError(
            "laspy is required for LAZ colourisation.  "
            "Install with:  pip install laspy[lazrs]"
        )

    log(f"Reading: {os.path.basename(laz_input_path)}")
    las_in = laspy.read(laz_input_path)

    # Project-wide convention: las.x stores Easting, las.y stores Northing.
    # (See note in pointcloud_processor.py.)
    raw_x = np.array(las_in.x, dtype=np.float64)   # Easting in file coords
    raw_y = np.array(las_in.y, dtype=np.float64)   # Northing  in file coords
    raw_z = np.array(las_in.z, dtype=np.float64)

    las_e = raw_x   # Easting
    las_n = raw_y   # Northing
    n_pts = len(las_e)
    stats['total_points'] = n_pts

    log(f"  {n_pts:,} points")
    log(f"  Easting  [{las_e.min():.3f},  {las_e.max():.3f}]")
    log(f"  Northing [{las_n.min():.3f},  {las_n.max():.3f}]")

    # Bounding box for pixel pre-filtering (one voxel of padding)
    e_lo = las_e.min() - voxel_size
    e_hi = las_e.max() + voxel_size
    n_lo = las_n.min() - voxel_size
    n_hi = las_n.max() + voxel_size

    # ------------------------------------------------------------------
    # 2. Read image list
    # ------------------------------------------------------------------
    df_imgs = pd.read_csv(image_list_csv)
    if selected_images is not None:
        df_imgs = df_imgs[df_imgs['file_name'].isin(selected_images)]

    # ------------------------------------------------------------------
    # 2a. Time-based pre-filter using the LLS list CSV
    # ------------------------------------------------------------------
    if lls_list_csv is not None:
        laz_basename = os.path.basename(laz_input_path)
        df_lls_list = pd.read_csv(lls_list_csv)

        # Match on exact filename; fall back to stem-only comparison
        match = df_lls_list[df_lls_list['v2_filename'] == laz_basename]
        if match.empty:
            laz_stem = os.path.splitext(laz_basename)[0]
            match = df_lls_list[
                df_lls_list['v2_filename'].str.replace(r'\.[^.]+$', '', regex=True) == laz_stem
            ]

        if match.empty:
            log(
                f"  Warning: '{laz_basename}' not found in LLS list CSV – "
                f"skipping time-based image filter."
            )
        else:
            lls_row = match.iloc[0]
            t_start = pd.to_datetime(lls_row['start_datetime'])
            t_end   = pd.to_datetime(lls_row['end_datetime'])

            # Strip timezone info for consistent comparison
            if t_start.tzinfo is not None:
                t_start = t_start.tz_localize(None)
            if t_end.tzinfo is not None:
                t_end = t_end.tz_localize(None)

            t_lo = t_start - pd.Timedelta(seconds=time_buffer)
            t_hi = t_end   + pd.Timedelta(seconds=time_buffer)

            if 'Date_Time' not in df_imgs.columns:
                log(
                    "  Warning: 'Date_Time' column not found in image list CSV – "
                    "skipping time-based image filter."
                )
            else:
                img_times = pd.to_datetime(df_imgs['Date_Time'])
                if img_times.dt.tz is not None:
                    img_times = img_times.dt.tz_localize(None)

                time_mask = (img_times >= t_lo) & (img_times <= t_hi)
                n_before = len(df_imgs)
                df_imgs  = df_imgs[time_mask].reset_index(drop=True)
                log(
                    f"  Time filter: LLS window {t_start} – {t_end}  "
                    f"(±{time_buffer} s)  →  "
                    f"{len(df_imgs)} / {n_before} images retained."
                )

    stats['total_images'] = len(df_imgs)
    log(f"  {len(df_imgs)} images to project")

    # ------------------------------------------------------------------
    # 2b. Determine whether to use chunked (time_us) or all-at-once processing
    # ------------------------------------------------------------------
    try:
        _ = las_in['time_us']
        has_time_us = True
    except Exception:
        has_time_us = False

    if has_time_us and 'Date_Time' in df_imgs.columns:
        log(
            f"\ntime_us dimension detected - using {chunk_step:.0f} s chunked "
            f"processing (+/-{time_window:.0f} s image overlap) ..."
        )

        # ----------------------------------------------------------------
        # 3b. Parse image timestamps; sort both images and LAZ by time
        # ----------------------------------------------------------------
        las_unix = np.array(las_in['time_us'], dtype=np.float64) / 1e6  # -> Unix seconds

        img_dt = pd.to_datetime(df_imgs['Date_Time'])
        if img_dt.dt.tz is not None:
            img_dt = img_dt.dt.tz_localize(None)
        img_unix_s = img_dt.apply(lambda dt: dt.timestamp()).values

        sort_img   = np.argsort(img_unix_s)
        df_imgs    = df_imgs.iloc[sort_img].reset_index(drop=True)
        img_unix_s = img_unix_s[sort_img]

        sort_pts    = np.argsort(las_unix)
        sorted_unix = las_unix[sort_pts]
        sorted_e    = las_e[sort_pts]
        sorted_n    = las_n[sort_pts]

        R_out = np.zeros(n_pts, dtype=np.uint16)
        G_out = np.zeros(n_pts, dtype=np.uint16)
        B_out = np.zeros(n_pts, dtype=np.uint16)

        # ----------------------------------------------------------------
        # 4b. Step through LAZ in chunk_step windows
        # ----------------------------------------------------------------
        t_lo_global = sorted_unix[0]
        t_hi_global = sorted_unix[-1]
        chunk_edges = np.arange(t_lo_global, t_hi_global + chunk_step, chunk_step)
        n_chunks    = len(chunk_edges)

        log(
            f"  {n_chunks} chunks  |  "
            f"LAZ span {t_hi_global - t_lo_global:.1f} s  |  "
            f"{len(df_imgs)} images available"
        )

        processed_img_set: set = set()

        for ci, chunk_lo in enumerate(chunk_edges):
            chunk_hi = chunk_lo + chunk_step

            # Points in this time chunk (binary search)
            pt_lo = int(np.searchsorted(sorted_unix, chunk_lo, side='left'))
            pt_hi = int(np.searchsorted(sorted_unix, chunk_hi, side='left'))
            if pt_lo >= pt_hi:
                continue

            chunk_e  = sorted_e[pt_lo:pt_hi]
            chunk_n  = sorted_n[pt_lo:pt_hi]
            orig_pts = sort_pts[pt_lo:pt_hi]   # original LAZ indices for this chunk

            # Spatial bounding box of this chunk
            e_lo_c = chunk_e.min() - voxel_size
            e_hi_c = chunk_e.max() + voxel_size
            n_lo_c = chunk_n.min() - voxel_size
            n_hi_c = chunk_n.max() + voxel_size

            # Images within this chunk's time window (with overlap buffer)
            win_lo = chunk_lo - time_window
            win_hi = chunk_hi + time_window
            img_lo = int(np.searchsorted(img_unix_s, win_lo, side='left'))
            img_hi = int(np.searchsorted(img_unix_s, win_hi, side='right'))
            chunk_img_df = df_imgs.iloc[img_lo:img_hi]

            if len(chunk_img_df) == 0:
                continue

            t_chunk_str = pd.Timestamp(chunk_lo, unit='s').strftime('%H:%M:%S')
            log(
                f"  [{ci + 1}/{n_chunks}]  {t_chunk_str}  "
                f"{pt_hi - pt_lo:,} pts  |  {len(chunk_img_df)} imgs"
            )

            accum_chunk: list = []

            for _, img_row in chunk_img_df.iterrows():
                fname = str(img_row['file_name'])
                fpath = os.path.join(image_dir, fname)
                if not os.path.exists(fpath):
                    continue
                try:
                    altitude = float(img_row['AUV_Altitude'])
                    pitch    = -float(img_row['AUV_Pitch'])
                    roll     = float(img_row['AUV_Roll'])
                    heading  = float(img_row['AUV_Heading'])
                    imu_e    = float(img_row['AUV_Easting'])
                    imu_n    = float(img_row['AUV_Northing'])
                    imu_u    = float(img_row['AUV_Depth'])

                    cam_pos, shift = imu_to_camera_enu(
                        imu_e, imu_n, imu_u,
                        lever_arm_x, lever_arm_y, lever_arm_z,
                        pitch, roll, heading,
                    )
                    px_e, px_n = pixels_to_world_coordinates(
                        distance_off_bottom=altitude + shift[2],
                        pitch=pitch + pitch_offset,
                        roll=roll + roll_offset,
                        heading=heading + heading_offset,
                        image_path=fpath,
                        camera_east=cam_pos[0],
                        camera_north=cam_pos[1],
                        downsample=downsample,
                    )
                    img_bgr = cv2.imread(fpath)
                    if img_bgr is None:
                        continue
                    if downsample > 1:
                        img_bgr = cv2.resize(
                            img_bgr,
                            (img_bgr.shape[1] // downsample, img_bgr.shape[0] // downsample),
                            interpolation=cv2.INTER_AREA,
                        )
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    e_flat   = px_e.flatten()
                    n_flat   = px_n.flatten()
                    rgb_flat = img_rgb.reshape(-1, 3)

                    mask = (
                        ~np.isnan(e_flat) & ~np.isnan(n_flat)
                        & (e_flat >= e_lo_c) & (e_flat <= e_hi_c)
                        & (n_flat >= n_lo_c) & (n_flat <= n_hi_c)
                    )
                    if not mask.any():
                        continue

                    ev = e_flat[mask]
                    nv = n_flat[mask]
                    rv = rgb_flat[mask].astype(np.int32)
                    ki = np.floor(ev / voxel_size).astype(np.int64)
                    kj = np.floor(nv / voxel_size).astype(np.int64)

                    if method == 'nearest':
                        vc_e    = (ki + 0.5) * voxel_size
                        vc_n    = (kj + 0.5) * voxel_size
                        dist_sq = (ev - vc_e) ** 2 + (nv - vc_n) ** 2
                        df_pix  = pd.DataFrame({
                            'ki': ki, 'kj': kj, 'dist_sq': dist_sq,
                            'R': rv[:, 0], 'G': rv[:, 1], 'B': rv[:, 2],
                        })
                        df_agg = (
                            df_pix.sort_values('dist_sq')
                            .groupby(['ki', 'kj'], sort=False).first()
                            .reset_index()[['ki', 'kj', 'dist_sq', 'R', 'G', 'B']]
                        )
                    elif method in ('mean', 'std'):
                        # For 'mean' we keep per-voxel sums+count; for 'std' also keep sum of squares
                        if method == 'mean':
                            df_pix = pd.DataFrame({
                                'ki': ki, 'kj': kj,
                                'R_sum': rv[:, 0].astype(np.float64),
                                'G_sum': rv[:, 1].astype(np.float64),
                                'B_sum': rv[:, 2].astype(np.float64),
                                'count': np.ones(len(rv), dtype=np.int64),
                            })
                            df_agg = (
                                df_pix.groupby(['ki', 'kj'])
                                .agg(
                                    R_sum=('R_sum', 'sum'), G_sum=('G_sum', 'sum'),
                                    B_sum=('B_sum', 'sum'), count=('count', 'sum'),
                                )
                                .reset_index()
                            )
                        else:  # std
                            df_pix = pd.DataFrame({
                                'ki': ki, 'kj': kj,
                                'R_sum': rv[:, 0].astype(np.float64),
                                'G_sum': rv[:, 1].astype(np.float64),
                                'B_sum': rv[:, 2].astype(np.float64),
                                'R_sq': (rv[:, 0].astype(np.float64) ** 2),
                                'G_sq': (rv[:, 1].astype(np.float64) ** 2),
                                'B_sq': (rv[:, 2].astype(np.float64) ** 2),
                                'count': np.ones(len(rv), dtype=np.int64),
                            })
                            df_agg = (
                                df_pix.groupby(['ki', 'kj'])
                                .agg(
                                    R_sum=('R_sum', 'sum'), G_sum=('G_sum', 'sum'),
                                    B_sum=('B_sum', 'sum'),
                                    R_sq=('R_sq', 'sum'), G_sq=('G_sq', 'sum'), B_sq=('B_sq', 'sum'),
                                    count=('count', 'sum'),
                                )
                                .reset_index()
                            )
                    else:  # median – pre-aggregate per voxel within this image
                        df_pix = pd.DataFrame({
                            'ki': ki, 'kj': kj,
                            'R': rv[:, 0], 'G': rv[:, 1], 'B': rv[:, 2],
                        })
                        df_agg = (
                            df_pix.groupby(['ki', 'kj'])[['R', 'G', 'B']]
                            .median().round().astype(np.int32).reset_index()
                        )

                    accum_chunk.append(df_agg)
                    processed_img_set.add(fname)

                except Exception as exc:
                    log(f"    Error processing {fname}: {exc}")
                    continue

            if not accum_chunk:
                continue

            # Aggregate voxel colours for this chunk
            df_all = pd.concat(accum_chunk, ignore_index=True)

            if method == 'nearest':
                df_color = (
                    df_all.sort_values('dist_sq')
                    .groupby(['ki', 'kj'], sort=False).first()[['R', 'G', 'B']]
                    .reset_index()
                )
            elif method == 'mean':
                df_total = (
                    df_all.groupby(['ki', 'kj'])
                    .agg(
                        R_sum=('R_sum', 'sum'), G_sum=('G_sum', 'sum'),
                        B_sum=('B_sum', 'sum'), count=('count', 'sum'),
                    )
                    .reset_index()
                )
                df_color = df_total[['ki', 'kj']].copy()
                df_color['R'] = (df_total['R_sum'] / df_total['count']).round().astype(np.int32)
                df_color['G'] = (df_total['G_sum'] / df_total['count']).round().astype(np.int32)
                df_color['B'] = (df_total['B_sum'] / df_total['count']).round().astype(np.int32)
            elif method == 'std':
                df_total = (
                    df_all.groupby(['ki', 'kj'])
                    .agg(
                        R_sum=('R_sum', 'sum'), G_sum=('G_sum', 'sum'), B_sum=('B_sum', 'sum'),
                        R_sq=('R_sq', 'sum'), G_sq=('G_sq', 'sum'), B_sq=('B_sq', 'sum'),
                        count=('count', 'sum'),
                    )
                    .reset_index()
                )
                df_color = df_total[['ki', 'kj']].copy()
                # compute per-channel std = sqrt(E[x^2] - E[x]^2)
                with np.errstate(invalid='ignore', divide='ignore'):
                    r_mean = df_total['R_sum'] / df_total['count']
                    g_mean = df_total['G_sum'] / df_total['count']
                    b_mean = df_total['B_sum'] / df_total['count']
                    r_var = df_total['R_sq'] / df_total['count'] - (r_mean ** 2)
                    g_var = df_total['G_sq'] / df_total['count'] - (g_mean ** 2)
                    b_var = df_total['B_sq'] / df_total['count'] - (b_mean ** 2)
                    r_std = np.sqrt(np.clip(r_var.values, 0.0, None))
                    g_std = np.sqrt(np.clip(g_var.values, 0.0, None))
                    b_std = np.sqrt(np.clip(b_var.values, 0.0, None))
                df_color['R'] = np.round(r_std).astype(np.int32)
                df_color['G'] = np.round(g_std).astype(np.int32)
                df_color['B'] = np.round(b_std).astype(np.int32)
            else:  # median
                df_color = (
                    df_all.groupby(['ki', 'kj'])[['R', 'G', 'B']]
                    .median().round().astype(np.int32).reset_index()
                )

            # Assign voxel colours to this chunk's LAZ points
            ki_las    = np.floor(chunk_e / voxel_size).astype(np.int64)
            kj_las    = np.floor(chunk_n / voxel_size).astype(np.int64)
            df_keys   = pd.DataFrame({'ki': ki_las, 'kj': kj_las})
            df_merged = df_keys.merge(df_color, on=['ki', 'kj'], how='left')

            has_col = df_merged['R'].notna().values

            def _u16_chunk(col: str) -> np.ndarray:
                return (df_merged[col].fillna(0).values.astype(np.uint16) << 8)

            R_out[orig_pts] = np.where(has_col, _u16_chunk('R'), np.uint16(0)).astype(np.uint16)
            G_out[orig_pts] = np.where(has_col, _u16_chunk('G'), np.uint16(0)).astype(np.uint16)
            B_out[orig_pts] = np.where(has_col, _u16_chunk('B'), np.uint16(0)).astype(np.uint16)
            stats['colored_points'] += int(has_col.sum())

        stats['processed_images'] = len(processed_img_set)
        pct = 100.0 * stats['colored_points'] / max(n_pts, 1)
        log(
            f"\nTotal: {stats['colored_points']:,} / {n_pts:,} points coloured "
            f"({pct:.1f} %)  |  {stats['processed_images']} unique images projected."
        )
        red_u16   = R_out
        green_u16 = G_out
        blue_u16  = B_out

    else:
        # ----------------------------------------------------------------
        # All-at-once fallback (no time_us dimension or no Date_Time column)
        # ----------------------------------------------------------------
        if has_time_us:
            log(
                "  Warning: 'Date_Time' column missing from image CSV - "
                "falling back to all-at-once processing."
            )
        else:
            log("  time_us not found in LAZ - using all-at-once processing.")

        if method == 'median':
            log(
                "  Note: 'median' pre-aggregates pixels per voxel within each image "
                "before the global median to reduce memory usage."
            )

        # ------------------------------------------------------------------
        # 3. Project images and accumulate voxel colour data
        # ------------------------------------------------------------------
        accum: list = []

        total = stats['total_images']
        for seq, (_, img_row) in enumerate(df_imgs.iterrows(), 1):
            fname = str(img_row['file_name'])
            fpath = os.path.join(image_dir, fname)

            if not os.path.exists(fpath):
                log(f"  [{seq}/{total}] Not found - skipping: {fname}")
                continue

            log(f"  [{seq}/{total}] {fname}")

            try:
                altitude = float(img_row['AUV_Altitude'])
                pitch    = -float(img_row['AUV_Pitch'])   # sign convention matches pointcloud_processor
                roll     = float(img_row['AUV_Roll'])
                heading  = float(img_row['AUV_Heading'])
                imu_e    = float(img_row['AUV_Easting'])
                imu_n    = float(img_row['AUV_Northing'])
                imu_u    = float(img_row['AUV_Depth'])

                cam_pos, shift = imu_to_camera_enu(
                    imu_e, imu_n, imu_u,
                    lever_arm_x, lever_arm_y, lever_arm_z,
                    pitch, roll, heading,
                )

                px_e, px_n = pixels_to_world_coordinates(
                    distance_off_bottom=altitude + shift[2],
                    pitch=pitch + pitch_offset,
                    roll=roll + roll_offset,
                    heading=heading + heading_offset,
                    image_path=fpath,
                    camera_east=cam_pos[0],
                    camera_north=cam_pos[1],
                    downsample=downsample,
                )

                # Load image and resize to match projected pixel grid
                img_bgr = cv2.imread(fpath)
                if img_bgr is None:
                    log(f"    Could not read image - skipping")
                    continue
                if downsample > 1:
                    img_bgr = cv2.resize(
                        img_bgr,
                        (img_bgr.shape[1] // downsample, img_bgr.shape[0] // downsample),
                        interpolation=cv2.INTER_AREA,
                    )
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                e_flat  = px_e.flatten()
                n_flat  = px_n.flatten()
                rgb_flat = img_rgb.reshape(-1, 3)

                # Keep only valid pixels that lie within the LAZ bounding box
                mask = (
                    ~np.isnan(e_flat) & ~np.isnan(n_flat)
                    & (e_flat >= e_lo) & (e_flat <= e_hi)
                    & (n_flat >= n_lo) & (n_flat <= n_hi)
                )
                if not mask.any():
                    log(f"    No pixels overlap LAZ extent - skipping")
                    continue

                ev  = e_flat[mask]
                nv  = n_flat[mask]
                rv  = rgb_flat[mask].astype(np.int32)

                # Voxel indices for each valid pixel
                ki = np.floor(ev / voxel_size).astype(np.int64)
                kj = np.floor(nv / voxel_size).astype(np.int64)

                log(f"    {int(mask.sum()):,} valid pixels within LAZ extent")

                if method == 'nearest':
                    vc_e   = (ki + 0.5) * voxel_size
                    vc_n   = (kj + 0.5) * voxel_size
                    dist_sq = (ev - vc_e) ** 2 + (nv - vc_n) ** 2

                    df_pix = pd.DataFrame({
                        'ki': ki, 'kj': kj,
                        'dist_sq': dist_sq,
                        'R': rv[:, 0], 'G': rv[:, 1], 'B': rv[:, 2],
                    })
                    df_agg = (
                        df_pix.sort_values('dist_sq')
                        .groupby(['ki', 'kj'], sort=False)
                        .first()
                        .reset_index()[['ki', 'kj', 'dist_sq', 'R', 'G', 'B']]
                    )

                elif method in ('mean', 'std'):
                    if method == 'mean':
                        df_pix = pd.DataFrame({
                            'ki': ki, 'kj': kj,
                            'R_sum': rv[:, 0].astype(np.float64),
                            'G_sum': rv[:, 1].astype(np.float64),
                            'B_sum': rv[:, 2].astype(np.float64),
                            'count': np.ones(len(rv), dtype=np.int64),
                        })
                        df_agg = (
                            df_pix.groupby(['ki', 'kj'])
                            .agg(
                                R_sum=('R_sum', 'sum'), G_sum=('G_sum', 'sum'),
                                B_sum=('B_sum', 'sum'), count=('count', 'sum'),
                            )
                            .reset_index()
                        )
                    else:  # std
                        df_pix = pd.DataFrame({
                            'ki': ki, 'kj': kj,
                            'R_sum': rv[:, 0].astype(np.float64),
                            'G_sum': rv[:, 1].astype(np.float64),
                            'B_sum': rv[:, 2].astype(np.float64),
                            'R_sq': (rv[:, 0].astype(np.float64) ** 2),
                            'G_sq': (rv[:, 1].astype(np.float64) ** 2),
                            'B_sq': (rv[:, 2].astype(np.float64) ** 2),
                            'count': np.ones(len(rv), dtype=np.int64),
                        })
                        df_agg = (
                            df_pix.groupby(['ki', 'kj'])
                            .agg(
                                R_sum=('R_sum', 'sum'), G_sum=('G_sum', 'sum'), B_sum=('B_sum', 'sum'),
                                R_sq=('R_sq', 'sum'), G_sq=('G_sq', 'sum'), B_sq=('B_sq', 'sum'),
                                count=('count', 'sum'),
                            )
                            .reset_index()
                        )

                else:  # median – pre-aggregate per voxel within this image
                    df_pix = pd.DataFrame({
                        'ki': ki, 'kj': kj,
                        'R': rv[:, 0], 'G': rv[:, 1], 'B': rv[:, 2],
                    })
                    df_agg = (
                        df_pix.groupby(['ki', 'kj'])[['R', 'G', 'B']]
                        .median().round().astype(np.int32).reset_index()
                    )

                accum.append(df_agg)
                stats['processed_images'] += 1

            except Exception as exc:
                log(f"    Error processing {fname}: {exc}")
                continue

        # ------------------------------------------------------------------
        # 4. Global aggregation across all accumulated image frames
        # ------------------------------------------------------------------
        if not accum:
            log("No colour data accumulated; output will have no RGB.")
            df_color = pd.DataFrame(columns=['ki', 'kj', 'R', 'G', 'B'])
        else:
            log(
                f"\nAggregating {len(accum)} image frame(s) - "
                f"method = '{method}',  voxel_size = {voxel_size} m ..."
            )
            df_all = pd.concat(accum, ignore_index=True)

            if method == 'nearest':
                df_color = (
                    df_all.sort_values('dist_sq')
                    .groupby(['ki', 'kj'], sort=False)
                    .first()[['R', 'G', 'B']]
                    .reset_index()
                )

            elif method == 'mean':
                df_total = (
                    df_all.groupby(['ki', 'kj'])
                    .agg(
                        R_sum=('R_sum', 'sum'),
                        G_sum=('G_sum', 'sum'),
                        B_sum=('B_sum', 'sum'),
                        count=('count', 'sum'),
                    )
                    .reset_index()
                )
                df_color = df_total[['ki', 'kj']].copy()
                df_color['R'] = (df_total['R_sum'] / df_total['count']).round().astype(np.int32)
                df_color['G'] = (df_total['G_sum'] / df_total['count']).round().astype(np.int32)
                df_color['B'] = (df_total['B_sum'] / df_total['count']).round().astype(np.int32)

            elif method == 'std':
                df_total = (
                    df_all.groupby(['ki', 'kj'])
                    .agg(
                        R_sum=('R_sum', 'sum'), G_sum=('G_sum', 'sum'), B_sum=('B_sum', 'sum'),
                        R_sq=('R_sq', 'sum'), G_sq=('G_sq', 'sum'), B_sq=('B_sq', 'sum'),
                        count=('count', 'sum'),
                    )
                    .reset_index()
                )
                df_color = df_total[['ki', 'kj']].copy()
                with np.errstate(invalid='ignore', divide='ignore'):
                    r_mean = df_total['R_sum'] / df_total['count']
                    g_mean = df_total['G_sum'] / df_total['count']
                    b_mean = df_total['B_sum'] / df_total['count']
                    r_var = df_total['R_sq'] / df_total['count'] - (r_mean ** 2)
                    g_var = df_total['G_sq'] / df_total['count'] - (g_mean ** 2)
                    b_var = df_total['B_sq'] / df_total['count'] - (b_mean ** 2)
                    r_std = np.sqrt(np.clip(r_var.values, 0.0, None))
                    g_std = np.sqrt(np.clip(g_var.values, 0.0, None))
                    b_std = np.sqrt(np.clip(b_var.values, 0.0, None))
                df_color['R'] = np.round(r_std).astype(np.int32)
                df_color['G'] = np.round(g_std).astype(np.int32)
                df_color['B'] = np.round(b_std).astype(np.int32)

            else:  # median
                df_color = (
                    df_all.groupby(['ki', 'kj'])[['R', 'G', 'B']]
                    .median()
                    .round()
                    .astype(np.int32)
                    .reset_index()
                )

        log(f"  {len(df_color):,} unique coloured voxels")

        # ------------------------------------------------------------------
        # 5. Assign voxel colours to LAZ points (vectorised pandas merge)
        # ------------------------------------------------------------------
        log(f"Assigning colours to {n_pts:,} LAZ points ...")

        ki_las = np.floor(las_e / voxel_size).astype(np.int64)
        kj_las = np.floor(las_n / voxel_size).astype(np.int64)

        df_las_keys = pd.DataFrame({'ki': ki_las, 'kj': kj_las})
        df_merged   = df_las_keys.merge(df_color, on=['ki', 'kj'], how='left')

        has_color = df_merged['R'].notna().values
        stats['colored_points'] = int(has_color.sum())
        pct = 100.0 * stats['colored_points'] / max(n_pts, 1)
        log(f"  Coloured {stats['colored_points']:,} / {n_pts:,} points  ({pct:.1f} %)")

        # Scale uint8 [0, 255] -> uint16 [0, 65535] as required by LAS point_format 3
        def _to_u16(col: str) -> np.ndarray:
            return (df_merged[col].fillna(0).values.astype(np.uint16) << 8)

        red_u16   = np.where(has_color, _to_u16('R'), np.uint16(0)).astype(np.uint16)
        green_u16 = np.where(has_color, _to_u16('G'), np.uint16(0)).astype(np.uint16)
        blue_u16  = np.where(has_color, _to_u16('B'), np.uint16(0)).astype(np.uint16)

    # ------------------------------------------------------------------
    # 6. Write output LAZ
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    hdr         = laspy.LasHeader(version="1.2", point_format=3)
    hdr.scales  = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.array([raw_x.min(), raw_y.min(), raw_z.min()])

    las_out         = laspy.LasData(header=hdr)
    las_out.x       = raw_x
    las_out.y       = raw_y
    las_out.z       = raw_z
    las_out.red     = red_u16
    las_out.green   = green_u16
    las_out.blue    = blue_u16

    # Preserve intensity if the source file carries it
    if hasattr(las_in, 'intensity'):
        try:
            las_out.intensity = np.array(las_in.intensity, dtype=np.uint16)
        except Exception:
            pass

    las_out.write(output_path)
    log(f"\nOutput written: {output_path}")
    log(f"Stats: {stats}")
    return stats




def colorize_laz_streaming(
    laz_input_path: str,
    image_list_csv: str,
    image_dir: str,
    output_path: str,
    voxel_size: float = 0.01,
    rgb_method: str = 'nearest',
    time_window: float = 10.0,
    chunk_step: float = 5.0,
    lever_arm_x: float = 0.1044,
    lever_arm_y: float = 0.6246,
    lever_arm_z: float = 0.0826,
    pitch_offset: float = 0.010,
    roll_offset: float = 0.010,
    heading_offset: float = 0.000,
    downsample: int = 3,
    lls_list_csv: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Memory-efficient LAZ colourisation using a sliding time window.

    Instead of projecting all images and holding their pixel data in memory
    simultaneously, LAZ points are processed in chronological order in chunks
    of *chunk_step* seconds.  For each chunk only images whose timestamp falls
    within ±*time_window* seconds of that chunk are loaded, projected, and
    then immediately discarded.  Peak simultaneous memory is bounded to
    roughly ``2 * time_window * (images / total_duration)`` images at once —
    typically 20–30 images rather than 600.

    Requirements
    ------------
    - The input LAZ must carry a ``time_us`` field (Unix microseconds).
    - The image CSV must contain a ``Date_Time`` column parseable by
      ``pd.to_datetime``.

    Parameters
    ----------
    laz_input_path : str
        Path to the input LAZ/LAS file.
    image_list_csv : str
        CSV with columns: file_name, Date_Time, AUV_Pitch, AUV_Roll,
        AUV_Heading, AUV_Easting, AUV_Northing, AUV_Depth, AUV_Altitude.
    image_dir : str
        Directory containing the image files.
    output_path : str
        Path for the coloured output LAZ file.
    voxel_size : float
        2-D voxel cell side length in metres.  Default 0.01 m.
    rgb_method : str
        'nearest', 'mean', or 'median'.
    time_window : float
        Half-width of the image search window around each chunk (seconds).
        Default 10 s.
    chunk_step : float
        Duration of each LAZ processing chunk (seconds).  Default 5 s.
    lever_arm_x/y/z : float
        IMU-to-camera body-frame offsets in metres.
    pitch_offset, roll_offset, heading_offset : float
        Angular correction offsets in degrees.
    downsample : int
        Image downsampling factor.  Default 3.
    lls_list_csv : str or None
        Not used for time alignment (time_us is already Unix).  Retained for
        API compatibility with the non-streaming function.
    progress_callback : callable or None
        Optional single-str logging function.

    Returns
    -------
    dict
        Keys: total_images, processed_images, colored_points, total_points,
        chunks_processed.
    """

    def log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    method = rgb_method.lower().strip()
    if method not in ('nearest', 'mean', 'median', 'std'):
        raise ValueError(
            f"rgb_method must be one of: 'nearest', 'mean', 'median', 'std'; got '{rgb_method}'"
        )

    try:
        import laspy
    except ImportError:
        raise ImportError(
            "laspy is required.  Install with:  pip install laspy[lazrs]"
        )

    # ------------------------------------------------------------------
    # 1. Read LAZ + time_us (Unix microseconds)
    # ------------------------------------------------------------------
    log(f"Reading: {os.path.basename(laz_input_path)}")
    las_in = laspy.read(laz_input_path)

    raw_x = np.array(las_in.x, dtype=np.float64)
    raw_y = np.array(las_in.y, dtype=np.float64)
    raw_z = np.array(las_in.z, dtype=np.float64)
    las_e = raw_x   # Easting (project convention: x=E, y=N)
    las_n = raw_y   # Northing
    n_pts = len(las_e)

    # time_us is a custom extra-byte dimension storing Unix time in microseconds
    try:
        _test_time = las_in['time_us']
    except Exception:
        raise ValueError(
            "LAZ file does not contain a 'time_us' field.  "
            "Expected a custom extra-byte dimension with Unix time in microseconds."
        )
    las_unix = np.array(las_in['time_us'], dtype=np.float64) / 1e6  # → Unix seconds
    log(f"  {n_pts:,} points  |  time_us range "
        f"[{las_unix.min():.3f}, {las_unix.max():.3f}] Unix s")

    # ------------------------------------------------------------------
    # 3. Read and sort image list by Unix time
    # ------------------------------------------------------------------
    df_imgs = pd.read_csv(image_list_csv)
    if 'Date_Time' not in df_imgs.columns:
        raise ValueError(
            "Image CSV must contain a 'Date_Time' column for streaming mode."
        )

    img_dt = pd.to_datetime(df_imgs['Date_Time'])
    if img_dt.dt.tz is not None:
        img_dt = img_dt.dt.tz_localize(None)
    img_unix_s = img_dt.apply(lambda dt: dt.timestamp()).values  # float Unix s

    sort_img   = np.argsort(img_unix_s)
    df_imgs    = df_imgs.iloc[sort_img].reset_index(drop=True)
    img_unix_s = img_unix_s[sort_img]

    log(
        f"  {len(df_imgs)} images  "
        f"[{pd.Timestamp(img_unix_s[0], unit='s')} – "
        f"{pd.Timestamp(img_unix_s[-1], unit='s')}]"
    )

    # ------------------------------------------------------------------
    # 4. Sort LAZ points by time; allocate output colour arrays
    # ------------------------------------------------------------------
    sort_pts    = np.argsort(las_unix)
    sorted_unix = las_unix[sort_pts]
    sorted_e    = las_e[sort_pts]
    sorted_n    = las_n[sort_pts]

    R_out = np.zeros(n_pts, dtype=np.uint16)
    G_out = np.zeros(n_pts, dtype=np.uint16)
    B_out = np.zeros(n_pts, dtype=np.uint16)

    stats = {
        'total_images':     len(df_imgs),
        'processed_images': 0,
        'colored_points':   0,
        'total_points':     n_pts,
        'chunks_processed': 0,
    }

    # ------------------------------------------------------------------
    # 5. Sliding-window chunk loop
    # ------------------------------------------------------------------
    t_lo_global = sorted_unix[0]
    t_hi_global = sorted_unix[-1]
    chunk_edges = np.arange(t_lo_global, t_hi_global + chunk_step, chunk_step)
    n_chunks    = len(chunk_edges)

    log(
        f"\nProcessing {n_chunks} time chunks  "
        f"(chunk_step={chunk_step} s, time_window=±{time_window} s) ..."
    )

    processed_img_set: set = set()

    for ci, chunk_lo in enumerate(chunk_edges):
        chunk_hi = chunk_lo + chunk_step

        # Points in this time chunk (binary search for speed)
        pt_lo = int(np.searchsorted(sorted_unix, chunk_lo, side='left'))
        pt_hi = int(np.searchsorted(sorted_unix, chunk_hi, side='left'))
        if pt_lo >= pt_hi:
            continue

        chunk_e  = sorted_e[pt_lo:pt_hi]
        chunk_n  = sorted_n[pt_lo:pt_hi]
        orig_pts = sort_pts[pt_lo:pt_hi]   # original LAZ indices for this chunk

        # Spatial bounding box of this chunk (for pixel pre-filtering)
        e_lo_c = chunk_e.min() - voxel_size
        e_hi_c = chunk_e.max() + voxel_size
        n_lo_c = chunk_n.min() - voxel_size
        n_hi_c = chunk_n.max() + voxel_size

        # Images within the time window for this chunk
        win_lo = chunk_lo - time_window
        win_hi = chunk_hi + time_window
        img_lo = int(np.searchsorted(img_unix_s, win_lo, side='left'))
        img_hi = int(np.searchsorted(img_unix_s, win_hi, side='right'))
        chunk_img_df = df_imgs.iloc[img_lo:img_hi]

        if len(chunk_img_df) == 0:
            continue

        log(
            f"  [{ci + 1}/{n_chunks}]  "
            f"{pt_hi - pt_lo:,} pts  |  {len(chunk_img_df)} imgs in window"
        )

        # Accumulate voxel colour data from images in this window
        accum_chunk: list = []

        for _, img_row in chunk_img_df.iterrows():
            fname = str(img_row['file_name'])
            fpath = os.path.join(image_dir, fname)
            if not os.path.exists(fpath):
                continue

            try:
                altitude = float(img_row['AUV_Altitude'])
                pitch    = -float(img_row['AUV_Pitch'])
                roll     = float(img_row['AUV_Roll'])
                heading  = float(img_row['AUV_Heading'])
                imu_e    = float(img_row['AUV_Easting'])
                imu_n    = float(img_row['AUV_Northing'])
                imu_u    = float(img_row['AUV_Depth'])

                cam_pos, shift = imu_to_camera_enu(
                    imu_e, imu_n, imu_u,
                    lever_arm_x, lever_arm_y, lever_arm_z,
                    pitch, roll, heading,
                )

                px_e, px_n = pixels_to_world_coordinates(
                    distance_off_bottom=altitude + shift[2],
                    pitch=pitch + pitch_offset,
                    roll=roll + roll_offset,
                    heading=heading + heading_offset,
                    image_path=fpath,
                    camera_east=cam_pos[0],
                    camera_north=cam_pos[1],
                    downsample=downsample,
                )

                img_bgr = cv2.imread(fpath)
                if img_bgr is None:
                    continue
                if downsample > 1:
                    img_bgr = cv2.resize(
                        img_bgr,
                        (img_bgr.shape[1] // downsample, img_bgr.shape[0] // downsample),
                        interpolation=cv2.INTER_AREA,
                    )
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                e_flat   = px_e.flatten()
                n_flat   = px_n.flatten()
                rgb_flat = img_rgb.reshape(-1, 3)

                mask = (
                    ~np.isnan(e_flat) & ~np.isnan(n_flat)
                    & (e_flat >= e_lo_c) & (e_flat <= e_hi_c)
                    & (n_flat >= n_lo_c) & (n_flat <= n_hi_c)
                )
                if not mask.any():
                    continue

                ev = e_flat[mask]
                nv = n_flat[mask]
                rv = rgb_flat[mask].astype(np.int32)

                ki = np.floor(ev / voxel_size).astype(np.int64)
                kj = np.floor(nv / voxel_size).astype(np.int64)

                if method == 'nearest':
                    vc_e    = (ki + 0.5) * voxel_size
                    vc_n    = (kj + 0.5) * voxel_size
                    dist_sq = (ev - vc_e) ** 2 + (nv - vc_n) ** 2
                    df_pix  = pd.DataFrame({
                        'ki': ki, 'kj': kj, 'dist_sq': dist_sq,
                        'R': rv[:, 0], 'G': rv[:, 1], 'B': rv[:, 2],
                    })
                    df_agg = (
                        df_pix.sort_values('dist_sq')
                        .groupby(['ki', 'kj'], sort=False).first()
                        .reset_index()[['ki', 'kj', 'dist_sq', 'R', 'G', 'B']]
                    )

                elif method in ('mean', 'std'):
                    if method == 'mean':
                        df_pix = pd.DataFrame({
                            'ki': ki, 'kj': kj,
                            'R_sum': rv[:, 0].astype(np.float64),
                            'G_sum': rv[:, 1].astype(np.float64),
                            'B_sum': rv[:, 2].astype(np.float64),
                            'count': np.ones(len(rv), dtype=np.int64),
                        })
                        df_agg = (
                            df_pix.groupby(['ki', 'kj'])
                            .agg(
                                R_sum=('R_sum', 'sum'), G_sum=('G_sum', 'sum'),
                                B_sum=('B_sum', 'sum'), count=('count', 'sum'),
                            )
                            .reset_index()
                        )
                    else:  # std
                        df_pix = pd.DataFrame({
                            'ki': ki, 'kj': kj,
                            'R_sum': rv[:, 0].astype(np.float64),
                            'G_sum': rv[:, 1].astype(np.float64),
                            'B_sum': rv[:, 2].astype(np.float64),
                            'R_sq': (rv[:, 0].astype(np.float64) ** 2),
                            'G_sq': (rv[:, 1].astype(np.float64) ** 2),
                            'B_sq': (rv[:, 2].astype(np.float64) ** 2),
                            'count': np.ones(len(rv), dtype=np.int64),
                        })
                        df_agg = (
                            df_pix.groupby(['ki', 'kj'])
                            .agg(
                                R_sum=('R_sum', 'sum'), G_sum=('G_sum', 'sum'), B_sum=('B_sum', 'sum'),
                                R_sq=('R_sq', 'sum'), G_sq=('G_sq', 'sum'), B_sq=('B_sq', 'sum'),
                                count=('count', 'sum'),
                            )
                            .reset_index()
                        )

                else:  # median – pre-aggregate per voxel within this image
                    df_pix = pd.DataFrame({
                        'ki': ki, 'kj': kj,
                        'R': rv[:, 0], 'G': rv[:, 1], 'B': rv[:, 2],
                    })
                    df_agg = (
                        df_pix.groupby(['ki', 'kj'])[['R', 'G', 'B']]
                        .median().round().astype(np.int32).reset_index()
                    )

                accum_chunk.append(df_agg)
                processed_img_set.add(fname)

            except Exception as exc:
                log(f"    Error processing {fname}: {exc}")
                continue

        if not accum_chunk:
            continue

        # Merge voxel maps across the images in this chunk's window
        df_all = pd.concat(accum_chunk, ignore_index=True)

        if method == 'nearest':
            df_color = (
                df_all.sort_values('dist_sq')
                .groupby(['ki', 'kj'], sort=False).first()[['R', 'G', 'B']]
                .reset_index()
            )

        elif method == 'mean':
            df_total = (
                df_all.groupby(['ki', 'kj'])
                .agg(
                    R_sum=('R_sum', 'sum'), G_sum=('G_sum', 'sum'),
                    B_sum=('B_sum', 'sum'), count=('count', 'sum'),
                )
                .reset_index()
            )
            df_color = df_total[['ki', 'kj']].copy()
            df_color['R'] = (df_total['R_sum'] / df_total['count']).round().astype(np.int32)
            df_color['G'] = (df_total['G_sum'] / df_total['count']).round().astype(np.int32)
            df_color['B'] = (df_total['B_sum'] / df_total['count']).round().astype(np.int32)

        elif method == 'std':
            df_total = (
                df_all.groupby(['ki', 'kj'])
                .agg(
                    R_sum=('R_sum', 'sum'), G_sum=('G_sum', 'sum'), B_sum=('B_sum', 'sum'),
                    R_sq=('R_sq', 'sum'), G_sq=('G_sq', 'sum'), B_sq=('B_sq', 'sum'),
                    count=('count', 'sum'),
                )
                .reset_index()
            )
            df_color = df_total[['ki', 'kj']].copy()
            with np.errstate(invalid='ignore', divide='ignore'):
                r_mean = df_total['R_sum'] / df_total['count']
                g_mean = df_total['G_sum'] / df_total['count']
                b_mean = df_total['B_sum'] / df_total['count']
                r_var = df_total['R_sq'] / df_total['count'] - (r_mean ** 2)
                g_var = df_total['G_sq'] / df_total['count'] - (g_mean ** 2)
                b_var = df_total['B_sq'] / df_total['count'] - (b_mean ** 2)
                r_std = np.sqrt(np.clip(r_var.values, 0.0, None))
                g_std = np.sqrt(np.clip(g_var.values, 0.0, None))
                b_std = np.sqrt(np.clip(b_var.values, 0.0, None))
            df_color['R'] = np.round(r_std).astype(np.int32)
            df_color['G'] = np.round(g_std).astype(np.int32)
            df_color['B'] = np.round(b_std).astype(np.int32)

        else:  # median
            df_color = (
                df_all.groupby(['ki', 'kj'])[['R', 'G', 'B']]
                .median().round().astype(np.int32).reset_index()
            )

        # Assign voxel colours to this chunk's LAZ points
        ki_las    = np.floor(chunk_e / voxel_size).astype(np.int64)
        kj_las    = np.floor(chunk_n / voxel_size).astype(np.int64)
        df_keys   = pd.DataFrame({'ki': ki_las, 'kj': kj_las})
        df_merged = df_keys.merge(df_color, on=['ki', 'kj'], how='left')

        has_col = df_merged['R'].notna().values

        def _u16(col: str) -> np.ndarray:
            return (df_merged[col].fillna(0).values.astype(np.uint16) << 8)

        R_out[orig_pts] = np.where(has_col, _u16('R'), np.uint16(0)).astype(np.uint16)
        G_out[orig_pts] = np.where(has_col, _u16('G'), np.uint16(0)).astype(np.uint16)
        B_out[orig_pts] = np.where(has_col, _u16('B'), np.uint16(0)).astype(np.uint16)

        stats['colored_points'] += int(has_col.sum())
        stats['chunks_processed'] += 1

    stats['processed_images'] = len(processed_img_set)
    pct = 100.0 * stats['colored_points'] / max(n_pts, 1)
    log(
        f"\nTotal: {stats['colored_points']:,} / {n_pts:,} points coloured "
        f"({pct:.1f} %)  |  {stats['processed_images']} unique images projected."
    )

    # ------------------------------------------------------------------
    # 6. Write output LAZ (preserving original point order)
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    hdr         = laspy.LasHeader(version="1.2", point_format=3)
    hdr.scales  = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.array([raw_x.min(), raw_y.min(), raw_z.min()])

    las_out       = laspy.LasData(header=hdr)
    las_out.x     = raw_x
    las_out.y     = raw_y
    las_out.z     = raw_z
    las_out.red   = R_out
    las_out.green = G_out
    las_out.blue  = B_out

    if hasattr(las_in, 'intensity'):
        try:
            las_out.intensity = np.array(las_in.intensity, dtype=np.uint16)
        except Exception:
            pass

    las_out.write(output_path)
    log(f"Output written: {output_path}")
    log(f"Stats: {stats}")
    return stats
