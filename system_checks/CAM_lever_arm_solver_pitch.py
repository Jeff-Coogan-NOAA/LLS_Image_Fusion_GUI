"""
image_lever_arm_solver.py
=========================
Solve for camera lever-arm X and Z offsets by interactively picking two known 
target points across multiple images that span a range of AUV roll angles.

Principle
---------
As the AUV rolls, a fixed target on the sea floor will appear at different pixel
coordinates (u, v) in the images. With the correct IMU-to-Camera lever arm 
[X_OFF, Y_OFF, Z_OFF], the calculated real-world Easting and Northing of those 
pixels must map to the exact same world coordinate regardless of roll.

The solver minimizes the total world-coordinate variance for both corners 
simultaneously using the georeferencing functions from `utils`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import optimize


# =============================================================================
# RUN CONFIGURATION  –  Edit these variables before running
# =============================================================================

# ┌─ Solved Camera Lever Arms ────────┐
# │  X_OFFSET = +0.125813 m           │
# │  Z_OFFSET = -0.213513 m           │
# └───────────────────────────────────┘
# │  Y_OFFSET = +0.945584 m           │
# └───────────────────────────────────┘

IMAGE_LIST_CSV = r"E:\VOYIS_System_Checks\Tank Test\Test01_PRC\processing\image\image_file_list.csv"
IMAGE_DIR = r"E:\VOYIS_System_Checks\Tank Test\Test01_RAW\image_raw\2026-05-29_00-40-46"

# Frame skipping / sorting
FRAME_INTERVAL = 1          # Present every N-th image. Increase if too many images.
SORT_FRAMES_BY_PITCH = True  # Sort by pitch to ensure diverse pitch picking

# ── Lever Arm Setup ──────────────────────────────────────────────────────────
# Initial (fixed) X and Z lever arm offsets in meters
X_OFFSET_INIT = +0.125813  # meters
Z_OFFSET_INIT = -0.213513  # meters

# Y offset (along-track) will be solved here while X and Z remain fixed.
LEVER_ARM_Y_FIXED = 0.6246

# Existing angular offsets (degrees)
PITCH_OFFSET = 0.0
ROLL_OFFSET = 0.0
HEADING_OFFSET = 0.000

# ── Click persistence ────────────────────────────────────────────────────────
CLICKS_SAVE_PATH = "camera_lever_arm_clicks.json"
REUSE_SAVED_CLICKS = False

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_PATH = "camera_lever_arm_solver_results.png"

# =============================================================================
# End of configuration
# =============================================================================

import numpy as np


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

def single_pixel_to_world_coordinates(
    u: float, 
    v: float, 
    distance_off_bottom: float, 
    pitch: float, 
    roll: float, 
    heading: float, 
    camera_east: float, 
    camera_north: float,
    image_shape: tuple = (3008, 4096), # Default to original height, width
    focal_length_px: float = 3801.37053,
    k1: float = 0.0113579,
    k2: float = -0.0143928,
    p1: float = 0.0042688,
    p2: float = -0.000244194
) -> tuple[float, float]:
    """
    Projects a single pixel to the seafloor plane using the original 
    working rotation and distortion logic.
    """
    # 1. Pitch correction for downward looking camera (Restored!)
    pitch = -90 + pitch
    
    # 2. Convert angles to radians
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll) 
    heading_rad = np.radians(heading)
    
    # 3. Camera intrinsics
    img_h, img_w = image_shape[:2]
    cx = img_w / 2.0
    cy = img_h / 2.0
    fx = fy = focal_length_px
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy], 
        [0, 0, 1]
    ])
    dist_coeffs = np.array([k1, k2, p1, p2, 0])
    
    # 4. Undistort the single pixel (Crucial for edge/corner clicks)
    pixel_coords = np.array([[[u, v]]], dtype=np.float32)
    undistorted_coords = cv2.undistortPoints(
        pixel_coords, camera_matrix, dist_coeffs, P=camera_matrix
    )
    u_undist, v_undist = undistorted_coords[0, 0]
    
    # 5. Convert to normalized coordinates
    x_norm = (u_undist - cx) / fx
    y_norm = (v_undist - cy) / fy
    
    # 6. Create ray in camera coordinate system (Restored original mapping!)
    # X=right, Y=forward, Z=up
    ray_camera = np.array([x_norm, 1.0, -y_norm])
    
    # 7. Rotation matrices (Restored original logic)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    R_roll = np.array([
        [np.cos(roll_rad), 0, np.sin(roll_rad)],
        [0, 1, 0],
        [-np.sin(roll_rad), 0, np.cos(roll_rad)]
    ])
    
    R_heading = np.array([
        [np.cos(heading_rad), np.sin(heading_rad), 0],
        [-np.sin(heading_rad), np.cos(heading_rad), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: pitch -> roll -> heading
    R_camera_to_world = R_heading @ R_roll @ R_pitch
    
    # 8. Transform ray to world coordinate system
    ray_world = R_camera_to_world @ ray_camera
    ray_x, ray_y, ray_z = ray_world
    
    # 9. Intersection with seafloor plane (Z = 0 relative to distance_off_bottom)
    # Avoid division by zero and reject rays pointing up
    if abs(ray_z) < 1e-10 or ray_z >= 0:
        return np.nan, np.nan
        
    t = -distance_off_bottom / ray_z
    
    # 10. Calculate final world Easting and Northing
    world_easting = camera_east + (t * ray_x)
    world_northing = camera_north + (t * ray_y)
    
    return float(world_easting), float(world_northing)


def apply_camera_transform(
    click: dict,
    x_off: float,
    z_off: float,
    y_off: float,
    pitch_off: float,
    roll_off: float,
    heading_off: float
) -> tuple[float, float]:
    """
    Calculates the world Easting and Northing for a SINGLE clicked pixel, 
    vastly speeding up the optimization loop.
    """
    # 1. IMU to Camera offset
    pitch = -click["pitch"]
    roll = click["roll"]
    heading = click["heading"]
    alt = click["altitude"]
    
    camera_pos, shift = imu_to_camera_enu(
        click["easting"], click["northing"], click["depth"],
        x_off, y_off, z_off,
        pitch, roll, heading
    )
    print(f"Camera easting: {camera_pos[0]}, northing: {camera_pos[1]}, depth: {camera_pos[2]}, Shift: {shift}")
    # Extract the exact pixel coordinates
    u, v = float(click["u"]), float(click["v"])
    
    # 2. Single Pixel to World
    # Note: We now call a targeted function instead of the full meshgrid generator
    world_easting, world_northing = single_pixel_to_world_coordinates(
        u=u, 
        v=v,
        distance_off_bottom=alt + shift[2],
        pitch=pitch + pitch_off,
        roll=roll + roll_off,
        heading=heading + heading_off,
        camera_east=camera_pos[0],
        camera_north=camera_pos[1],
        # If the saved click does not include an image shape (e.g., older JSON),
        # fall back to a reasonable default to avoid None being passed.
        image_shape=click.get("image_shape") or (3008, 4096)
    )
    
    return world_easting, world_northing


def _world_positions(params: np.ndarray, clicks: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Compute world Easting, Northing for a set of clicks given [y_off].

    X and Z lever arms are taken from the fixed module-level values
    `X_OFFSET_INIT` and `Z_OFFSET_INIT`.
    """
    y_off = float(params[0])
    x_off = float(X_OFFSET_INIT)
    z_off = float(Z_OFFSET_INIT)

    E_list, N_list = [], []
    for c in clicks:
        E, N = apply_camera_transform(c, x_off, z_off, y_off, PITCH_OFFSET, ROLL_OFFSET, HEADING_OFFSET)
        E_list.append(E)
        N_list.append(N)

    return np.array(E_list), np.array(N_list)


def _cost(params: np.ndarray, clicks_a: list[dict], clicks_b: list[dict]) -> float:
    """Total world-coordinate variance for both corners given trial Y lever arm."""
    total = 0.0
    for clicks in (clicks_a, clicks_b):
        if len(clicks) < 2:
            continue
        E, N = _world_positions(params, clicks)

        # Avoid NaN spreading if some pixels fell off valid footprint
        valid = ~(np.isnan(E) | np.isnan(N))
        if np.any(valid):
            total += float(np.var(E[valid]) + np.var(N[valid]))
    return total

# ─────────────────────────────────────────────────────────────────────────────
# Interactive click collection (Image Based)
# ─────────────────────────────────────────────────────────────────────────────

def collect_image_clicks_interactive(
    df: pd.DataFrame,
    img_dir: str,
    interval: int = 1,
    sort_by_pitch: bool = True
) -> tuple[list[dict], list[dict]]:
    
    from matplotlib.widgets import Button
    
    if sort_by_pitch:
        df = df.sort_values("pitch").reset_index(drop=True)
        
    candidates = df.iloc[::interval].to_dict('records')
    n_frames = len(candidates)
    
    if n_frames == 0:
        print("Error: No images found matching the criteria.")
        return [], []

    S: dict = {
        "idx": 0, "mode": "A", "clicks_a": [], "clicks_b": [],
        "done_a": False, "done_b": False, "current_img_shape": (0,0)
    }

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_axes([0.05, 0.17, 0.90, 0.77])

    _bax = {
        "prev5": fig.add_axes([0.03, 0.03, 0.09, 0.07]),
        "prev":  fig.add_axes([0.13, 0.03, 0.07, 0.07]),
        "next":  fig.add_axes([0.21, 0.03, 0.07, 0.07]),
        "next5": fig.add_axes([0.29, 0.03, 0.09, 0.07]),
        "modeA": fig.add_axes([0.44, 0.03, 0.10, 0.07]),
        "modeB": fig.add_axes([0.55, 0.03, 0.10, 0.07]),
        "doneA": fig.add_axes([0.74, 0.03, 0.10, 0.07]),
        "doneB": fig.add_axes([0.85, 0.03, 0.10, 0.07]),
    }
    _btns = {
        "prev5": Button(_bax["prev5"], "\xab Prev 5"),
        "prev":  Button(_bax["prev"],  "\u2039 Prev"),
        "next":  Button(_bax["next"],  "Next \u203a"),
        "next5": Button(_bax["next5"], "Next 5 \xbb"),
        "modeA": Button(_bax["modeA"], "Point A", color="steelblue", hovercolor="royalblue"),
        "modeB": Button(_bax["modeB"], "Point B", color="0.85", hovercolor="0.95"),
        "doneA": Button(_bax["doneA"], "Done A", color="lightcoral", hovercolor="salmon"),
        "doneB": Button(_bax["doneB"], "Done B", color="moccasin", hovercolor="navajowhite"),
    }
    _btns["modeA"].label.set_color("white")

    def _refresh_btns():
        if S["mode"] == "A" and not S["done_a"]:
            _btns["modeA"].ax.set_facecolor("steelblue")
            _btns["modeA"].label.set_color("white")
            _btns["modeB"].ax.set_facecolor("0.85")
            _btns["modeB"].label.set_color("black")
        else:
            _btns["modeA"].ax.set_facecolor("0.85")
            _btns["modeA"].label.set_color("black")
            _btns["modeB"].ax.set_facecolor("darkorange")
            _btns["modeB"].label.set_color("white")
        if S["done_a"]:
            _btns["doneA"].ax.set_facecolor("lightgreen")
            _btns["doneA"].label.set_text("Done A \u2713")
        if S["done_b"]:
            _btns["doneB"].ax.set_facecolor("lightgreen")
            _btns["doneB"].label.set_text("Done B \u2713")

    def _draw():
        ax.clear()
        row = candidates[S["idx"]]
        fname = row["filename"]
        pitch = row["pitch"]
        
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            S["current_img_shape"] = img.shape
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Image not found:\n{fname}", ha='center', va='center', transform=ax.transAxes)

        # Draw clicks
        for c in S["clicks_a"]:
            if c["filename"] == fname:
                ax.plot(c["u"], c["v"], "*", c="royalblue", markersize=14, markeredgecolor="navy")
        for c in S["clicks_b"]:
            if c["filename"] == fname:
                ax.plot(c["u"], c["v"], "*", c="darkorange", markersize=14, markeredgecolor="sienna")

        ax.set_title(
            f"Image {S['idx'] + 1}/{n_frames}  |  File: {fname}  |  Pitch: {pitch:+.2f}\u00b0\n"
            f"\u2605 A = {len(S['clicks_a'])}   \u2605 B = {len(S['clicks_b'])}"
        )
        ax.axis('off')
        _refresh_btns()
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes is not ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        row = candidates[S["idx"]]
        # Save exact dict copy + uv coords
        rec = dict(row)
        rec["u"] = event.xdata
        rec["v"] = event.ydata
        # Store the current image shape so downstream code can use it.
        rec["image_shape"] = S["current_img_shape"]
        
        mode = S["mode"]
        if mode == "A" and not S["done_a"]:
            S["clicks_a"] = [c for c in S["clicks_a"] if c["filename"] != row["filename"]]
            S["clicks_a"].append(rec)
            print(f"  [A] File: {row['filename']} | Pitch={row['pitch']:+.2f}\u00b0 | u={rec['u']:.1f}, v={rec['v']:.1f}")
        elif mode == "B" and not S["done_b"]:
            S["clicks_b"] = [c for c in S["clicks_b"] if c["filename"] != row["filename"]]
            S["clicks_b"].append(rec)
            print(f"  [B] File: {row['filename']} | Pitch={row['pitch']:+.2f}\u00b0 | u={rec['u']:.1f}, v={rec['v']:.1f}")
        _draw()

    def _nav(delta):
        S["idx"] = max(0, min(n_frames - 1, S["idx"] + delta))
        _draw()

    def _set_mode(m):
        if (m == "A" and S["done_a"]) or (m == "B" and S["done_b"]): return
        S["mode"] = m
        _draw()

    def _done(pt):
        S[f"done_{pt.lower()}"] = True
        other = "B" if pt == "A" else "A"
        if not S[f"done_{other.lower()}"]:
            S["mode"] = other
        _draw()
        if S["done_a"] and S["done_b"]:
            plt.close(fig)

    _btns["prev5"].on_clicked(lambda _: _nav(-5))
    _btns["prev"].on_clicked(lambda _:  _nav(-1))
    _btns["next"].on_clicked(lambda _:  _nav(+1))
    _btns["next5"].on_clicked(lambda _: _nav(+5))
    _btns["modeA"].on_clicked(lambda _: _set_mode("A"))
    _btns["modeB"].on_clicked(lambda _: _set_mode("B"))
    _btns["doneA"].on_clicked(lambda _: _done("A"))
    _btns["doneB"].on_clicked(lambda _: _done("B"))
    fig.canvas.mpl_connect("button_press_event", _on_click)

    print(f"\nInteractive collection: {n_frames} images.")
    _draw()
    plt.show(block=True)
    
    return S["clicks_a"], S["clicks_b"]

# ─────────────────────────────────────────────────────────────────────────────
# Execution Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_solver():
    print(f"Loading Navigation CSV: {IMAGE_LIST_CSV}")
    df = pd.read_csv(IMAGE_LIST_CSV)
    
    # Check bounds
    print(f"Loaded {len(df)} images.")
    pitch_min, pitch_max = df['pitch'].min(), df['pitch'].max()
    print(f"Pitch range: {pitch_min:.2f}\u00b0 to {pitch_max:.2f}\u00b0")

    # Load/Collect Clicks
    if REUSE_SAVED_CLICKS and os.path.exists(CLICKS_SAVE_PATH):
        print(f"Loading clicks from {CLICKS_SAVE_PATH}")
        with open(CLICKS_SAVE_PATH, 'r') as f:
            data = json.load(f)
            clicks_a = data.get("point_a", [])
            clicks_b = data.get("point_b", [])
    else:
        clicks_a, clicks_b = collect_image_clicks_interactive(df, IMAGE_DIR, FRAME_INTERVAL, SORT_FRAMES_BY_PITCH)
        with open(CLICKS_SAVE_PATH, 'w') as f:
            json.dump({"point_a": clicks_a, "point_b": clicks_b}, f, indent=4)

    if len(clicks_a) < 2 and len(clicks_b) < 2:
        print("Not enough clicks to optimize. Exiting.")
        return

    print(f"\nStarting optimization (Fixed X={X_OFFSET_INIT:.4f}, Z={Z_OFFSET_INIT:.4f}) — solving for Y")
    print("Evaluating cost function... (This may take a minute due to image footprint mappings)")
    x0 = np.array([LEVER_ARM_Y_FIXED])
    cost_before = _cost(x0, clicks_a, clicks_b)
    print(f"Cost before: {cost_before:.6e} m²")

    # Optimize Y (X and Z are fixed)
    result = optimize.minimize(
        _cost,
        x0=x0,
        args=(clicks_a, clicks_b),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 1000}
    )

    y_off_opt = float(result.x[0])
    cost_after = result.fun

    print(f"\nOptimisation finished ({result.message})")
    print(f"  Cost after: {cost_after:.6e} m²")
    print(f"\n┌─ Solved Camera Lever Arm Y ────────────┐")
    print(f"│  Y_OFFSET = {y_off_opt:+.6f} m           │")
    print(f"└────────────────────────────────────────┘")

    # Final Scatter Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for label, clicks, marker in [('A', clicks_a, 'o'), ('B', clicks_b, '^')]:
        if not clicks: continue
        E_bef, N_bef = _world_positions(np.array([LEVER_ARM_Y_FIXED]), clicks)
        E_aft, N_aft = _world_positions(result.x, clicks)
        
        ax1.scatter(E_bef, N_bef, label=f"Point {label} Before", alpha=0.7, marker=marker)
        ax2.scatter(E_aft, N_aft, label=f"Point {label} After", alpha=0.9, marker=marker, s=80)

    ax1.set_title("World Coordinates - BEFORE")
    ax2.set_title("World Coordinates - AFTER")
    for ax in [ax1, ax2]:
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', 'datalim')

    plt.suptitle(f"Camera Y Lever Arm Solver (Pitch Test)\nSolved: Y={y_off_opt:+.4f}")
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {OUTPUT_PATH}")
    plt.show()

if __name__ == "__main__":
    run_solver()