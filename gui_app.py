"""
GUI Application for LLS and Image Fusion Processing
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import pandas as pd
import os
from typing import Optional
from pointcloud_processor import process_images_to_pointclouds
from geotiff_processor import process_images_to_geotiffs
from laz_colorizer import colorize_laz_from_images
from panoramic_strip_processor import create_panoramic_strips
from utils import haversine_distance


class LLSImageProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LLS & Image Fusion Processor")
        self.root.geometry("900x800")
        
        # Configuration variables (default paths)
        self.image_list_csv = tk.StringVar(value=r"")
        self.lls_list_csv = tk.StringVar(value=r"")
        self.lls_dir = tk.StringVar(value=r"")
        self.image_dir = tk.StringVar(value=r"")
        self.output_dir = tk.StringVar(value=r"")
        
        # Processing options
        self.create_geotiff = tk.BooleanVar(value=True)
        self.create_pointcloud = tk.BooleanVar(value=False)
        self.copy_original_images = tk.BooleanVar(value=False)
        
        # Image selection mode
        self.selection_mode = tk.StringVar(value="all")
        self.selected_images_file = tk.StringVar()
        self.center_lat = tk.DoubleVar(value=0.0)
        self.center_lon = tk.DoubleVar(value=0.0)
        self.radius_m = tk.DoubleVar(value=100.0)
        self.skip_interval = tk.IntVar(value=1)
        
        # Camera parameters
        self.lever_arm_x = tk.DoubleVar(value=0.1044)
        self.lever_arm_y = tk.DoubleVar(value=0.6246)
        self.lever_arm_z = tk.DoubleVar(value=0.0826)
        self.pitch_offset = tk.DoubleVar(value=0.010)
        self.roll_offset = tk.DoubleVar(value=0.010)
        self.heading_offset = tk.DoubleVar(value=0.000)
        
        # Point cloud parameters
        self.downsample = tk.IntVar(value=3)
        self.max_distance = tk.DoubleVar(value=0.03)
        self.pointcloud_format = tk.StringVar(value='LAZ')
        
        # LAZ Colorization parameters
        self.colorize_laz = tk.BooleanVar(value=False)
        self.voxel_size = tk.DoubleVar(value=0.01)
        self.voxel_rgb_method = tk.StringVar(value='median')
        
        # GeoTIFF parameters
        self.utm_zone = tk.IntVar(value=16)
        self.utm_hemisphere = tk.StringVar(value='N')
        self.dpi = tk.IntVar(value=500)

        # Panoramic strip parameters
        self.create_panoramic_strips = tk.BooleanVar(value=False)
        self.strip_input_mode = tk.StringVar(value='from_run')
        self.strip_geotiff_dir = tk.StringVar()
        self.images_per_strip = tk.IntVar(value=100)
        self.strip_resolution_m = tk.DoubleVar(value=0.005)
        self.strip_prefix = tk.StringVar(value='strip')

        # Processing state
        self.is_processing = False
        
        self.create_widgets()
    
    def create_widgets(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 1: Input/Output Configuration
        io_frame = ttk.Frame(notebook)
        notebook.add(io_frame, text="Input/Output")
        self.create_io_tab(io_frame)
        
        # Tab 2: Processing Options
        options_frame = ttk.Frame(notebook)
        notebook.add(options_frame, text="Processing Options")
        self.create_options_tab(options_frame)
        
        # Tab 3: Image Selection
        selection_frame = ttk.Frame(notebook)
        notebook.add(selection_frame, text="Image Selection")
        self.create_selection_tab(selection_frame)
        
        # Tab 4: Camera Parameters
        camera_frame = ttk.Frame(notebook)
        notebook.add(camera_frame, text="Camera Parameters")
        self.create_camera_tab(camera_frame)

        # Tab 5: Panoramic Strips
        strip_frame = ttk.Frame(notebook)
        notebook.add(strip_frame, text="Panoramic Strips")
        self.create_strip_tab(strip_frame)

        # Bottom: Progress and Control
        self.create_bottom_panel()
    
    def create_io_tab(self, parent):
        """Input/Output file selection"""
        frame = ttk.LabelFrame(parent, text="File Paths", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Image list CSV
        ttk.Label(frame, text="Image List CSV:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(frame, textvariable=self.image_list_csv, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.image_list_csv)).grid(row=0, column=2, pady=5)
        
        # LLS list CSV (only for point clouds)
        ttk.Label(frame, text="LLS List CSV:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(frame, textvariable=self.lls_list_csv, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_file(self.lls_list_csv)).grid(row=1, column=2, pady=5)
        ttk.Label(frame, text="(Required only for Point Cloud generation)", font=('Arial', 8, 'italic')).grid(row=1, column=3, sticky='w', padx=5)
        
        # LLS directory
        ttk.Label(frame, text="LLS Directory:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Entry(frame, textvariable=self.lls_dir, width=60).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.lls_dir)).grid(row=2, column=2, pady=5)
        ttk.Label(frame, text="(Required only for Point Cloud generation)", font=('Arial', 8, 'italic')).grid(row=2, column=3, sticky='w', padx=5)
        
        # Image directory
        ttk.Label(frame, text="Image Directory:").grid(row=3, column=0, sticky='w', pady=5)
        ttk.Entry(frame, textvariable=self.image_dir, width=60).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.image_dir)).grid(row=3, column=2, pady=5)
        
        # Output directory
        ttk.Label(frame, text="Output Directory:").grid(row=4, column=0, sticky='w', pady=5)
        ttk.Entry(frame, textvariable=self.output_dir, width=60).grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.browse_directory(self.output_dir)).grid(row=4, column=2, pady=5)
        
        # Note: LAZ files for colorization are read from the LLS Directory
        ttk.Label(frame, text="(LAZ colorization reads .laz/.las files from the LLS Directory)",
              font=('Arial', 8, 'italic')).grid(row=5, column=1, sticky='w', padx=5)
    
    def create_options_tab(self, parent):
        """Processing options"""
        frame = ttk.LabelFrame(parent, text="Output Types", padding=10)
        frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Checkbutton(frame, text="Create GeoTIFF", variable=self.create_geotiff).pack(anchor='w', pady=5)
        ttk.Checkbutton(frame, text="Create Point Cloud", variable=self.create_pointcloud).pack(anchor='w', pady=5)
        ttk.Checkbutton(frame, text="Colorize LAZ Point Cloud (voxel RGB mapping)", variable=self.colorize_laz).pack(anchor='w', pady=5)
        ttk.Checkbutton(frame, text="Create Panoramic Strips (mosaic GeoTIFFs into long strips)", variable=self.create_panoramic_strips).pack(anchor='w', pady=5)
        ttk.Checkbutton(frame, text="Copy Original Images (for radius-based search)", variable=self.copy_original_images).pack(anchor='w', pady=5)
        
        # Point cloud specific parameters
        pc_frame = ttk.LabelFrame(parent, text="Point Cloud Parameters", padding=10)
        pc_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(pc_frame, text="Downsample Factor:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Spinbox(pc_frame, from_=1, to=10, textvariable=self.downsample, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(pc_frame, text="Max Distance (m):").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(pc_frame, textvariable=self.max_distance, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(pc_frame, text="Output Format:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Combobox(pc_frame, textvariable=self.pointcloud_format, values=['LAZ', 'XYZ'], width=10, state='readonly').grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # GeoTIFF specific parameters
        gt_frame = ttk.LabelFrame(parent, text="GeoTIFF Parameters", padding=10)
        gt_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(gt_frame, text="UTM Zone:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Spinbox(gt_frame, from_=1, to=60, textvariable=self.utm_zone, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(gt_frame, text="Hemisphere:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Combobox(gt_frame, textvariable=self.utm_hemisphere, values=['N', 'S'], width=8, state='readonly').grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(gt_frame, text="DPI:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Spinbox(gt_frame, from_=100, to=1000, increment=100, textvariable=self.dpi, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # LAZ Colorization specific parameters
        laz_frame = ttk.LabelFrame(parent, text="LAZ Colorization Parameters", padding=10)
        laz_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(laz_frame, text="Voxel Grid Size (m):").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(laz_frame, textvariable=self.voxel_size, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(laz_frame, text="1 cm = 0.01  |  2 cm = 0.02  |  5 cm = 0.05",
                  font=('Arial', 8, 'italic')).grid(row=0, column=2, sticky='w', padx=5)
        
        ttk.Label(laz_frame, text="RGB Assignment:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Combobox(laz_frame, textvariable=self.voxel_rgb_method,
                     values=['nearest', 'mean', 'median', 'std'],
                     width=10, state='readonly').grid(row=1, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(laz_frame,
                  text="nearest: closest pixel to voxel centre  |  mean: average RGB  |  median: median RGB",
                  font=('Arial', 8, 'italic')).grid(row=1, column=2, sticky='w', padx=5)
    
    def create_selection_tab(self, parent):
        """Image selection options"""
        frame = ttk.LabelFrame(parent, text="Selection Mode", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Radiobutton(frame, text="Process All Images", variable=self.selection_mode, value="all").pack(anchor='w', pady=5)
        
        # Specific images
        specific_frame = ttk.Frame(frame)
        specific_frame.pack(fill='x', pady=5)
        ttk.Radiobutton(specific_frame, text="Select Specific Images from File:", variable=self.selection_mode, value="specific").pack(anchor='w')
        entry_frame = ttk.Frame(specific_frame)
        entry_frame.pack(fill='x', padx=20)
        ttk.Entry(entry_frame, textvariable=self.selected_images_file, width=50).pack(side='left', padx=5)
        ttk.Button(entry_frame, text="Browse", command=lambda: self.browse_file(self.selected_images_file)).pack(side='left')
        ttk.Label(entry_frame, text="(Text file with one filename per line)", font=('Arial', 8, 'italic')).pack(side='left', padx=5)
        
        # Location-based
        location_frame = ttk.Frame(frame)
        location_frame.pack(fill='x', pady=5)
        ttk.Radiobutton(location_frame, text="Filter by Location (Lat/Lon + Radius):", variable=self.selection_mode, value="location").pack(anchor='w')
        
        coords_frame = ttk.Frame(location_frame)
        coords_frame.pack(fill='x', padx=20)
        
        ttk.Label(coords_frame, text="Latitude:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(coords_frame, textvariable=self.center_lat, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(coords_frame, text="Longitude:").grid(row=0, column=2, sticky='w', padx=10, pady=5)
        ttk.Entry(coords_frame, textvariable=self.center_lon, width=15).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(coords_frame, text="Radius (m):").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(coords_frame, textvariable=self.radius_m, width=15).grid(row=1, column=1, padx=5, pady=5)
        
        # Skip interval option (applies to all selection modes)
        skip_frame = ttk.LabelFrame(frame, text="Image Interval", padding=10)
        skip_frame.pack(fill='x', pady=10)
        
        ttk.Label(skip_frame, text="Process every Nth image:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Spinbox(skip_frame, from_=1, to=100, textvariable=self.skip_interval, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(skip_frame, text="(1 = all images, 2 = every other, 3 = every third, etc.)", font=('Arial', 8, 'italic')).grid(row=0, column=2, sticky='w', padx=5)
    
    def create_camera_tab(self, parent):
        """Camera calibration parameters"""
        frame = ttk.LabelFrame(parent, text="Lever Arms (Body Frame)", padding=10)
        frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(frame, text="X (forward):").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(frame, textvariable=self.lever_arm_x, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame, text="Y (right):").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(frame, textvariable=self.lever_arm_y, width=15).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(frame, text="Z (up):").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Entry(frame, textvariable=self.lever_arm_z, width=15).grid(row=2, column=1, padx=5, pady=5)
        
        offset_frame = ttk.LabelFrame(parent, text="Angular Offsets (degrees)", padding=10)
        offset_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(offset_frame, text="Pitch Offset:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(offset_frame, textvariable=self.pitch_offset, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(offset_frame, text="Roll Offset:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(offset_frame, textvariable=self.roll_offset, width=15).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(offset_frame, text="Heading Offset:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Entry(offset_frame, textvariable=self.heading_offset, width=15).grid(row=2, column=1, padx=5, pady=5)

    def create_strip_tab(self, parent):
        """Panoramic strip mosaic settings."""

        # --- Input source -----------------------------------------------
        src_frame = ttk.LabelFrame(parent, text="GeoTIFF Input Source", padding=10)
        src_frame.pack(fill='x', padx=10, pady=10)

        ttk.Radiobutton(
            src_frame,
            text="Use GeoTIFFs generated in the current processing run",
            variable=self.strip_input_mode,
            value='from_run',
            command=self._update_strip_dir_state,
        ).grid(row=0, column=0, columnspan=3, sticky='w', pady=3)

        ttk.Radiobutton(
            src_frame,
            text="Use an existing GeoTIFF directory:",
            variable=self.strip_input_mode,
            value='from_dir',
            command=self._update_strip_dir_state,
        ).grid(row=1, column=0, sticky='w', pady=3)

        self._strip_dir_entry = ttk.Entry(src_frame, textvariable=self.strip_geotiff_dir, width=55)
        self._strip_dir_entry.grid(row=1, column=1, padx=5, pady=3)

        self._strip_dir_button = ttk.Button(
            src_frame,
            text="Browse",
            command=lambda: self.browse_directory(self.strip_geotiff_dir),
        )
        self._strip_dir_button.grid(row=1, column=2, pady=3)

        ttk.Label(
            src_frame,
            text="(Files are merged in filename sort order — ensure filenames are zero-padded / sequential)",
            font=('Arial', 8, 'italic'),
        ).grid(row=2, column=0, columnspan=3, sticky='w', padx=5, pady=2)

        # --- Strip parameters -------------------------------------------
        param_frame = ttk.LabelFrame(parent, text="Strip Parameters", padding=10)
        param_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(param_frame, text="Images per strip:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Spinbox(
            param_frame, from_=2, to=1000, increment=10,
            textvariable=self.images_per_strip, width=10,
        ).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(
            param_frame,
            text="Number of individual GeoTIFFs merged into each panoramic strip",
            font=('Arial', 8, 'italic'),
        ).grid(row=0, column=2, sticky='w', padx=5)

        ttk.Label(param_frame, text="Resolution (m/pixel):").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(param_frame, textvariable=self.strip_resolution_m, width=10).grid(
            row=1, column=1, sticky='w', padx=5, pady=5
        )
        ttk.Label(
            param_frame,
            text="e.g. 0.002 = 2 mm  |  0.005 = 5 mm  |  0.01 = 1 cm  |  0.02 = 2 cm",
            font=('Arial', 8, 'italic'),
        ).grid(row=1, column=2, sticky='w', padx=5)

        ttk.Label(param_frame, text="Output filename prefix:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Entry(param_frame, textvariable=self.strip_prefix, width=20).grid(
            row=2, column=1, sticky='w', padx=5, pady=5
        )
        ttk.Label(
            param_frame,
            text="Strips are named <prefix>_001.tif, <prefix>_002.tif, …",
            font=('Arial', 8, 'italic'),
        ).grid(row=2, column=2, sticky='w', padx=5)

        # --- Output note ------------------------------------------------
        note_frame = ttk.LabelFrame(parent, text="Output", padding=10)
        note_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(
            note_frame,
            text="Panoramic strips are written to <Output Directory>/panoramic_strips/",
            font=('Arial', 9),
        ).pack(anchor='w')
        ttk.Label(
            note_frame,
            text=(
                "Alignment is coordinate-based (no feature matching). "
                "Each input GeoTIFF's embedded CRS and affine transform determine "
                "its exact position in the merged strip."
            ),
            font=('Arial', 8, 'italic'),
            wraplength=700,
            justify='left',
        ).pack(anchor='w', pady=4)

        # Set initial widget states
        self._update_strip_dir_state()

    def _update_strip_dir_state(self):
        """Enable / disable the directory entry based on the selected input mode."""
        state = 'normal' if self.strip_input_mode.get() == 'from_dir' else 'disabled'
        self._strip_dir_entry.config(state=state)
        self._strip_dir_button.config(state=state)

    def create_bottom_panel(self):
        """Progress display and control buttons"""
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Progress text area
        ttk.Label(bottom_frame, text="Processing Log:").pack(anchor='w')
        self.progress_text = scrolledtext.ScrolledText(bottom_frame, height=10, wrap=tk.WORD)
        self.progress_text.pack(fill='both', expand=True, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(fill='x', pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side='right', padx=5)
    
    def browse_file(self, var):
        """Open file browser for file selection"""
        filename = filedialog.askopenfilename()
        if filename:
            var.set(filename)
    
    def browse_laz_file(self, var):
        """Open file browser filtered to LAZ/LAS files"""
        filename = filedialog.askopenfilename(
            filetypes=[("LAZ/LAS files", "*.laz *.las"), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)
    
    def browse_directory(self, var):
        """Open directory browser"""
        dirname = filedialog.askdirectory()
        if dirname:
            var.set(dirname)
    
    def log_message(self, message):
        """Add message to progress log"""
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        """Clear the progress log"""
        self.progress_text.delete(1.0, tk.END)

    def normalize_image_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize user-provided image list dataframe column names to the expected schema.

        Expected user columns include:
        filename,image_time,image_number,pitch,roll,heading,northing,easting,depth,altitude,velocity,latitude,longitude

        This maps them to the internal names used by processors (e.g., file_name, AUV_Pitch, AUV_Latitude, etc.).
        """
        df = df.copy()
        mapping = {
            'filename': 'file_name',
            'image_time': 'Date_Time',
            'image_number': 'Image_Number',
            'pitch': 'AUV_Pitch',
            'roll': 'AUV_Roll',
            'heading': 'AUV_Heading',
            'northing': 'AUV_Northing',
            'easting': 'AUV_Easting',
            'depth': 'AUV_Depth',
            'altitude': 'AUV_Altitude',
            'latitude': 'AUV_Latitude',
            'longitude': 'AUV_Longitude',
            'velocity': 'AUV_Velocity'
        }

        for src, dst in mapping.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]

        return df
    
    def get_selected_images(self):
        """Get list of selected images based on selection mode"""
        mode = self.selection_mode.get()
        skip = self.skip_interval.get()
        
        selected_images = None
        
        if mode == "all":
            # Apply skip interval to entire dataset
            if skip > 1:
                image_list_csv = self.image_list_csv.get()
                if not image_list_csv or not os.path.exists(image_list_csv):
                    raise ValueError("Image list CSV not found")

                df = pd.read_csv(image_list_csv)
                df = self.normalize_image_df(df)
                # Select every Nth image
                selected_images = df.iloc[::skip]['file_name'].tolist()
                self.log_message(f"Skip interval {skip}: Processing {len(selected_images)} of {len(df)} total images")
            else:
                selected_images = None  # Process all images
        
        elif mode == "specific":
            filepath = self.selected_images_file.get()
            if not filepath or not os.path.exists(filepath):
                raise ValueError("Selected images file not found")
            
            with open(filepath, 'r') as f:
                images = [line.strip() for line in f if line.strip()]
            
            # Apply skip interval
            if skip > 1:
                selected_images = images[::skip]
                self.log_message(f"Skip interval {skip}: Processing {len(selected_images)} of {len(images)} selected images")
            else:
                selected_images = images
        
        elif mode == "location":
            # Filter images by distance from center point
            image_list_csv = self.image_list_csv.get()
            if not image_list_csv or not os.path.exists(image_list_csv):
                raise ValueError("Image list CSV not found")
            
            df = pd.read_csv(image_list_csv)
            df = self.normalize_image_df(df)

            # Assuming the CSV has 'AUV_Latitude' and 'AUV_Longitude' columns
            center_lat = self.center_lat.get()
            center_lon = self.center_lon.get()
            radius = self.radius_m.get()

            if 'AUV_Latitude' not in df.columns or 'AUV_Longitude' not in df.columns:
                raise ValueError("Image list CSV must contain 'AUV_Latitude' and 'AUV_Longitude' columns")

            # Calculate distances
            distances = df.apply(
                lambda row: haversine_distance(
                    center_lat, center_lon,
                    row['AUV_Latitude'], row['AUV_Longitude']
                ),
                axis=1
            )
            
            # Filter by radius
            within_radius = df[distances <= radius]
            self.log_message(f"Found {len(within_radius)} images within {radius}m of ({center_lat}, {center_lon})")
            
            # Apply skip interval
            if skip > 1:
                selected_images = within_radius.iloc[::skip]['file_name'].tolist()
                self.log_message(f"Skip interval {skip}: Processing {len(selected_images)} of {len(within_radius)} images in radius")
            else:
                selected_images = within_radius['file_name'].tolist()
        
        return selected_images
    
    def validate_inputs(self):
        """Validate all required inputs"""
        if not self.image_list_csv.get() or not os.path.exists(self.image_list_csv.get()):
            raise ValueError("Image list CSV missing or not found")
        
        if not self.image_dir.get() or not os.path.exists(self.image_dir.get()):
            raise ValueError("Image directory is required")
        
        if not self.output_dir.get():
            raise ValueError("Output directory not specified")
        
        if self.create_pointcloud.get():
            if not self.lls_list_csv.get() or not os.path.exists(self.lls_list_csv.get()):
                raise ValueError("LLS list CSV is required for point cloud generation")
            
            if not self.lls_list_csv.get() or not os.path.exists(self.lls_list_csv.get()):
                raise ValueError("LLS list CSV required for point cloud generation and not found")
            if not self.lls_dir.get() or not os.path.exists(self.lls_dir.get()):
                raise ValueError("LLS directory required for point cloud generation and not found")
        
        if self.colorize_laz.get():
            lls_dir = self.lls_dir.get().strip()
            if not lls_dir or not os.path.exists(lls_dir):
                raise ValueError("LLS directory is required for LAZ colorization and was not found")

        if self.create_panoramic_strips.get():
            if self.strip_input_mode.get() == 'from_dir':
                sdir = self.strip_geotiff_dir.get().strip()
                if not sdir or not os.path.exists(sdir):
                    raise ValueError("A valid GeoTIFF directory is required when using 'existing directory' strip mode")
            try:
                res = self.strip_resolution_m.get()
                if res <= 0:
                    raise ValueError("Strip resolution must be > 0")
            except tk.TclError:
                raise ValueError("Strip resolution must be a positive number (e.g. 0.005)")
            try:
                n = self.images_per_strip.get()
                if n < 2:
                    raise ValueError("Images per strip must be at least 2")
            except tk.TclError:
                raise ValueError("Images per strip must be a positive integer")

        if not self.create_geotiff.get() and not self.create_pointcloud.get() and not self.colorize_laz.get() and not self.create_panoramic_strips.get():
            raise ValueError("At least one output type must be selected")
    
    def start_processing(self):
        """Start the processing in a separate thread"""
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing is already running")
            return
        
        try:
            self.validate_inputs()
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
            return
        
        self.is_processing = True
        self.start_button.config(state='disabled')
        self.clear_log()
        
        # Run processing in separate thread
        thread = threading.Thread(target=self.run_processing)
        thread.daemon = True
        thread.start()
    
    def run_processing(self):
        """Run the actual processing"""
        try:
            selected_images = self.get_selected_images()

            # Create output directories
            output_base = self.output_dir.get()
            os.makedirs(output_base, exist_ok=True)

            # Read and write a normalized image list CSV that matches processor expectations
            try:
                orig_csv = self.image_list_csv.get()
                df_orig = pd.read_csv(orig_csv)
                df_norm = self.normalize_image_df(df_orig)
                normalized_csv_path = os.path.join(output_base, 'normalized_image_list.csv')
                df_norm.to_csv(normalized_csv_path, index=False)
                self.log_message(f"Wrote normalized image list to {normalized_csv_path}")
            except Exception as e:
                self.log_message(f"Warning: Failed to create normalized CSV: {e}")
                normalized_csv_path = self.image_list_csv.get()
            
            # Process GeoTIFFs
            if self.create_geotiff.get():
                self.log_message("="*60)
                self.log_message("Starting GeoTIFF Generation")
                self.log_message("="*60)
                
                geotiff_output = os.path.join(output_base, 'GeoTIFFs')
                
                stats = process_images_to_geotiffs(
                    image_list_csv=normalized_csv_path,
                    image_dir=self.image_dir.get(),
                    output_dir=geotiff_output,
                    selected_images=selected_images,
                    lever_arm_x=self.lever_arm_x.get(),
                    lever_arm_y=self.lever_arm_y.get(),
                    lever_arm_z=self.lever_arm_z.get(),
                    pitch_offset=self.pitch_offset.get(),
                    roll_offset=self.roll_offset.get(),
                    heading_offset=self.heading_offset.get(),
                    utm_zone=self.utm_zone.get(),
                    utm_hemisphere=self.utm_hemisphere.get(),
                    dpi=self.dpi.get(),
                    progress_callback=self.log_message
                )
                
                self.log_message(f"\nGeoTIFF Summary: {stats}")
            
            # Process Point Clouds
            if self.create_pointcloud.get():
                self.log_message("\n" + "="*60)
                self.log_message("Starting Point Cloud Generation")
                self.log_message("="*60)
                
                pointcloud_output = os.path.join(output_base, 'pointclouds')
                
                stats = process_images_to_pointclouds(
                    image_list_csv=normalized_csv_path,
                    lls_list_csv=self.lls_list_csv.get(),
                    lls_dir=self.lls_dir.get(),
                    image_dir=self.image_dir.get(),
                    output_dir=pointcloud_output,
                    selected_images=selected_images,
                    lever_arm_x=self.lever_arm_x.get(),
                    lever_arm_y=self.lever_arm_y.get(),
                    lever_arm_z=self.lever_arm_z.get(),
                    pitch_offset=self.pitch_offset.get(),
                    roll_offset=self.roll_offset.get(),
                    heading_offset=self.heading_offset.get(),
                    downsample=self.downsample.get(),
                    max_distance=self.max_distance.get(),
                    progress_callback=self.log_message,
                    output_format=self.pointcloud_format.get()
                )
                
                self.log_message(f"\nPoint Cloud Summary: {stats}")
            
            # Colorize LAZ Point Cloud (support multiple files in lls_dir)
            if self.colorize_laz.get():
                self.log_message("\n" + "="*60)
                self.log_message("Starting LAZ Colorization")
                self.log_message("="*60)

                laz_out_dir = os.path.join(output_base, 'colorized_laz')
                os.makedirs(laz_out_dir, exist_ok=True)

                # Scan the LLS directory for .laz/.las files
                lls_dir_val = self.lls_dir.get().strip()
                laz_files = []
                if lls_dir_val and os.path.isdir(lls_dir_val):
                    for fn in os.listdir(lls_dir_val):
                        if fn.lower().endswith(('.laz', '.las')):
                            laz_files.append(os.path.join(lls_dir_val, fn))

                if not laz_files:
                    self.log_message("No LAZ/LAS files found for colorization. Skipping.")
                else:
                    for laz_in in laz_files:
                        laz_basename = os.path.splitext(os.path.basename(laz_in))[0]
                        laz_out = os.path.join(laz_out_dir, f"{laz_basename}_colorized.laz")
                        try:
                            stats = colorize_laz_from_images(
                                laz_input_path=laz_in,
                                image_list_csv=normalized_csv_path,
                                image_dir=self.image_dir.get(),
                                output_path=laz_out,
                                voxel_size=self.voxel_size.get(),
                                rgb_method=self.voxel_rgb_method.get(),
                                lever_arm_x=self.lever_arm_x.get(),
                                lever_arm_y=self.lever_arm_y.get(),
                                lever_arm_z=self.lever_arm_z.get(),
                                pitch_offset=self.pitch_offset.get(),
                                roll_offset=self.roll_offset.get(),
                                heading_offset=self.heading_offset.get(),
                                downsample=self.downsample.get(),
                                selected_images=selected_images,
                                lls_list_csv=self.lls_list_csv.get() or None,
                                progress_callback=self.log_message,
                            )
                            self.log_message(f"\nLAZ Colorization Summary for {os.path.basename(laz_in)}: {stats}")
                        except Exception as e:
                            self.log_message(f"Error colorizing {laz_in}: {e}")
                            import traceback
                            self.log_message(traceback.format_exc())
            
            # Copy original images if requested (specifically for radius-based search)
            if self.copy_original_images.get():
                self.log_message("\n" + "="*60)
                self.log_message("Copying Original Images")
                self.log_message("="*60)
                
                images_output = os.path.join(output_base, 'original_images')
                os.makedirs(images_output, exist_ok=True)
                
                # Get the selected images (particularly useful for radius-based search)
                import shutil
                image_dir = self.image_dir.get()
                copied_count = 0
                failed_count = 0
                
                if selected_images:
                    # Copy only the selected images
                    for img_name in selected_images:
                        src_path = os.path.join(image_dir, img_name)
                        dst_path = os.path.join(images_output, img_name)
                        try:
                            if os.path.exists(src_path):
                                shutil.copy2(src_path, dst_path)
                                copied_count += 1
                            else:
                                self.log_message(f"Warning: Image not found: {img_name}")
                                failed_count += 1
                        except Exception as e:
                            self.log_message(f"Error copying {img_name}: {str(e)}")
                            failed_count += 1
                    
                    self.log_message(f"Copied {copied_count} images to {images_output}")
                    if failed_count > 0:
                        self.log_message(f"Failed to copy {failed_count} images")
                else:
                    self.log_message("No image selection applied - skipping copy (use with specific/radius-based selection)")

            # Create Panoramic Strips
            if self.create_panoramic_strips.get():
                self.log_message("\n" + "="*60)
                self.log_message("Starting Panoramic Strip Generation")
                self.log_message("="*60)

                # Determine source GeoTIFF directory
                if self.strip_input_mode.get() == 'from_run':
                    src_geotiff_dir = os.path.join(output_base, 'GeoTIFFs')
                    if not os.path.isdir(src_geotiff_dir):
                        self.log_message(
                            f"  WARNING: GeoTIFF output directory not found at {src_geotiff_dir}. "
                            "Enable 'Create GeoTIFF' in the same run, or switch to "
                            "'Use existing GeoTIFF directory' mode."
                        )
                    else:
                        strip_output_dir = os.path.join(output_base, 'panoramic_strips')
                        strip_stats = create_panoramic_strips(
                            geotiff_dir=src_geotiff_dir,
                            output_dir=strip_output_dir,
                            images_per_strip=self.images_per_strip.get(),
                            resolution_m=self.strip_resolution_m.get(),
                            strip_prefix=self.strip_prefix.get(),
                            progress_callback=self.log_message,
                        )
                        self.log_message(f"\nPanoramic Strip Summary: {strip_stats}")
                else:
                    src_geotiff_dir = self.strip_geotiff_dir.get().strip()
                    strip_output_dir = os.path.join(output_base, 'panoramic_strips')
                    strip_stats = create_panoramic_strips(
                        geotiff_dir=src_geotiff_dir,
                        output_dir=strip_output_dir,
                        images_per_strip=self.images_per_strip.get(),
                        resolution_m=self.strip_resolution_m.get(),
                        strip_prefix=self.strip_prefix.get(),
                        progress_callback=self.log_message,
                    )
                    self.log_message(f"\nPanoramic Strip Summary: {strip_stats}")

            self.log_message("\n" + "="*60)
            self.log_message("ALL PROCESSING COMPLETE!")
            self.log_message("="*60)
            
            messagebox.showinfo("Success", "Processing completed successfully!")
            
        except Exception as e:
            self.log_message(f"\nERROR: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        
        finally:
            self.is_processing = False
            self.start_button.config(state='normal')


def main():
    root = tk.Tk()
    app = LLSImageProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
