# Rotation Giffer Plugin

## Overview
Rotation Giffer is an export plugin designed for MoleditPy. It enables users to generate smooth, high-quality rotating GIFs of loaded molecules. Instead of rotating the model itself, the plugin mathematically orbits the camera around your choice of global or view-specific axes, ensuring consistent lighting and perfect looping capability.

## Key Features
* **Flexible Rotation Axes:** Choose to orbit around absolute Global Axes (X, Y, Z) or relative Local View Axes (Roll, Pitch/Elevation, Yaw/Azimuth).
* **Direction Control:** Toggle inverse rotation to instantly reverse the camera's orbital path.
* **Granular Animation Controls:** Custom inputs for Total Rotation Angle (10° to 3600°), Total Frames (for animation smoothness), and Speed (FPS).
* **Transparency Support:** Export GIFs with cleanly masked transparent backgrounds (requires `Pillow`).
* **High-Quality Rendering:** Utilizes Super Sample Anti-Aliasing (SSAA) and adaptive color quantization palettes for crisp, accurate visuals without the typical GIF color-banding artifacts.

## Requirements
This plugin relies on the host application's environment (specifically a PyVista `plotter` and PyQt6 UI). Ensure the following dependencies are available:
* `PyQt6`
* `numpy`
* `pyvista`
* `Pillow` / `PIL` (Optional, but **highly recommended** for transparent backgrounds and the high-quality color pipeline).

## Usage Instructions
1. Ensure a molecule or 3D object is actively loaded in the main viewer.
2. Open the **Export** menu and select **Generate Rotation GIF...**.
3. In the Rotation Giffer dialog, adjust your parameters:
   * **Rotation Axis:** Select the pivot axis for the camera.
   * **Direction:** Check "Inverse Rotation" if you want a counter-clockwise orbit.
   * **Total Rotation:** Set the total degrees for the animation (e.g., 360° for a full loop).
   * **Total Frames / Speed:** Balance these to achieve your desired smoothness and file size.
   * **Options:** Enable "Transparent Background" and "High Quality Colors" for the best results.
4. Click **Generate GIF**, select your output directory, and allow the renderer to process the frames. The camera state will automatically restore to its original position once complete.

## Development and Testing

This repository includes a comprehensive unit test suite to verify dialog setup, vector rotation mathematics, and GIF generation flows.

To run the unit tests, execute:
```bash
python -m pytest tests/ -v
```

See the [tests/README.md](tests/README.md) directory for details on test coverage, stubs, and architecture.

