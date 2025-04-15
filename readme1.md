# Drone Path Visualization

## Overview

This project processes drone footage to localize frames on a global map and visualize the drone's trajectory. It generates:
- A static image (`path.png`) showing the drone's path with a gradient-colored trajectory and key points marked with bold dots indicating where frames were captured.
- An animation video (`trajectory_with_trail.mp4`) displaying the drone's path on the map, with a moving arrow, trail, and inset drone footage at key points.

The solution uses feature matching (SIFT for global search, AKAZE for local search) to localize drone frames, PCHIP interpolation for smooth trajectory visualization, and Matplotlib/OpenCV/Pillow for rendering.

## Deliverables

- **Code**: Organized and commented Python script (`main.py`) for processing drone frames, localizing them, and generating visualizations.
- **Static Visualization**: `path.png` shows the drone's route on the global map.
- **Animation**: `trajectory_with_trail.mp4`.
- **Requirements**: `requirements.txt` lists necessary Python libraries.
- **README**: This file, describing the solution and usage.

## Requirements

- Python 3.8+
- Libraries listed in `requirements.txt`:
  - numpy
  - matplotlib
  - scipy
  - opencv-python
  - pillow
- A local Python environment (e.g., PyCharm, VS Code, or similar IDE). The code is designed to run locally, not in a cloud environment.

## Usage

The project includes two main functions in `main.py` for generating results:

1. **`static_path(coordinates_array, global_map)`**:
   - Generates a static image (`path.png`) quickly.
   - Visualizes the drone's trajectory with a gradient-colored path and marks key points (where frames were captured) with bold dots.

2. **`animate_path(coordinates_array, global_map, drone_frames)`**:
   - Generates an animation video (`trajectory_with_trail.mp4`) showing the drone's path with a moving arrow, trail, and zoomed-in drone footage at key points.
   - This function is computationally intensive and takes significant time to execute. It is commented out in the code by default to avoid long runtimes.
   - A pre-generated result (`trajectory_with_trail.mp4`) is included in the project for convenience.

To run the code:
1. Set up a local Python environment with Python 3.8+.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `main.py` in your IDE (e.g., PyCharm or VS Code). By default, it will execute `static_path` to generate `path.png`.
4. To generate the animation, uncomment the call to `animate_path` in `main.py`. Note that this may take considerable time.

## Notes

- The included `trajectory_with_trail.mp4` demonstrates the output of `animate_path` without needing to run the time-intensive function.
- Ensure sufficient disk space and memory when running the animation generation.
