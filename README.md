# 3D from Stereo Vision System

## Overview
This repository contains a Python-based stereo vision system developed for the "3D from Stereo" project, part of the Image Processing and Computer Vision course at the University of Bristol. The system is designed to perform 3D reconstruction from stereo images using OpenCV and Open3D libraries. It processes stereo images to detect spheres, estimate their positions and radii, and reconstruct their 3D spatial layout. The project emphasizes flexibility through command-line arguments, allowing dynamic adjustments for various parameters.

## Requirements
- Python 3.8+ 
- OpenCV library (cv2) 
- Open3D 0.16.0 
- NumPy 

## Installation
### For MVB-2.11/1.15 Lab Machine in University of Bristol
1. Ensure ```Anaconda``` is ready to use, you may need to:
   ```bash
   module load anaconda/3-2023
   ```
2. Create a virtual environment using conda:  
   ```bash
   conda create -n ipcv-2 python=3.8
   ```
3. Activate the virtual environment:  
   ```bash
   conda activate ipcv-2
   ```
4. Install OpenCV packages:  
   ```bash
   pip install opencv-python open3d==0.16.0 numpy
   ```

### For the Linux-Based Machine
1. Ensure Python 3.8+ is installed on your system. You can download it from [Python's official website](https://www.python.org/downloads/).

2. Install the required libraries using pip. Run the following commands in your terminal or command prompt:

    ```bash
    pip install opencv-python open3d==0.16.0 numpy
    ```
   or
    ```bash
   pip3 install opencv-python open3d==0.16.0 numpy
    ```

### For the Mac (Intel & ARM) - Based Machine
You may want to use ```Conda```
1. Install a virtual environment using conda:  
    ```bash 
   conda create -n ipcv-2 python=3.8
   ```
2. Activate the virtual environment:
    ```bash
    conda activate ipcv-2
   ```
3. Might need to update pip: 
    ```bash
    pip install --upgrade pip
    ```
4. Install OpenCV packages:
    ```bash
    pip install opencv-python
    ```
5. Install Open3D packages: 
    ```bash
    pip install open3d==0.16.0
    ```
### For the Windows - Based Machine
You may want to use ```Conda```, recommend use Python 3.8, numpy 1.23.3 and Open3D 0.11.2
1. Install a virtual environment using conda:
    ```bash 
   conda create -n ipcv-2 python=3.8
   ```
2. Activate the virtual environment:
    ```bash
    conda activate ipcv-2
   ```
3. Might need to update pip:
    ```bash
    pip install --upgrade pip
    ```
4. Install OpenCV packages:
    ```bash
    pip install opencv-python
    ```
5. Install Open3D packages:
    ```bash
    pip install open3d==0.11.2
    ```

## Usage

### Running the Script
1. The main script ```CWII2324-v2.py``` can be run from the command line.  
2. Navigate to the project directory containing the script in the command line.  
3. Run the script with the file name, add the additional command line argument as a parameter if needed. 

#### For example:   
1. If just want to run the script without the Command Line Arguments, and get the result, just simply run:  
    ```bash
    python CWII2324-v2.py
    ```
2. Run the script with custom parameters:
    ```bash
    python CWII2324-v2.py --display_centers
    ```
   it will show the 3D render first, and after pressed ```esc```, it will display the centers.

### Command Line Arguments
The script accepts various arguments to control different aspects of the 3D reconstruction process:

- `--num`: Number of spheres (default: 6)
- `--sph_rad_min`: Minimum sphere radius (default: 10)
- `--sph_rad_max`: Maximum sphere radius (default: 16)
- `--sph_sep_min`: Minimum sphere separation (default: 4)
- `--sph_sep_max`: Maximum sphere separation (default: 8)
- `--display_centre`: Visualize centres (boolean flag)
- `--coords`: Display coordinates (boolean flag)
- `--display_spheres`: Display epipolar lines and spheres in the 3D visualizer (boolean flag)
- `--display_centers`: Display estimated and ground truth centers in the 3D visualizer (boolean flag)
- `--pos_noise`: Noise level for position (default: 0)
- `--orient_noise`: Noise level for orientation (default: 0)

#### Example
Run the script with custom parameters:

```bash
python CWII2324-v2.py --num=8 --sph_rad_min=12 --sph_rad_max=20 --display_centers
```

This command will run the script for 8 spheres with radii ranging from 12 to 20 and display the estimated and ground truth centers in the 3D visualizer.

## Key Functionalities
After ran the script:
```bash
python CWII2324-v2.py
```
you will get the following:
- **Circle Detection**: Detects spheres in stereo images using Hough Circle Transform, the saved images are stored as `````'hough_circles_img0.png'````` and `````'hough_circles_img1.png'`````
- **Epipolar Line Calculation**: Computes and draws Epipolar lines for each detected sphere in stereo-image pairs, the saved image is stored as `````'view1_eline_fmat.png'`````
- **Finding Correspondences**: Establishes correspondences between detected circles in two different views, showing the data in the terminal as ```Corredpondence found: Image 0 Center: [...], Image 1 Center: [...]```
- **3D Location of Sphere Centers**: Calculates 3D positions of sphere centers based on stereo image correspondences, showing the data in the terminal as ```Reconstructed 3D center: [...]```.
- **Evaluation and Visualization the Centers**: Compares the estimated 3D locations with ground truth and visualizes them, please run the code: 
   ```bash
   python CWII2324-v2.py --display_centers
   ```
  and press the ```esc```, then the 3D render for reconstructed 3D vs ground truth centers, the error will show in the terminal as ```Error in estimated center: ...```.
- **3D Radius Estimation**: Estimates the 3D radii of the detected spheres, the data showing in the terminal as```Estimated 3D radius for sphere ...: ...```
- **Evaluation and Visualization the Spheres**: Use calculated 3D positions of sphere centers, and use Radius found in estimation, compares with the ground truth and visualizes them, please run the code:
   ```bash
  python CWII2324-v2.py --display_spheres
   ```
  and press the ```esc```, then the 3D render for 3D radius of each sphere vs ground truth centers, the error will show in the terminal as ```Radiux Error: ```.
- **Impact of Noise**: Investigates the effect of noise on the relative pose of cameras in the stereo vision setup, please use ```Command Line Arguments``` to visualize the difference.

---
*This README is part of the 3D from Stereo Vision System project, developed as part of the coursework at the University of Bristol.*