# Pool-project-experiment

## dependencies
- openCV
- Numpy
- PySimpleGUI
- Dlib

## To do:

### User Interface
- [x] Calibration
- [x] Export & import video
- [x] Camera Capture
- [x] Create folder
- [x] Timer
- [x] Image Slideshow

### Program
- [x] Main Eye tracking
- [ ] Screen projection
- [ ] After & live processing
- [ ] global class to store fault data(like when the blobdetector doesn't find a blob)

## Coding guides

- mark when going from one cordinate system to an other with 
  -- "#--------transform to ****** space------------"
  
- All time units are measured in ms
- openCV library is written as cv2
- numpy library is written as np
- PySimpleGUI library is written as sg
- all interfaces with SystemDefaultForReal theme

## Installation

Clone this project:

```shell
git clone https://github.com/sarif200/pool-project-experiment
```

### For Pip install
Install these dependencies (NumPy, OpenCV, Dlib, PySimpleGUI):

```shell
pip install -r requirements.txt
```
CMake is required for this project.

## Manual
1. Open the main file
2. Select New Project
3. Fill project name in
4. Complete Calibration
5. Images are shown
6. Navigate to results page
7. Open folder
8. Video is projected on screen

## Explanation
### main.py
- Line 1-5: Import of libraries & functions from other files.
- Line 8: Define the function.
- Line 9: Set theme for the UI.
- Line 10-28: Here we define the layout for the UI.
- Line 30-31: Here we define the window with the window title and the layout.
- Line 33-43: We define the event loop and listen to all the event and execute the actions.
- Line 44: When out of the event loop, close the window.
- Line 46-47: When file is ran, execute the main function.

### new_project.py

### calibration.py

### experiment.py

### results.py

### tracking.py
