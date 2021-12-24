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
- [ ] Camera Capture
- [x] Create folder
- [x] Timer
- [x] Image Slideshow

### Program
- [ ] Main Eye tracking
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
