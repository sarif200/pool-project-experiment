import cv2
import os
import time
import sys
import ctypes
import numpy as np

scriptDir = os.path.dirname(__file__)
tracking_folder = os.path.join(scriptDir, '../tracking/')
path = os.path.abspath(tracking_folder)

sys.path.append(path)
from tracking import pupil_tracker, gaze_tracker

tracker = pupil_tracker
gaze = gaze_tracker

def show_image(img_path):
    # Get screen size
    user32 = ctypes.windll.user32
    size_screen = user32.GetSystemMetrics(1), user32.GetSystemMetrics(0)
    
    # Create white background
    background = (np.zeros((int(size_screen[0]), int(size_screen[1]), 3)) + 255).astype('uint8')

    # Calculate midpoint of screen
    mid_x = int(size_screen[0]) / 2
    mid_y = int(size_screen[1]) / 2
    
    # Get images
    img = cv2.imread(img_path)

    # Get height & width from image
    img_width, img_height = img.shape[:2]

    # Caclulate middle of screen
    yoff = round((mid_y - img_height)/2)
    xoff = round((mid_x + img_width/4))

    # Creating overlay
    dst = background.copy()
    dst[yoff: yoff + img_height, xoff:xoff + img_width] = img
    
    # Show images
    cv2.namedWindow("Display", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Display', dst)

def cycle_images(final_folder_path):
    # Get file path from current data directory
    filename = "original_video.avi"
    currentdir_folder = os.path.join(final_folder_path, filename)
    project_folder = os.path.abspath(currentdir_folder)

    TIME = 3
    t = TIME * 1000 # transform to miliseconds

    # Get file path
    scriptDir = os.path.dirname(__file__)
    imgdir_folder = os.path.join(scriptDir, '../img/')
    img_folder = os.path.abspath(imgdir_folder)

    # Get images from folder
    images = os.listdir(img_folder)

    cnt = len(images)
    idx = 1
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    fps = 15

    # Video Codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(project_folder, fourcc, fps, (640, 480))
    
    prev_time = time.time()
    delta_since_last_change = 0

    show_image(os.path.join(img_folder, images[0]))

    while (idx < cnt):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # pupils = tracker.detect_in_frame(tracker,frame)
        pupils = gaze.track_in_frame(gaze,frame)

        # output.write(frame)
        pupil_l = (int(pupils[0][0]),int(pupils[0][1]))
        pupil_r = (int(pupils[1][0]),int(pupils[1][1]))

        cv2.circle(frame,(int(pupils[0][0]),int(pupils[0][1])),10,(0, 255, 0),3)
        cv2.circle(frame,(int(pupils[1][0]),int(pupils[1][1])),10,(255, 0, 0),3)
        cv2.imshow('frame', frame)
        
        delta = time.time() - prev_time 
        delta_since_last_change += delta
        prev_time = time.time()

        if delta_since_last_change >= TIME:
            delta_since_last_change = 0
            img_path = os.path.join(img_folder, images[idx])
            show_image(img_path)
            print(images[idx])

            gaze.export(delta_since_last_change, pupil_l, pupil_r, project_folder, images[idx])

            idx += 1 if idx < cnt else cnt

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
