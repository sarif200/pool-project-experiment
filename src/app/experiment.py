from weakref import ref
import cv2
import os
import time
import ctypes
import numpy as np

def show_image(img_path):
    # Get screen size
    user32 = ctypes.windll.user32
    size_screen = user32.GetSystemMetrics(1), user32.GetSystemMetrics(0)
    print(size_screen)
    
    # Create white background
    background = (np.zeros((int(size_screen[0]), int(size_screen[1]), 3)) + 255).astype('uint8')
    
    # Get images
    # background = cv2.imread(background)
    img = cv2.imread(img_path)

    # Get height & width from image
    img_width, img_height = img.shape

    # Calculate midpoint of screen
    mid_x = int(size_screen[0])/2
    mid_y = int(size_screen[1])/2

    # Calculate width & height of image
    img_w_mid = int(img_width) / 2
    img_h_mid = int(img_height) / 2

    # Calculate offset point
    ref_x = int(mid_x + img_w_mid)
    ref_y = int(mid_y + img_h_mid)
    print(ref_x, ref_y)

    # Creating overlay
    bg = background.copy()
    bg[mid_y:mid_y + img_h_mid, mid_x:mid_x + img_w_mid] = img

    output = background.copy()

    # Blend images
    dst = cv2.addWeighted(overlay, 1, output, 1, 0, output)
    
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

    while (idx < 10):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        output.write(frame)
        cv2.imshow('frame', frame)
        
        delta = time.time() - prev_time 
        delta_since_last_change += delta
        prev_time = time.time()

        if delta_since_last_change >= TIME:
            delta_since_last_change = 0
            img_path = os.path.join(img_folder, images[idx])
            show_image(img_path)
            idx += 1 if idx < cnt else 10

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
