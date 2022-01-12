import cv2
import os
import time
import ctypes

def show_image(img_path):
    # Get screen size
    ser32 = ctypes.windll.user32
    size_screen = user32.GetSystemMetrics(1), user32.GetSystemMetrics(0)
    
    # Calculate midpoint of screen
    mid_x, mid_y = int(size_screen[0])/2, int(size_screen[1])/2
    
    # Create white background
    background = (np.zeros((int(size_screen[0]), int(size_screen[1]), 3)) + 255).astype('uint8')
    
    # Get images
    background = cv2.imread(background, cv2.IMREAD_COLOR)
    img = cv2.imread(img_path)
    
    # Blend images
    dst = cv2.addWeighted(background, 1, img, 1, 0.0)
    
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
