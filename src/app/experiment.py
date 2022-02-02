import cv2
import os
import time
import sys
import ctypes
import numpy as np
import pandas as pd
from tracking import analytical_tracker

scriptDir = os.path.dirname(__file__)
tracking_folder = os.path.join(scriptDir, '../tracking/')
path = os.path.abspath(tracking_folder)

sys.path.append(path)

tracker = analytical_tracker()
#gaze = gaze_tracker

def export(delta_since_last_change, pupil_l, pupil_r, project_folder, image):
        # Set data structure
        data = {
                'Time Stamp': delta_since_last_change,
                'Pupil Left': pupil_l,
                'Pupil Right': pupil_r
            }
        
        # Convert to panda data frame
        df = pd.DataFrame(data, columns = ['Time Stamp', 'Pupil Left', 'Pupil Right'])
        path = r"C:\Users\fedel\Desktop\excelData\PhD_data.xlsx"
        writer = pd.ExcelWriter(project_folder, engine='openpyxl')
        
        writer.save()
        # Convert & export to excel
        # Converted to 1 file with different sheet
        #df.to_excel(project_folder + 'results.xlsx', sheet_name=image, index=False)
        df.to_excel(writer, sheet_name=image, index=False)
        writer.save()

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
    writer = pd.ExcelWriter(final_folder_path + 'results.xlsx', engine='openpyxl')
    show_image(os.path.join(img_folder, images[0]))
    pupil_l = []
    pupil_r = []
    while (idx < cnt):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # pupils = tracker.detect_in_frame(tracker,frame)
        #pupils = gaze.track_in_frame(gaze,frame)

        frame = cv2.flip(frame, 1)
        tracker.refresh(frame)
        try:
            pupils = (tracker.pupil_left_screen_coords(),tracker.pupil_right_screen_coords())
            # output.write(frame)
            pupil_l.append( (int(pupils[0][0]),int(pupils[0][1])))
            pupil_r.append((int(pupils[1][0]),int(pupils[1][1])))

            cv2.circle(frame,(int(pupils[0][0]),int(pupils[0][1])),10,(0, 255, 0),3)
            cv2.circle(frame,(int(pupils[1][0]),int(pupils[1][1])),10,(255, 0, 0),3)
            cv2.imshow('frame', frame)
        except Exception:
            pupil_l.append((0,0))
            pupil_r.append((0,0))
        
        delta = time.time() - prev_time 
        delta_since_last_change += delta
        prev_time = time.time()

        if delta_since_last_change >= TIME:
            delta_since_last_change = 0
            img_path = os.path.join(img_folder, images[idx])
            show_image(img_path)
            print(images[idx])

            # Set data structure
            data = {
                    'Time Stamp': delta_since_last_change,
                    'Pupil Left': pupil_l,
                    'Pupil Right': pupil_r
                }
        
            # Convert to panda data frame
            df = pd.DataFrame(data, columns = ['Time Stamp', 'Pupil Left', 'Pupil Right'])
            
            
        
            #writer.save()
            # Convert & export to excel
            # Converted to 1 file with different sheet
            #df.to_excel(project_folder + 'results.xlsx', sheet_name=image, index=False)
            df.to_excel(writer, sheet_name=images[idx], index=False)
            writer.save()
            #export(delta_since_last_change, pupil_l, pupil_r, final_folder_path, images[idx])
            pupil_l = []
            pupil_r = []

            idx += 1 if idx < cnt else cnt

        key = cv2.waitKey(1)
        if key == 27:
            break
    writer.close()
    cap.release()
    output.release()
    cv2.destroyAllWindows()
