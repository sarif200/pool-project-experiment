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
    #img = cv2.imread(img_path)

    #from https://stackoverflow.com/questions/31656366/cv2-imread-and-cv2-imshow-return-all-zeros-and-black-image 
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:     # we have an alpha channel
        a1 = ~img[:,:,3]        # extract and invert that alpha
        img = cv2.add(cv2.merge([a1,a1,a1,a1]), img)   # add up values (with clipping)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)    # strip alpha channel

    # Get height & width from image
    img_width, img_height = img.shape[:2]

    # Caclulate middle of screen
    yoff = round((mid_y - img_height)/2)
    #xoff = round((mid_x + img_width/8))
    xoff = round((mid_x + img_width/8))

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
    
    exel_currentdir_folder = os.path.join(final_folder_path, "results.xlsx")
    exel_project_folder = os.path.abspath(exel_currentdir_folder)
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
    cap = cv2.VideoCapture(0)

    fps = 15

    # Video Codec
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #output = cv2.VideoWriter(project_folder, fourcc, fps, (640, 480))
    
    prev_time = time.time()
    delta_since_last_change = 0
    
    #writer = pd.ExcelWriter(final_folder_path + 'results.xlsx', engine='openpyxl')
    writer = pd.ExcelWriter(exel_project_folder, engine='openpyxl')
    show_image(os.path.join(img_folder, images[0]))
    pupil_l_x = []
    pupil_l_y = []
    pupil_r_x = []
    pupil_r_y = []
    deltas = []
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
            #print(pupils)
            # output.write(frame)
            #if pupils[0] == None:
            #    pupil_l.append((0,0))
            #else:
            #    pupil_l.append(pupils[0])
            #if pupils[1] == None:
            #    pupil_r.append((0,0))
            #else:
            #    pupil_r.append(pupils[1])
            
            pupil_l_x.append(pupils[0][0])
            pupil_l_y.append(pupils[0][1])
            pupil_r_x.append(pupils[1][0])
            pupil_r_y.append(pupils[1][1])
            #print(pupil_l)
            #cv2.circle(frame,(int(pupils[0][0]),int(pupils[0][1])),10,(0, 255, 0),3)
            #cv2.circle(frame,(int(pupils[1][0]),int(pupils[1][1])),10,(255, 0, 0),3)
            #cv2.imshow('frame', frame)
        except Exception:
            pupil_l_x.append(0)
            pupil_l_y.append(0)
            pupil_r_x.append(0)
            pupil_r_y.append(0)
        
        delta = time.time() - prev_time 
        delta_since_last_change += delta
        prev_time = time.time()
        deltas.append(delta)
        if delta_since_last_change >= TIME:
            
            img_path = os.path.join(img_folder, images[idx])
            if images[idx] == "Twit(512,512).png":
                # Get screen size
                user32 = ctypes.windll.user32
                size_screen = user32.GetSystemMetrics(1), user32.GetSystemMetrics(0)
    
                # Create white background
                background = (np.zeros((int(size_screen[0]), int(size_screen[1]), 3)) + 255).astype('uint8')
                dst = background.copy()
                cv2.circle(dst,(512,512),10,(0,0,255),-1)
                cv2.namedWindow("Display", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Display', dst)
            else:

                show_image(img_path)
            
            print(images[idx])

            # Set data structure
            data = {
                    'Time Stamp': deltas,
                    'Pupil Left x': pupil_l_x,
                    'Pupil Left y': pupil_l_y,
                    'Pupil Right x': pupil_r_x,
                    'Pupil Right y': pupil_r_y 
                }
        
            # Convert to panda data frame
            df = pd.DataFrame(data, columns = ['Time Stamp', 'Pupil Left x','Pupil Left y', 'Pupil Right x','Pupil Right y'])
            
            
        
            #writer.save()
            # Convert & export to excel
            # Converted to 1 file with different sheet
            #df.to_excel(project_folder + 'results.xlsx', sheet_name=image, index=False)
            df.to_excel(writer, sheet_name=images[idx], index=False)
            writer.save()
            #export(delta_since_last_change, pupil_l, pupil_r, final_folder_path, images[idx])
            delta_since_last_change = 0
            pupil_l_x = []
            pupil_l_y = []
            pupil_r_x = []
            pupil_r_y = []
            deltas = []
            idx += 1 if idx < cnt else cnt

        key = cv2.waitKey(1)
        if key == 27:
            break
    writer.close()
    cap.release()
    #output.release()
    cv2.destroyAllWindows()
