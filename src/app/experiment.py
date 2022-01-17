import cv2
import os
import time
import pandas as pd

def show_image(img_path):
    img = cv2.imread(img_path)

    cv2.namedWindow("Display", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Display', img)
    
def export(delta_since_last_change, pupil_l, pupil_r, project_folder, images):
    # Set data structure
    data = {
            'Time Stamp': delta_since_last_change,
            'Pupil Left': pupil_l,
            'Pupil Right': pupil_r
           }
    
    # Convert to panda data frame
    df = pd.DataFrame(data, colums = ['Time Stamp', 'Pupil Left', 'Pupil Right'])
    
    # Convert & export to excel
    # Converted to 1 file with different sheet
    df.to_excel(project_folder + 'results.xlsx', sheet_name=images[idx], index=False)

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

        output.write(frame)
        cv2.imshow('frame', frame)
        
        delta = time.time() - prev_time 
        delta_since_last_change += delta
        prev_time = time.time()

        if delta_since_last_change >= TIME:
            delta_since_last_change = 0
            img_path = os.path.join(img_folder, images[idx])
            show_image(img_path)
            export(delta_since_last_change, pupil_l, pupil_r, project_folder, images)
            #idx += 1 if idx < cnt else cnt
            if idx < cnt:
                idx += 1
            else:
                if delta_since_last_change >= TIME:
                    idx = cnt
                    break

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
