import cv2
import os
#from datetime import datetime
import time

def cycle_images():
    # Get file path from current data directory
    foldername = "Test" # Get foldername from new_project file
    filename = "original_video.avi"
    scriptDir = os.path.dirname(__file__)
    currentdir_folder = os.path.join(scriptDir, '../data/', foldername, filename)
    project_folder = os.path.abspath(currentdir_folder)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Video Codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(project_folder, fourcc, 20.0, (640, 480))
    get_images()
    while True:
        ret, frame = cap.read()
        if ret:
            output.write(frame)
            cv2.imshow('frame', frame)
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if cv2.waitKey == 27: #if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()

def get_images():
    

    # Get file path
    scriptDir = os.path.dirname(__file__)
    imgdir_folder = os.path.join(scriptDir, '../img/')
    img_folder = os.path.abspath(imgdir_folder)
    # Get files from folder
    images = os.listdir(img_folder)
    # print(images)

    cnt = len(images)
    idx = 0
    while True:
        img_path = os.path.join(img_folder, images[idx])
        remaining_img = int(cnt - idx)
        print("Images remaining: " +  str(remaining_img) + " out of " + str(cnt), end="\r")
        img = cv2.imread(img_path)

        cv2.namedWindow("Display", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Display', img)

        if cv2.waitKey(1000) >= 0:
            break

        idx += 1
        if idx >= cnt:
            break

        if cv2.waitKey == 27: #if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()

def show_image(img_path):
    img = cv2.imread(img_path)

    cv2.namedWindow("Display", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Display', img)

def cycle_images2():
    # Get file path from current data directory
    foldername = "Test" # Get foldername from new_project file
    filename = "original_video.avi"
    scriptDir = os.path.dirname(__file__)
    currentdir_folder = os.path.join(scriptDir, '../data/', foldername, filename)
    project_folder = os.path.abspath(currentdir_folder)

    TIME = 3
    t = TIME * 1000 # transform to miliseconds

    # Get file path
    scriptDir = os.path.dirname(__file__)
    imgdir_folder = os.path.join(scriptDir, '../img/')
    img_folder = os.path.abspath(imgdir_folder)
    # Get files from folder
    images = os.listdir(img_folder)
    # print(images)

    cnt = len(images)
    idx = 1
    print(os.path.dirname(__file__))
    cap = cv2.VideoCapture('../assets/video.mp4', cv2.CAP_DSHOW)

    # Video Codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(project_folder, fourcc, 20.0, (640, 480))
    
    prev_time = time.time()
    delta_since_last_change = 0

    show_image(os.path.join(img_folder, images[0]))

    while True:
        ret, frame = cap.read()
        if ret:
            output.write(frame)
            cv2.imshow('frame', frame)
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        #img_path = os.path.join(img_folder, images[idx])
        
        #img = cv2.imread(img_path)

        #cv2.namedWindow("Display", cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #cv2.imshow('Display', img)
        
        delta = time.time() - prev_time 
        delta_since_last_change += delta
        prev_time = time.time()
        print(delta)
        if delta_since_last_change >= TIME:
            delta_since_last_change = 0
            img_path = os.path.join(img_folder, images[idx])
            show_image(img_path)
            idx += 1 if idx +1 < cnt else 0
            
        #if cv2.waitKey(1000) >= 0:
            #break

        #idx += 1
        #if idx >= cnt:
            #break


        if cv2.waitKey == 27: #if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()

cycle_images2()