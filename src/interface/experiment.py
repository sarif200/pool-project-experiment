import cv2 as cv
import os, time

def cycle_images():
    TIME = 3
    t = TIME * 1000

    # Get file path
    scriptDir = os.path.dirname(__file__)
    dir_folder = os.path.join(scriptDir, '../img/')
    img_folder = os.path.abspath(dir_folder)
    print(img_folder)

    # Get files from folder
    images = os.listdir(img_folder)
    print(images)

    # # Count files in folder
    # img_length = len(images)
    # print(img_length)
    
    for image in images:
        img_path = os.path.join(img_folder, image)
        print(img_path)
        img = cv.imread(img_path, 0)
        cv.namedWindow("Image", cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty("Image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow('Image', img)
        key = cv.waitKey(t)
        if key == 27:#if ESC is pressed, exit loop
            cv.destroyAllWindows()
            break

    cv.destroyAllWindows()