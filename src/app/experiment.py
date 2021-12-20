import cv2
import os

def cycle_images():
    TIME = 3
    t = TIME * 1000 # transform to miliseconds

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
        img = cv2.imread(img_path, 0)
        cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Image', img)
        key = cv2.waitKey(t) # is miliseconds
        if key == 27: #if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()