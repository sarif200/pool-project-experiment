# Functions for calibration
import os
import cv2
import numpy as np

dict_color = {'green': (0,255,0),
              'blue':(255,0,0),
              'red': (0,0,255),
              'yellow': (0,255,255),
              'white': (255, 255, 255),
              'black': (0,0,0)}

def init_camera(camera_ID):
    camera = cv2.VideoCapture(camera_ID)
    return camera

def make_white_page(size):
    page = (np.zeros((int(size[0]), int(size[1]), 3)) + 255).astype('uint8')
    return page

def adjust_frame(frame):
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = cv2.flip(frame, 1)
    return frame

def show_window(title_window, window):
    cv2.namedWindow(title_window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(title_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(title_window,window)

def shut_off(camera):
    camera.release() # When everything done, release the capture
    cv2.destroyAllWindows()

def get_eye_coordinates(landmarks, points):
    x_left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
    x_right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)

    y_top = half_point(landmarks.part(points[1]), landmarks.part(points[2]))
    y_bottom = half_point(landmarks.part(points[5]), landmarks.part(points[4]))

    return x_left, x_right, y_top, y_bottom

def half_point(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def display_eye_lines(img, coordinates, color):
    cv2.line(img, coordinates[0], coordinates[1], dict_color[color], 2)
    cv2.line(img, coordinates[2], coordinates[3], dict_color[color], 2)

def display_face_points(img, landmarks, points_to_draw, color):
    for point in range(points_to_draw[0], points_to_draw[1]):
        x = landmarks.part(point).x
        y = landmarks.part(point).y
        cv2.circle(img, (x, y), 4, dict_color[color], 2)

def find_cut_limits(calibration_cut):
    x_cut_max = np.transpose(np.array(calibration_cut))[0].max()
    x_cut_min = np.transpose(np.array(calibration_cut))[0].min()
    y_cut_max = np.transpose(np.array(calibration_cut))[1].max()
    y_cut_min = np.transpose(np.array(calibration_cut))[1].min()

    return x_cut_min, x_cut_max, y_cut_min, y_cut_max

def save_calibration(foldername, offset_calibrated_cut):
    filename = "offset.txt"
    scriptDir = os.path.dirname(__file__)
    currentdir_folder = os.path.join(scriptDir, '../data/', foldername, filename)
    file = os.path.abspath(currentdir_folder)
    document = open(file, "a+") # Will open & create if file is not found

    document.write(str(offset_calibrated_cut))

    print("Succesfully writed to file")