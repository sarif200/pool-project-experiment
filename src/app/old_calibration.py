# Code for calibration tracking
from functions import *
import dlib
import cv2
import sys
import ctypes

# used https://github.com/michemingway/eye-writing-easy

# Input
cap_ID = 0

# Config
width = 1860
height = 1020
offset = (30, 30)

user32 = ctypes.windll.user32
size_screen = user32.GetSystemMetrics(1), user32.GetSystemMetrics(0)

print(size_screen)
# size_screen = (1080, 1920)
calibration_page = make_white_page(size = size_screen)

def calibration(final_folder_path, foldername):
    # Initialize
    cap = init_camera(cap_ID = cap_ID)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./src/assets/shape_predictor_68_face_landmarks.dat")

    # Calibration
    corners = [
        (offset), # Point 1
        (width + offset[0], height + offset[1]), # Point 2
        (width + offset[0], offset[1]), # Point 3
        (offset[0], height + offset[1])  # Point 4
    ]
    calibration_cut = []
    corner  = 0

    while (corner<4): # calibration of 4 corners

        ret, frame = cap.read()   # Capture frame
        frame = adjust_frame(frame)  # rotate / flip

        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray-scale to work with

        # draw circle
        cv2.circle(calibration_page, corners[corner], 40, (0, 255, 0), -1)

        # detect faces in frame
        faces = detector(gray_scale_frame)
        if len(faces)> 1:
            print('Please avoid multiple faces.')
            sys.exit()

        for face in faces:
            landmarks = predictor(gray_scale_frame, face) # find points in face

            # get position of right eye and display lines
            right_eye_coordinates = get_eye_coordinates(landmarks, [42, 43, 44, 45, 46, 47])
            display_eye_lines(frame, right_eye_coordinates, 'green')

        # define the coordinates of the pupil from the centroid of the right eye
        pupil_coordinates = np.mean([right_eye_coordinates[2], right_eye_coordinates[3]], axis = 0).astype('int')

        if cv2.waitKey(33) == ord('a'):
            calibration_cut.append(pupil_coordinates)
            print(pupil_coordinates)

            # visualize message
            cv2.putText(calibration_page, 'ok',
                        tuple(np.array(corners[corner])-5), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 0), 5)
            corner += 1

        # Display results
        # print(calibration_cut, '    len: ', len(calibration_cut))
        show_window('projection', calibration_page)
        # show_window('frame', cv2.resize(frame,  (640, 360)))

        if cv2.waitKey(113) == ord('q'):
            break

    # Process calibration
    x_cut_min, x_cut_max, y_cut_min, y_cut_max = find_cut_limits(calibration_cut)
    offset_calibrated_cut = [ x_cut_min, y_cut_min ]

    save_calibration(foldername, offset_calibrated_cut)

    cap.release()
    print('Calibration Finished')
    cv2.destroyAllWindows()
    start_message(final_folder_path)