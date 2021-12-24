# Code for calibration tracking
from calibrationFunctions import *
import dlib
import cv2
import sys
import time

# Input
camera_ID = 0

width = 1000
height = 500
offset = (100, 80)

def calibration():
    # Initialize
    camera = init_camera(camera_ID = camera_ID)

    # size_screen = (camera.get(cv2.CAP_PROP_FRAME_HEIGHT), camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    size_screen = (1920, 1080)

    calibration_page = make_white_page(size = size_screen)

    resize_eye_frame = 4.5 # scaling factor for window's size
    resize_frame = 0.3 # scaling factor for window's size

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

        ret, frame = camera.read()   # Capture frame
        frame = adjust_frame(frame)  # rotate / flip

        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray-scale to work with

        # messages for calibration
        cv2.putText(calibration_page, 'Calibration: look at the circle and press the a key', tuple((np.array(size_screen)/7).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 0, 255), 3)
        cv2.circle(calibration_page, corners[corner], 40, (0, 255, 0), -1)

        # detect faces in frame
        faces = detector(gray_scale_frame)
        if len(faces)> 1:
            print('Please avoid multiple faces.')
            sys.exit()

        for face in faces:
            # display_box_around_face(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (20, 40))

            landmarks = predictor(gray_scale_frame, face) # find points in face
            # display_face_points(frame, landmarks, [0, 68], color='red') # draw face points

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
                        tuple(np.array(corners[corner])-5), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 5)
            # to avoid is_blinking=True in the next frame
            time.sleep(0.3)
            corner += 1

        # print(calibration_cut, '    len: ', len(calibration_cut))
        show_window('projection', calibration_page)
        # show_window('frame', cv2.resize(frame,  (640, 360)))

        if cv2.waitKey(113) == ord('q'):
            break

    cv2.destroyAllWindows()

calibration()