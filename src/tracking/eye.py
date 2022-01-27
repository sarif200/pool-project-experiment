import math
import numpy as np
import cv2
from pupil import Pupil


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration,translation,normal):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        self.origin_3d = None
        self.pupil_3d = None
        self.gaze_dir_3d = None
        self.screen_cord = None
        
        self._analyze(original_frame, landmarks, side, calibration)
        #print(self.to_3d_vec_from_cam(self.origin,calibration))
        #print(isect_line_plane_v3((0,0,0),self.to_3d_vec_from_cam(self.origin,calibration),translation,normal))
        #self.origin_3d = isect_line_plane_v3((0,0,0),self.to_3d_vec_from_cam((self.origin[0]+self.center[0],self.origin[1]+self.center[1]),calibration),translation,normal)
        #self.origin_3d = np.array([(self.origin_3d[0],self.origin_3d[1],self.origin_3d[2])]) - cv2.normalize(normal,0,1) * 24
        
        self.origin_3d = self.to_3d_on_plane((self.origin[0]+self.center[0],self.origin[1]+self.center[1]),translation,normal,calibration) - cv2.normalize(normal,0,1) * 24
        self.pupil_3d = self.to_3d_on_plane((self.origin[0]+self.pupil.x,self.origin[1]+self.pupil.y),translation,normal,calibration)
        self.gaze_dir_3d = self.origin_3d - self.pupil_3d 
        print(self.origin_3d[0],self.pupil_3d[0])
        self.screen_cord = isect_line_plane_v3(self.origin_3d[0],self.pupil_3d[0],(0,0,0),(0,0,1))
        print(self.screen_cord)
        #(end_point2D, jacobian) = cv2.projectPoints((screen_cord[0],screen_cord[1],screen_cord[2]), (0,0,0), (0,0,0), calibration.camera_matrix, np.zeros((4,1)))
        self.origin_3d_projected = self.screen_cord
        #print("2dprojection: ",end_point2D)

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """Calculates a ratio that can indicate whether an eye is closed or not.
        It's the division of the width of the eye, by its height.

        Arguments:
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)

        Returns:
            The computed ratio
        """
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)

    def to_3d_vec_from_cam(self,point,calibration):
        point = (point[0],point[1],1)
        #cv2.invert()
        matrix = cv2.invert(calibration.camera_matrix)
        
        return (matrix[1][0][0]*point[0] + matrix[1][0][1]*point[1] + matrix[1][0][2]*point[2],matrix[1][1][0]*point[0] + matrix[1][1][1]*point[1] + matrix[1][1][2]*point[2],matrix[1][2][0]*point[0] + matrix[1][2][1]*point[1] + matrix[1][2][2]*point[2])

    def to_3d_on_plane(self,point,translation,normal,calibration):
        ray_vec = self.to_3d_vec_from_cam(point,calibration)
        point_3d = isect_line_plane_v3((0,0,0),ray_vec,translation,normal)
        return np.array([(point_3d[0],point_3d[1],point_3d[2])])





    #from https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
    # intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)
    
    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)

    # The segment is parallel to plane.
    return None

# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
    )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
    )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )