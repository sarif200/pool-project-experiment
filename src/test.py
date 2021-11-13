import cv2
import dlib as dl
import numpy as np

def face_cord(r):
    return  r.left(), r.right(),r.top(), r.bottom()

def keypoints_to_np(keypoints, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (keypoints.part(i).x, keypoints.part(i).y)
    return coords

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    return img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)return img

def eye_points(keypoints , side):
    points = [keypoints[i] for i in side]
    return np.array(points, dtype=np.int32)

def find_min_max_points(points,img_width,img_height):
    x_max = 0
    y_max = 0
    x_min = img_width
    y_min = img_height
    for (x,y) in points:
        x_max = max(x_max,x)
        y_max = max(y_max,y)
        x_min = min(x_min,x)
        y_min = min(y_min,y)
    return x_max,y_max,x_min,y_min

def detect_pupil(keypoints,side):
    points = eye_points(keypoints,side)
    #x_max,y_max,x_min,y_min = find_min_max_points(points)
    #img_eye = img_gray[y_min:y_max,x_min:x_max] 
    x_sum =0
    y_sum =0
    for (x,y) in points:
        x_sum += x 
        y_sum += y
    coord = (int(x_sum/len(points)),int(y_sum/len(points)))
    return coord

    

#def detect_pupil_old(keypoints,side):
    points = eye_points(keypoints,side)
    x_max,y_max,x_min,y_min = find_min_max_points(points)
    img_eye = img_gray[y_min:y_max,x_min:x_max] 
    _ ,thresh = cv2.threshold(img_eye, 82, 255, cv2.THRESH_BINARY)
    #thresh = cv2.erode(thresh, None, iterations=2) #1
    #thresh = cv2.dilate(thresh, None, iterations=4) #2
    #thresh = cv2.medianBlur(thresh, 5)
    blobs = blob_detector.detect(img_eye)
    circle_point_global = 0
    cv2.imshow(str(side),thresh)
    if len(blobs) > 0:
        X = blobs[0].pt[0] 
        Y = blobs[0].pt[1]
        circle_point_global = (int(X+x_min),int(Y+y_min))
        #img_eye = cv2.circle(img_eye, (int(X),int(Y)), 10, (255), -1)
    return blobs, circle_point_global



#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


left = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
right = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye



img = cv2.imread('foto/DSC05998.jpeg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.circle(img, (20,80), 10, (255), -1)
keypoint_predictor = dl.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_detector = dl.get_frontal_face_detector()
params = cv2.SimpleBlobDetector_Params()	 
params.minThreshold = 1
params.maxThreshold = 200
blob_detector = cv2.SimpleBlobDetector_create(params)
#img_width = img.shape[1]
#img_height= img.shape[0]

cap = cv2.VideoCapture(0)

	 

	# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

	 

	# Read until video is completed

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        rects = face_detector(frame, 1)
        for (i, rect) in enumerate(rects):
            X,Y,x,y = face_cord(rect)

            eyes = eye_cascade.detectMultiScale(img_gray[X:x,Y:y])
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow('Frame',frame)

        """for (i, rect) in enumerate(rects):
            keypoints = keypoint_predictor(frame, rect)
            keypoints = keypoints_to_np(keypoints)
            for (x, y) in keypoints:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    #points = eye_points(keypoints,left)
    #print(points)
    
            circle_point_global_left = detect_pupil(keypoints,left)
            circle_point_global_right = detect_pupil(keypoints,right)
    
            cv2.circle(frame, circle_point_global_left, 5, (255), 1)
            cv2.circle(frame, circle_point_global_right, 5, (255), 1)
            cv2.imshow('Frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    else:
        break

"""
	 



def main():
    

    rects = face_detector(img, 1) # rects contains all the faces detected
    # loop over the bounding boxes

    


    for (i, rect) in enumerate(rects):
        keypoints = keypoint_predictor(img, rect)
        keypoints = keypoints_to_np(keypoints)
        for (x, y) in keypoints:
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    #points = eye_points(keypoints,left)
    #print(points)
    
        circle_point_global_left = detect_pupil(keypoints,left)
        circle_point_global_right = detect_pupil(keypoints,right)
    
        cv2.circle(img, circle_point_global_left, 10, (255), -1)
        cv2.circle(img, circle_point_global_right, 10, (255), -1)
    
'''''
    #cv2.fillConvexPoly(img, points, (255,0,0)) 
    
    #cv2.rectangle(img,(x_max,y_max),(x_min,y_min),(255,0,0),-1)  
    img_eye = img_gray[y_min:y_max,x_min:x_max] 
    blobs = blob_detector.detect(img_eye)
    if len(blobs) > 0:
        X = blobs[0].pt[0] 
        Y = blobs[0].pt[1]
        
        circle_point = (int(X+x_min),int(Y+y_min))
        print(circle_point)
        img = cv2.circle(img, circle_point, 10, (255), -1)
        img_eye = cv2.circle(img_eye, (int(X),int(Y)), 10, (255), -1)
    
    #img_eye = cv2.drawKeypoints(img_eye, blobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(str(i),img_eye)

'''
#main()
#cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
#cv2.createTrackbar('slider', "image", 0, 100, on_change)
#cv2.moveWindow("image", 900,-900)
#cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#cv2.imshow('image',img)
#cv2.imwrite('test.jpg', img)
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()


#used sources:
#https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
#https://www.pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
#https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6
#https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6