import cv2
import numpy as np
import dlib
import os 

# used https://github.com/MCodez/PUPIL-Detection-using-OpenCV

#face predictor
# C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
#300 faces In-the-wild challenge: Database and results. 
#Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


filename = './src/assets/video.mp4'

if filename == '':
    filename = 0

cap = cv2.VideoCapture(filename)

cap.set(cv2.CAP_PROP_FPS,60)

if (cap.isOpened()== False):
    print("Error opening video stream or file")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./src/assets/shape_predictor_68_face_landmarks.dat')

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 0

params.filterByConvexity = False
params.minConvexity = 0.1


blob_detector = cv2.SimpleBlobDetector_create(params)


left = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
right = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def calc_pupil(landmarks,side):
    x_sum =0
    y_sum =0
    for i in side:
        x_sum += landmarks.part(i).x 
        y_sum += landmarks.part(i).y
    coord = (int(x_sum/len(side)),int(y_sum/len(side)))
    return coord

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    return img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)return img


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        pass

    landmarks = predictor(gray, face)
    left_point = (landmarks.part(36).x, landmarks.part(36).y)
    right_point = (landmarks.part(39).x, landmarks.part(39).y)

    center_top = midpoint(landmarks.part(37), landmarks.part(38))
    center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

    img_l_eye = gray[center_top[1]:center_bottom[1],landmarks.part(36).x:landmarks.part(39).x]

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    
    #left_cord = calc_pupil(landmarks,left)
    #right_cord = calc_pupil(landmarks,right)

    #cv2.circle(frame, left_cord, 10, (255), -1)
    #cv2.circle(frame, right_cord, 10, (255), -1)

    #cv2.rectangle(frame,(landmarks.part(36).x,center_top[1]),(landmarks.part(39).x,center_bottom[1]),(0,255,0),3)

    #pupil = blob_detector.detect(img_l_eye)
    #cut_eyebrow = cut_eyebrows(img_l_eye)
    blur = cv2.GaussianBlur(img_l_eye,(5,5),0)
    #inv = cv2.bitwise_not(blur)
    #thresh = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(blur,55,255,cv2.THRESH_BINARY)
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(thresh1,kernel,iterations = 1)
    cnts, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for  cnt in cnts:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        print((x,y))
        #cv2.circle(erosion,(int(x),int(y)),10,(0,255,0),3)
    
    print(cnts)
    #cv2.drawContours(erosion, cnts, -1, (0,255,0), 3)
    #im_with_keypoints = cv2.drawKeypoints(img_l_eye, pupil, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #ret,im_tresh = cv2.threshold(img_l_eye,55,255,cv2.THRESH_BINARY)

    #im_tresh = cv2.adaptiveThreshold(img_l_eye,80,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

    #im_tresh = cut_eyebrows(img_l_eye)
    pupil = blob_detector.detect(erosion)
    im_with_keypoints = cv2.drawKeypoints(erosion, pupil, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for keyPoint in pupil:
        x = keyPoint.pt[0]
        y = keyPoint.pt[1]
        x += landmarks.part(36).x
        y += center_top[1]
        s = keyPoint.size
        cv2.circle(frame,(int(x),int(y)),10,(0,255,0),3)


    print(pupil)

    cv2.imshow("Frame",frame )
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
