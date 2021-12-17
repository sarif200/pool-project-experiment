import cv2
import numpy as np
import dlib
import os 

# used https://github.com/MCodez/PUPIL-Detection-using-OpenCV

#face predictor
# C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
#300 faces In-the-wild challenge: Database and results. 
#Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


# l = left (-x) and r = right (+x) 



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


def subtract_tuples(tuple1,tuple2):
    return (tuple1[0]-tuple2[0],tuple1[1]-tuple2[1])

def add_tuples(tuple1,tuple2):
    return (tuple1[0]+tuple2[0],tuple1[1]+tuple2[1])

def abs_tuples(tuple):
    return (abs(tuple[0]),abs(tuple[1]))

class pupil_tracker:
    left = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
    right = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye

    #global declerations
    face_detector = dlib.get_frontal_face_detector()
    predictor
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    blob_detector = cv2.SimpleBlobDetector_create(params)

    #get the middle of two points in 2d  space
    def midpoint(p1, p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

    #cut the top 4th of a image
    def cut_eyebrows(img):
        height, width = img.shape[:2]
        eyebrow_h = int(height / 4)
        return img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)return img

    #calculate the midle of the landmarks this aproximates in most case the place of the pupil
    #is not accurate when the eye is recorded from extreem angles
    def calc_pupil(landmarks,side):
        x_sum =0
        y_sum =0
        for i in side:
            x_sum += landmarks.part(i).x 
            y_sum += landmarks.part(i).y
        coord = (int(x_sum/len(side)),int(y_sum/len(side)))
        return coord
    #this wil find and return the most probable place of the pupil 
    def detect_pupil(eye,backup): # backup is a point that is returned by a calc pupil func

        # edit the images so it the blob detector wil work better
        blur = cv2.GaussianBlur(eye,(5,5),0)
        ret,thresh1 = cv2.threshold(blur,55,255,cv2.THRESH_BINARY)
        kernel = np.ones((2,2),np.uint8)
        erosion = cv2.erode(thresh1,kernel,iterations = 1)
        

        #im_tresh = cut_eyebrows(img_l_eye)
        #blob detection 
        pupil = blob_detector.detect(erosion)
        #im_with_keypoints = cv2.drawKeypoints(erosion, pupil, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #if no blobs are found we  return the backup
        if len(pupil) ==0:
            return backup

        #if  we find keypoints we select the closed one to the     
        selected_keypoint = (0,0)
        selected_keypoint_len_cubed = 1000000 #random number that is hopfully larger then all the real values #Needs fix
        for keyPoint in pupil:
            x = keyPoint.pt[0]
            y = keyPoint.pt[1]

            len_cubed = (backup[0]-x)**2 + (backup[1]-y)**2 # no square root needed because every distance is calculated with pythagoras and only the diference is important 

            if len_cubed<selected_keypoint_len_cubed:
                selected_keypoint = (x,y)
                selected_keypoint_len_cubed = len_cubed


            #--------transform to full screen------------
            #x += landmarks.part(36).x 
            #y += center_top[1]
            #s = keyPoint.size
            #cv2.circle(frame,(int(x),int(y)),10,(0,255,0),3)


        #print(pupil)

        #cv2.imshow("Frame",erosion )
        return selected_keypoint
    
    def __init__(self, shape_predictor_file = './src/assets/shape_predictor_68_face_landmarks.dat'):
        self.predictor = dlib.shape_predictor(shape_predictor_file)
    

    def detect_in_frame(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = detector(gray)
        for face in faces:
            #x, y = face.left(), face.top()
            #x1, y1 = face.right(), face.bottom()
            #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            pass

        landmarks = predictor(gray, face)
        #left_point = (landmarks.part(36).x, landmarks.part(36).y)
        #right_point = (landmarks.part(39).x, landmarks.part(39).y)

        center_top_l = self.midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom_l = self.midpoint(landmarks.part(41), landmarks.part(40))

        center_top_r = self.midpoint(landmarks.part(43), landmarks.part(44))
        center_bottom_r = self.midpoint(landmarks.part(47), landmarks.part(46))
        

        img_l_eye = gray[center_top_l[1]:center_bottom_l[1],landmarks.part(36).x:landmarks.part(39).x]
        img_r_eye = gray[center_top_r[1]:center_bottom_r[1],landmarks.part(42).x:landmarks.part(45).x]

        backup_l = self.calc_pupil(landmarks,left)
        backup_r = self.calc_pupil(landmarks,right)
        #--------transform to eye space------------
        #backup_l = (abs(backup_l[0]-landmarks.part(36).x),abs(backup_l[1]-center_top_l[1])) 
        #backup_r = (abs(backup_r[0]-landmarks.part(42).x),abs(backup_r[1]-center_top_r[1])) 
        backup_l = abs_tuples(subtract_tuples(backup_l,(landmarks.part(36).x,center_top_l[1])))
        backup_r = abs_tuples(subtract_tuples(backup_r,(landmarks.part(42).x,center_top_r[1])))
        
        pupil_l = self.detect_pupil(img_l_eye,backup_l)
        pupil_r = self.detect_pupil(img_r_eye,backup_r)

        #--------transform to screen space------------
        pupil_l = add_tuples(pupil_l,(landmarks.part(36).x,center_top_l[1]))
        pupil_r = add_tuples(pupil_r,(landmarks.part(42).x,center_top_r[1]))
        
        #blur = cv2.GaussianBlur(img_l_eye,(5,5),0)
        #ret,thresh1 = cv2.threshold(blur,55,255,cv2.THRESH_BINARY)
        #kernel = np.ones((2,2),np.uint8)
        #erosion = cv2.erode(thresh1,kernel,iterations = 1)
        
        
        #im_tresh = cut_eyebrows(img_l_eye)
        #pupil = blob_detector.detect(erosion)
        #im_with_keypoints = cv2.drawKeypoints(erosion, pupil, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #if len(pupil) ==0:
            #return calc_pupil(landmarks,left)

        #for keyPoint in pupil:
            #x = keyPoint.pt[0]
            #y = keyPoint.pt[1]
            #--------transform to full screen------------
            #x += landmarks.part(36).x 
            #y += center_top[1]
            #s = keyPoint.size
            #cv2.circle(frame,(int(x),int(y)),10,(0,255,0),3)


        #print(pupil)

        #cv2.imshow("Frame",erosion )
        
        return pupil_l,pupil_r

tracker = pupil_tracker

while True:
    _, frame = cap.read()
    
    pupils = tracker.detect_in_frame(tracker,frame)
    cv2.circle(frame,(int(pupils[0][0]),int(pupils[0][1])),10,(0,255,0),3)
    cv2.circle(frame,(int(pupils[1][0]),int(pupils[1][1])),10,(0,255,0),3)
    cv2.imshow("Frame",frame )
    key = cv2.waitKey(1)
    if key == 27:
        break
   


while False:
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
