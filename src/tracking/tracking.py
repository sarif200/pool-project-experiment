import cv2
import numpy as np
import dlib
import os 
import ctypes
import sys

# used https://github.com/MCodez/PUPIL-Detection-using-OpenCV

#face predictor
# C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
#300 faces In-the-wild challenge: Database and results. 
#Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


#https://www.cs.ru.ac.za/research/g09W0474/Thesis.pdf

# l = left (-x) and r = right (+x) 

filename = './src/assets/video.mp4'
filename = ''

if filename == '':
    file = 0
else:
    scriptDir = os.path.dirname(__file__)
    assets_folder = os.path.join(scriptDir, '../assets/', filename)
    
    file = os.path.abspath(filename)
    print(file)
   
cap = cv2.VideoCapture(file)

cap.set(cv2.CAP_PROP_FPS,60)

if (cap.isOpened()== False):
    print("Error opening video stream or file")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./src/assets/shape_predictor_68_face_landmarks.dat')

detector_params = cv2.SimpleBlobDetector_Params()

detector_params.minThreshold = 0

detector_params.filterByArea = True
detector_params.maxArea = 1500

detector_params.filterByConvexity = False
detector_params.minConvexity = 0.1


#blob_detector = cv2.SimpleBlobDetector_create(detector_params)


#left = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
#right = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye
'''
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

'''
def subtract_tuples(tuple1,tuple2):
    return (tuple1[0]-tuple2[0],tuple1[1]-tuple2[1])

def add_tuples(tuple1,tuple2):
    return (tuple1[0]+tuple2[0],tuple1[1]+tuple2[1])

def abs_tuples(tuple):
    return (abs(tuple[0]),abs(tuple[1]))

def multi_tuples(tuple1,tuple2):
    return (tuple1[0]*tuple2[0],tuple1[1]*tuple2[1])

def div_tuples(tuple1,tuple2):
    return (tuple1[0]/tuple2[0],tuple1[1]/tuple2[1])


class pupil_tracker:
    left = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
    right = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye
    start_pos = 0

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


    def image_blob_detection(self,image):
        blur = cv2.GaussianBlur(image,(9,9),0)
        ret,thresh1 = cv2.threshold(blur,8,255,cv2.THRESH_BINARY)
        kernel = np.ones((2,2),np.uint8)
        erosion = cv2.erode(thresh1,kernel,iterations = 1)
        cv2.imshow("test",erosion)
        return self.blob_detector.detect(erosion)
    
    #this wil find and return the most probable place of the pupil 
    def detect_pupil(self,eye,backup): # backup is a point that is returned by a calc pupil func
        #return backup
        # edit the images so it the blob detector wil work better
        #inverted = np.invert(eye)
        blur = cv2.GaussianBlur(eye,(9,9),0)
        ret,thresh1 = cv2.threshold(blur,10,255,cv2.THRESH_BINARY)
        #kernel = np.ones((2,2),np.uint8) 
        #erosion = cv2.erode(thresh1,kernel,iterations = 1)
        #closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
        blur = cv2.GaussianBlur(eye,(9,9),0)
        ret,thresh1 = cv2.threshold(blur,8,255,cv2.THRESH_BINARY)
        kernel = np.ones((2,2),np.uint8)
        erosion = cv2.erode(thresh1,kernel,iterations = 1)

        pupil = cv2.HoughCircles(erosion,cv2.HOUGH_GRADIENT,1,20,param1=300,param2=0.8,minRadius=0,maxRadius=0)
        print(pupil)
        #contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(erosion, contours, -1, (0,255,0), 3)
        #cv2.imshow("test",erosion)
        #print(hierarchy)
        #im_tresh = cut_eyebrows(img_l_eye)
        #blob detection 
        pupil = self.image_blob_detection(self,eye)
        
        #im_with_keypoints = cv2.drawKeypoints(erosion, pupil, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #pupil = np.uint16(np.around(pupil))

        #print(pupil)
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
        scriptDir = os.path.dirname(__file__)
        assets_folder = os.path.join(scriptDir, '../assets/', shape_predictor_file)
        file = os.path.abspath(assets_folder)
        #print(assets_folder)
        self.predictor = dlib.shape_predictor(file)
    

    def detect_in_frame(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = detector(gray)
        if len(faces) ==0:
            return ((0,0),(0,0))
        if len(faces)> 1:
            print('Please avoid multiple faces.')
            sys.exit()
        #for face in faces:
            #x, y = face.left(), face.top()
            #x1, y1 = face.right(), face.bottom()
            #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            #pass

        landmarks = predictor(gray, faces[0])
        #left_point = (landmarks.part(36).x, landmarks.part(36).y)
        #right_point = (landmarks.part(39).x, landmarks.part(39).y)

        center_top_l = self.midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom_l = self.midpoint(landmarks.part(41), landmarks.part(40))

        center_top_r = self.midpoint(landmarks.part(43), landmarks.part(44))
        center_bottom_r = self.midpoint(landmarks.part(47), landmarks.part(46))
        

        img_l_eye = gray[center_top_l[1]:center_bottom_l[1],landmarks.part(36).x:landmarks.part(39).x]
        img_r_eye = gray[center_top_r[1]:center_bottom_r[1],landmarks.part(42).x:landmarks.part(45).x]

        backup_l_global = self.calc_pupil(landmarks,self.left)
        backup_l_global = div_tuples(add_tuples(center_top_l,center_bottom_l),(2,2))
        backup_r_global = self.calc_pupil(landmarks,self.right)
        backup_r_global = div_tuples(add_tuples(center_top_r,center_bottom_r),(2,2))
        #--------transform to eye space------------
        #backup_l = (abs(backup_l[0]-landmarks.part(36).x),abs(backup_l[1]-center_top_l[1])) 
        #backup_r = (abs(backup_r[0]-landmarks.part(42).x),abs(backup_r[1]-center_top_r[1])) 
        backup_l = abs_tuples(subtract_tuples(backup_l_global,(landmarks.part(36).x,center_top_l[1])))
        backup_r = abs_tuples(subtract_tuples(backup_r_global,(landmarks.part(42).x,center_top_r[1])))
        
        pupil_l = self.detect_pupil(self,img_l_eye,backup_l)
        pupil_r = self.detect_pupil(self,img_r_eye,backup_r)

        #--------transform to screen space------------
        pupil_l = add_tuples(pupil_l,(landmarks.part(36).x,center_top_l[1]))
        pupil_r = add_tuples(pupil_r,(landmarks.part(42).x,center_top_r[1]))
        
        #print(backup_l_global,backup_r_global)
        #print(pupil_l,pupil_r)
        
        middle = self.midpoint(landmarks.part(40),landmarks.part(43))#div_tuples(add_tuples((landmarks.part(40).x,landmarks.part(40).y),(landmarks.part(43).x,landmarks.part(43).y)),(2,2))
        if self.start_pos == 0:

            self.start_pos = middle
        else:
            difference = subtract_tuples(self.start_pos,middle)
            pupil_l = add_tuples(difference,pupil_l)
            pupil_r = add_tuples(difference,pupil_r)
        
        return pupil_l,pupil_r



class gaze_tracker:
    
    pupilTracker = pupil_tracker
    width = 1366#1860
    height = 768#1020
    offset = (40, 40)

    user32 = ctypes.windll.user32
    size_screen = user32.GetSystemMetrics(1), user32.GetSystemMetrics(0)

    calibration_page = (np.zeros((int(size_screen[0]), int(size_screen[1]), 3)) + 255).astype('uint8')

    #x_cut_min_l, x_cut_max_l, y_cut_min_l, y_cut_max_l
    offset_calibrated_cut_left = []
    #x_cut_min_r, x_cut_max_r, y_cut_min_r, y_cut_max_r
    offset_calibrated_cut_right = []

    cut_frame_l = 0 

    cut_frame_r = 0

    scale_l = 0 
    scale_r = 0

    # Calibration
    corners = [
        (offset), # Point 1
        (width - offset[0], height - offset[1]), # Point 2
        (width - offset[0], offset[1]), # Point 3
        (offset[0], height - offset[1])  # Point 4
    ]
    
    
    

    def save_calibration(foldername, offset_calibrated_cut_left,offset_calibrated_cut_right):
        filename = "offset.txt"
        scriptDir = os.path.dirname(__file__)
        currentdir_folder = os.path.join(scriptDir, '../data/', foldername, filename)
        file = os.path.abspath(currentdir_folder)
        document = open(file, "a+") # Will open & create if file is not found

        document.write(str(offset_calibrated_cut_left))

        print("Succesfully writen to file")

    def find_cut_limits(calibration_cut):
        x_cut_max = np.transpose(np.array(calibration_cut))[0].max()
        x_cut_min = np.transpose(np.array(calibration_cut))[0].min()
        y_cut_max = np.transpose(np.array(calibration_cut))[1].max()
        y_cut_min = np.transpose(np.array(calibration_cut))[1].min()

        return x_cut_min, x_cut_max, y_cut_min, y_cut_max



    def find_cut_limits_offset(calibration_cut):
        x_cut_max = (np.transpose(np.array(calibration_cut))[0].max()).astype('int')
        x_cut_min = (np.transpose(np.array(calibration_cut))[0].min()).astype('int')
        y_cut_max = (np.transpose(np.array(calibration_cut))[1].max()).astype('int')
        y_cut_min = (np.transpose(np.array(calibration_cut))[1].min()).astype('int')
        offset = x_cut_min,y_cut_min
        return x_cut_min, x_cut_max, y_cut_min, y_cut_max,offset
   
    def offset_calibrated_cut(calibration_cut):
        x_cut_min = np.transpose(np.array(calibration_cut))[0].min()
        y_cut_min = np.transpose(np.array(calibration_cut))[1].min()
        return x_cut_min,y_cut_min

    ''''def calc_scale(frome_s,to_s):
        return tuple((to_s[0]/frome_s[0]),(to_s[1]/frome_s[1]))
'''

    def show_window(title_window, window):
        cv2.namedWindow(title_window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(title_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(title_window,window)

        
    def calibration(self,final_folder_path, foldername,camera):
        corner  = 0
        calibration_cut_left = []
        calibration_cut_right = []
        started = False
        while (corner<len(self.corners)): # calibration of 4 corners
            
            # draw circle
            cv2.circle(self.calibration_page, self.corners[corner], 40, (0, 255, 0), -1)

            



            ret, frame = camera.read()   # Capture frame
            frame = cv2.flip(frame, 1)  # flip camera sees things mirorred

            #gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray-scale to work with

            

            pupils = self.pupilTracker.detect_in_frame(self.pupilTracker,frame)
            print(pupils)
            '''
            # detect faces in frame
            faces = detector(gray_scale_frame)
            if len(faces)> 1:
                print('Please avoid multiple faces.')
                sys.exit()

            for face in faces:
                landmarks = predictor(gray_scale_frame, face) # find points in face

                # get position of right eye and display lines
                #right_eye_coordinates = get_eye_coordinates(landmarks, [42, 43, 44, 45, 46, 47])
                right_eye_coordinates = get_eye_coordinates(landmarks, self.pupilTracker.right)
                display_eye_lines(frame, right_eye_coordinates, 'green')

            # define the coordinates of the pupil from the centroid of the right eye
            pupil_coordinates = np.mean([right_eye_coordinates[2], right_eye_coordinates[3]], axis = 0).astype('int')
            '''

            if cv2.waitKey(33) == ord('a'):
                calibration_cut_left.append(pupils[0])
                calibration_cut_right.append(pupils[1])
                

                # visualize message
                cv2.putText(self.calibration_page, 'ok',tuple(np.array(self.corners[corner])-5), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 0), 5)
                
                corner += 1
                started = False

            # Display results
            # print(calibration_cut, '    len: ', len(calibration_cut))
            self.show_window('projection', self.calibration_page)
            # show_window('frame', cv2.resize(frame,  (640, 360)))

            if cv2.waitKey(113) == ord('q'):
                break
        print(calibration_cut_left)
        # Process calibration
        #x_cut_min_l, x_cut_max_l, y_cut_min_l, y_cut_max_l = self.find_cut_limits(calibration_cut)
        #offset_calibrated_cut = [ x_cut_min, y_cut_min ]
        x_cut_min_l, x_cut_max_l, y_cut_min_l, y_cut_max_l,self.offset_calibrated_cut_left  = self.find_cut_limits_offset(calibration_cut_left)
        x_cut_min_r, x_cut_max_r, y_cut_min_r, y_cut_max_r,self.offset_calibrated_cut_right  = self.find_cut_limits_offset(calibration_cut_right)
        
        self.cut_frame_l = np.copy(self.calibration_page[y_cut_min_l:y_cut_max_l, x_cut_min_l:x_cut_max_l, :])
        self.cut_frame_r = np.copy(self.calibration_page[y_cut_min_r:y_cut_max_r, x_cut_min_r:x_cut_max_r, :])
        #self.offset_calibrated_cut_left = self.offset_calibrated_cut(calibration_cut_left)
        #self.offset_calibrated_cut_right = self.offset_calibrated_cut(calibration_cut_right)

        
        #self.scale_l = (np.array(self.calibration_page.shape) / np.array(self.cut_frame_l.shape))
        #self.scale_r = (np.array(self.calibration_page.shape) / np.array(self.cut_frame_r.shape))

        self.scale_l = div_tuples(self.calibration_page.shape,self.cut_frame_l.shape)
        self.scale_r = div_tuples(self.calibration_page.shape,self.cut_frame_r.shape)
        #self.scale_l = self.calc_scale(tuple(x_cut_max_l-x_cut_min_l,y_cut_max_l-y_cut_min_l),tuple(x_cut_max_r-x_cut_min_r,y_cut_max_r-y_cut_min_r))

        #self.save_calibration(foldername, offset_calibrated_cut)

        #camera.release()
        print('Calibration Finished')
        cv2.destroyAllWindows()
        #start_message(final_folder_path)
    def track_in_frame(self,frame):
        frame = cv2.flip(frame, 1)  # flip camera sees things mirorred
        pupil_l,pupil_r = self.pupilTracker.detect_in_frame(self.pupilTracker,frame)
        #return pupil_l + self.offset_calibrated_cut_left,pupil_r + self.offset_calibrated_cut_right
        #return pupil_l - self.offset_calibrated_cut_left,pupil_r - self.offset_calibrated_cut_right
        left = multi_tuples(subtract_tuples(pupil_l,self.offset_calibrated_cut_left),self.scale_l)
        right = multi_tuples(subtract_tuples(pupil_r,self.offset_calibrated_cut_right),self.scale_r)
        return left ,right
        

        
        

tracker = pupil_tracker
gaze = gaze_tracker
print(gaze.corners)
#gaze.calibration(gaze,"test","test",cap)
user32 = ctypes.windll.user32
size_screen = user32.GetSystemMetrics(1), user32.GetSystemMetrics(0)
blank_page = (np.zeros((int(size_screen[0]), int(size_screen[1]), 3)) + 255).astype('uint8')
while True:
     _, frame = cap.read()
    
     pupils = tracker.detect_in_frame(tracker,frame)
     #pupils = gaze.track_in_frame(gaze,frame)
     #print(pupils)
     main_window = (np.zeros((int(size_screen[0]), int(size_screen[1]), 3)) + 255).astype('uint8')
     main_window = frame
     cv2.circle(main_window,(int(pupils[0][0]),int(pupils[0][1])),10,(0,255,0),3)
     cv2.circle(main_window,(int(pupils[1][0]),int(pupils[1][1])),10,(0,255,0),3)
     #cv2.imshow("Frame",frame )
     cv2.namedWindow("title_window", cv2.WND_PROP_FULLSCREEN)
     cv2.setWindowProperty("title_window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
     cv2.imshow("title_window",main_window)
     key = cv2.waitKey(1)
     if key == 27:
         break
   



cap.release()
cv2.destroyAllWindows()
