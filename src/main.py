import cv2
import numpy as np

class main:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()




