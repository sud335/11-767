import os
import cv2
import time
import numpy as np
import tensorflow as tf

from resolutions import resolution
from tensorflow.keras.models import load_model

# clear camera workers
os.system("sudo service nvargus-daemon restart")


class CamStreamer:
    def __init__(self, res, disp_x=None, disp_y=None, framerate=20, flipmethod=0):
        
        self.capture_width = res[0]
        self.capture_height= res[1]
        self.classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        if disp_x is None:
            self.display_width = res[0]
        else:
            self.display_width = disp_x
        if disp_y is None:
            self.display_height= res[1]
        else:
            self.display_height= disp_y
            
        self.framerate = framerate
        self.flip_method = flipmethod

    def get_pipeline(self):
        return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink" %
                (self.capture_width, self.capture_height, self.framerate, self.flip_method,
                 self.display_width, self.display_height))
    
    def stream(self):
        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        cap = cv2.VideoCapture(self.get_pipeline(), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            window_handle = cv2.namedWindow("ODML767", cv2.WINDOW_AUTOSIZE)
            
            while cv2.getWindowProperty("ODML767", 0) >= 0:
                ret_val, img = cap.read()
                cv2.imshow("ODML767", img)
                keyCode = cv2.waitKey(30) & 0xFF
                if keyCode == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("Unable to open camera")

            
    def stream_caminference(self):
        detector = cv2.CascadeClassifier('haar_frontal_face.xml')
        cap = cv2.VideoCapture(self.get_pipeline(), cv2.CAP_GSTREAMER)
        
        model_filename = "emonet.tflite"
        model = tf.lite.Interpreter(model_path=model_filename)
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        print("--Done Loading Model--")
        
        if cap.isOpened():
            window_handle = cv2.namedWindow("ODML767", cv2.WINDOW_AUTOSIZE)    
            while cv2.getWindowProperty("ODML767", 0) >= 0:
                ret_val, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector.detectMultiScale(gray, scaleFactor=1.15,
                                minNeighbors=6, minSize=(20, 20),
                                flags=cv2.CASCADE_SCALE_IMAGE)

                if len(rects) != 0:
                    (x, y, w, h) = rects[0]
                else:
                    (x,y,w,h) = (0,0,48,48) 
                
                cropped = cv2.resize(img[y:y+h,x:x+w], (48,48), interpolation = cv2.INTER_AREA)
                cropped = cropped[:,:,0].reshape(1,48,48,1)
                
                model.resize_tensor_input(input_details[0]['index'], (1, 48, 48, 1))
                model.allocate_tensors()
                model.set_tensor(input_details[0]['index'], cropped.astype(np.float32))
                model.invoke()
                
                preds = model.get_tensor(output_details[0]['index'])
                
                print(self.classes[preds[0].argmax()])
                
                for (x, y, w, h) in rects:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                img = cv2.putText(img,self.classes[preds[0].argmax()],(0, 12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                cv2.imshow("ODML767", img)
                keyCode = cv2.waitKey(30) & 0xFF
                if keyCode == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            print("Unable to open camera")