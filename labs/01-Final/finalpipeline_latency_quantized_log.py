# audio libraries imports
import wave
import torchlibrosa
import librosa
import pyaudio
import noisereduce as nr


# system libraries imports
import os
import time
import joblib
import argparse
import threading
from pathlib import Path


# neuralnet libraries imports
import cv2
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


from utildicts import *

class Buffer:
    def __init__(self):
        self.capacity = 10
        self.storage = [0]*(self.capacity)
        self.index = 0
        self.full = False        
        
    def log(self, value):
        
        if self.full:
            return
        
        if self.index == self.capacity:
            self.index = 0
            self.full = True
            
        self.storage[self.index] = value
        self.index += 1
    
    def getlast(self):
        for i in range(self.capacity-1, -1, -1):
            if self.storage[i] != 0:
                return self.storage[i]
        return "null"
    
    
    def __str__(self):
        return str(self.storage)
    
# global buffers
b1 = Buffer()
b2 = Buffer()
b3 = ""

#locks
mut1 = threading.Lock()
mut2 = threading.Lock()
mut3 = threading.Lock()

class AudStreamer:
    # Variables
    def __init__(self):

        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = self.RATE # OR  self.CHUNK = 60
        self.MICROPHONES_LIST = []
        self.MICROPHONE_INDEX = 15
        self.MICROPHONES_DESCRIPTION = []
        self.RECORD_SECONDS = 5
        self.WAVE_OUTPUT_FILENAME = "./pytest-1.wav"
        
        self.mic_desc = ""
        self.model = 0
        # self.FPS = 60.0
        
        self.data_label = {"clapping":0,
                           "coughing":1,
                           "drinking_sipping":2,
                           "hand_saw":3,
                           "keyboard_typing":4,
                           "laughing":5,
                           "mouse_click":6,
                           "null":7,
                           "pouring_water":8,
                           "snoring":9,
                           "vacuum_cleaner":10}

    
    def list_microphones(self):
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        self.MICROPHONES_LIST = []
        self.MICROPHONES_DESCRIPTION = []
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                desc = "# %d - %s" % (i, p.get_device_info_by_host_api_device_index(0, i).get('name'))
                self.MICROPHONES_DESCRIPTION.append(desc)
                self.MICROPHONES_LIST.append(i)

        output = []
        output.append("=== Available Microphones: ===")
        output.append("\n".join(self.MICROPHONES_DESCRIPTION))
        output.append("======================================")
        return "\n".join(output), self.MICROPHONES_DESCRIPTION, self.MICROPHONES_LIST
    
    
    def get_microphone(self):
        print("=====")
        print("1 / 2: Checking Microphones... ")
        print("=====")
        desc, mics, indices = self.list_microphones()
        print(desc)
        if (len(mics) == 0):
            print("Error: No microphone found.")
            exit()

        # Read Command Line Args
        self.MICROPHONE_INDEX = indices[0]
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--mic", help="Select which microphone / input device to use")
        args = parser.parse_args()
        try:
            if args.mic:
                self.MICROPHONE_INDEX = int(args.mic)
                print("User selected mic: %d" % self.MICROPHONE_INDEX)
            else:
                mic_in = input("Select microphone [%d]: " % self.MICROPHONE_INDEX)
                if (mic_in != ''):
                    self.MICROPHONE_INDEX = int(mic_in)
        except Exception as e:
            print(e)
            print("Invalid microphone")
            exit()

        # Find description that matches the mic index
        self.mic_desc = ""
        for k in range(len(indices)):
            i = indices[k]
            if (i == self.MICROPHONE_INDEX):
                self.mic_desc = mics[k]
        print("Using mic: %s" % self.mic_desc)
        
        
    def get_model(self):
        MODEL_PATH = "audionet.tflite"
        print("=====")
        print("2 / 2: Checking model... ")
        print("=====")

        print("Using deep learning model: %s" % (MODEL_PATH))
        self.model = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        print("--Quantized model loaded--")
        # model.summary()
        
        
    def convert_to_stft(self, audio_data, sample_rate):
#         print("Converting the Data to STFT")
#         print(audio_data.shape)
        print("--", end="")

        # noise reduction
        noisy_part = audio_data[0:25000]
        reduced_noise = nr.reduce_noise(y=audio_data, y_noise=noisy_part, sr=sample_rate)

        # trimming
        #trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)

        # extract features
        stft = np.abs(librosa.stft(reduced_noise, n_fft=512, hop_length=256, win_length=512))
        #print("Converted data to STFT")
        print(">>", end="")
        return stft
    
    def get_key(self,val):
        for key, value in self.data_label.items():
            if val == value:
                return key

        return "key doesn't exist"
    
    def audio_samples(self, in_data, frame_count, time_info, status_flags):
#         print("Obtaining Audio samples, time_infor: ", time_info)
#         print("Obtaining Audio samples, frame_count: ", frame_count)
#         print("Obtaining Audio samples, status_flags: ", status_flags)

        np_wav = np.fromstring(in_data, dtype=np.int16) / 32768.0  # Convert to [-1.0, +1.0]
        # Convert to mono.
        if len(np_wav.shape) > 1:
            np_wav = np.mean(np_wav, axis=1)

        stft = self.convert_to_stft(np_wav, self.RATE)
        stft = np.array(stft, dtype=np.float32).T
        stft = np.mean(stft, axis=0)
        stft = np.array([stft])

        start = time.time()
        self.model.resize_tensor_input(self.input_details[0]['index'], (1, 257))
        self.model.allocate_tensors()
#         print(stft.shape)
        self.model.set_tensor(self.input_details[0]['index'], stft)
        self.model.invoke()
        res = np.argmax(self.model.get_tensor(self.output_details[0]['index']), axis=1)
        end = time.time()
        
        tot_time = end-start
        mut2.acquire()
        log_data = self.get_key(res[0]) + ","+ str(tot_time)
        b2.log(log_data)
        
        mut2.release()
        print("Activity: ",self.get_key(res[0]))

        #print("in_data")

        return (in_data, pyaudio.paContinue)
    
    def stream_audinference(self):
        #self.get_microphone()
        self.get_model()
        while True:
            p = pyaudio.PyAudio()
            stream = p.open(format=self.FORMAT, 
                            channels=self.CHANNELS, 
                            rate=self.RATE, 
                            input=True, 
                            frames_per_buffer=self.CHUNK,
                            stream_callback=self.audio_samples, 
                            input_device_index=self.MICROPHONE_INDEX)
            
            stream.start_stream()    
            while stream.is_active():
                time.sleep(0.001)
        
        stream.stop_stream()
        stream.close()
        p.terminate()

class CamStreamer:
    def __init__(self, res, disp_x=None, disp_y=None, framerate=20, flipmethod=0):
        
        self.capture_width = res[0]
        self.capture_height= res[1]
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
        
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
                
                cropped = cv2.resize(gray[y:y+h,x:x+w], (48,48), interpolation = cv2.INTER_AREA)
                cropped = cropped.reshape(1,48,48,1)
                
                model.resize_tensor_input(input_details[0]['index'], (1, 48, 48, 1))
                model.allocate_tensors()
                model.set_tensor(input_details[0]['index'], cropped.astype(np.float32))
                model.invoke()
                
                # prediction for emotion
                st = time.time()
                preds = model.get_tensor(output_details[0]['index'])
                end = time.time()
                tot_time = end-st
                
                infer = self.classes[preds[0].argmax()]
                logstr = infer + "," + str(tot_time)
                #
                mut1.acquire()
                b1.log(logstr)
                mut1.release()
                
                
                for (x, y, w, h) in rects:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                mut2.acquire()
                
                act = b2.getlast() 
                if act == "null":
                    act = "" 
                
                mut3.acquire()
                situ_comm = f"Situation: {b3}"
                img = cv2.putText(img, situ_comm, (0, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
                mut3.release()
                cmt ="Emotion: {0}, Activity: {1}".format(infer,act.split(',')[0])
                img = cv2.putText(img, cmt, (0, 32),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)
                mut2.release()
                
                cv2.imshow("ODML767", img)
                keyCode = cv2.waitKey(30) & 0xFF
                if keyCode == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            print("Unable to open camera")


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(18, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
        )
          
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 18),
            torch.nn.Tanh()

        )
  
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def classifier():
    x = []
    
    model = AE()
    model.load_state_dict(torch.load("ae.h5"))
    model.eval()
    
    clf = joblib.load('clf.model')
    
    while True:
        if b1.full and b2.full:
            mut1.acquire()
            mut2.acquire()
            
            # nullify temporary buffers
            temp = [0]*18
            x=[]

            emotion = b1.storage[-1].split(',')[0]
            tot_emotion = b1.storage[-1].split(',')[1]
            activity = b2.storage[-1].split(',')[0]
            tot_activity = b2.storage[-1].split(',')[1]
            
            temp[emotionLabels[emotion]] = 1
            temp[7+activityLabels[activity]] = 1
            temp = np.array(temp,dtype=np.float32)
            sample = torch.from_numpy(temp)

            x.append(model.forward(sample).tolist())
            
            start = time.time()
            pred = clf.predict(x)[0]
            end = time.time()
            
            tot = end - start
            total_time_taken = tot + float(tot_emotion) + float(tot_activity)
            
            #ToDO:  Log to file
            
            mut3.acquire()
            global b3
            for k in finalLabels:
                if pred == finalLabels[k]:
                    b3 = k
                    
            f = open("latencies.txt", "a")
            f.write(f"TS: {time.time():.02},Situation: {b3},Time-taken:{total_time_taken:.06}s\n")
            f.close()

            print("---PRED---: ", pred)
            print(f"Situation:- {b3}")
            print("Activity: {0}, Emotion {1}".format(activity,emotion))
            print("---PRED---: ", pred)
            mut3.release()
            
            pred = ""
            b1.storage = [0]*(b1.capacity)
            b2.storage = [0]*(b2.capacity)
            b1.full = False
            b2.full = False
            mut2.release()
            mut1.release()

# third model test dummy
def reader():  
    last_val_b1 = 0
    last_val_b2 = 0
    while True:   
        if b1.full and b2.full:
            mut1.acquire()
            print(f"--Buffer1: {b1}")  # reading
            last_val_b1 = b1.storage
            b1.storage = [0]*(b1.capacity)  # emptire
            b1.full = False
            print(f"---{b1}")
            mut1.release()

        if b2.full:
            mut2.acquire()
            print(f"--Buffer2: {b2}")
            last_val_b2 = b2.storage
            b2.storage = [0]*(b1.capacity)
            b2.full = False
            print(f"---{b2}")
            mut2.release()
        time.sleep(0.1)

        
        
if __name__ == "__main__":
    
    # clear camera workers
    os.system("sudo service nvargus-daemon restart")

    camera_inference = CamStreamer(resolution['4CIF'])
    audio_inference = AudStreamer()

    r = threading.Thread(target=classifier)
    s_a = threading.Thread(target=camera_inference.stream_caminference)
    s_b = threading.Thread(target=audio_inference.stream_audinference)

    
    s_a.start()
    s_b.start()
    r.start()
    s_a.join()
    s_b.join()
    r.join()