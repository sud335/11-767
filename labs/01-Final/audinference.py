import time
import wave
import pyaudio
import argparse
import torchlibrosa
import librosa
import numpy as np
import noisereduce as nr
from pathlib import Path

from tensorflow.keras.models import load_model

class AudStreamer:
    # Variables
    def __init__(self):

        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = self.RATE # OR  self.CHUNK = 60
        self.MICROPHONES_LIST = []
        self.MICROPHONES_INDEX = 0
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
        MODEL_PATH = "audionet.h5"
        print("=====")
        print("2 / 2: Checking model... ")
        print("=====")
        
        print("Using deep learning model: %s" % (MODEL_PATH))
        self.model = load_model(MODEL_PATH, compile=False)
        print("--model loaded--")
        # model.summary()
        
        
    def convert_to_stft(self, audio_data, sample_rate):
#         print("Converting the Data to STFT")
#         print(audio_data.shape)
        print("--", end="")

        # noise reduction
        noisy_part = audio_data[0:25000]
        reduced_noise = nr.reduce_noise(y=audio_data, y_noise=noisy_part, sr=sample_rate)

        # trimming
        trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)

        # extract features
        stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
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
        stft = np.array(stft).T
        stft = np.mean(stft, axis=0)
        stft = np.array([stft])
        res = np.argmax(self.model.predict(stft), axis=1)
        print("Activity: ",self.get_key(res[0]))

        #print("in_data")

        return (in_data, pyaudio.paContinue)
    
    def stream_audinference(self):
        self.get_microphone()
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
            
            # print("# Live Prediction Using Microphone: %s" % (self.mic_desc))
            stream.start_stream()
            
            while stream.is_active():
                time.sleep(0.1)