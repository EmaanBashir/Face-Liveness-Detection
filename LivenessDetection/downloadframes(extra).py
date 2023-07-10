from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2,os,urllib.request,pickle
import numpy as np
from django.conf import settings
import os,time, utils
from tqdm import tqdm
from scipy import signal

# from . import train_model
# # load our serialized face detector model from disk
# protoPath = os.path.sep.join([settings.BASE_DIR, "face_detection_model\\deploy.prototxt"])
# modelPath = os.path.sep.join([settings.BASE_DIR,"face_detection_model\\res10_300x300_ssd_iter_140000.caffemodel"])
# detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# # load our serialized face embedding model from disk
# embedder = cv2.dnn.readNetFromTorch(os.path.join(settings.BASE_DIR,'face_detection_model/openface_nn4.small2.v1.t7'))
# # load the actual face recognition model along with the label encoder
# recognizer = os.path.sep.join([settings.BASE_DIR, "output\\recognizer.pickle"])
# recognizer = pickle.loads(open(recognizer, "rb").read())
# le = os.path.sep.join([settings.BASE_DIR, "output\\le.pickle"])
# le = pickle.loads(open(le, "rb").read())
# dataset = os.path.sep.join([settings.BASE_DIR, "dataset"])
# user_list = [ f.name for f in os.scandir(dataset) if f.is_dir() ]

import mediapipe as mp
import math
import random

class BloodFlowDetect(object):
    def __init__(self):

        self.cap = cv2.VideoCapture(0)
        self.timestamps = []
        self.max_frames = 700
        self.stopRecording = False
        self.counter = 0
        self.recorded = False

        self.background = cv2.imread('static/images/background_new.png')
        self.background = cv2.resize(self.background, (1920, 1080))

        #face_mesh object
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5    
        )

        self.name = "static/recording_rest"

        if not os.path.exists(self.name):
            os.mkdir(self.name)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    # def detect(self):


    def get_frame(self):
        ret, frame = self.cap.read()

        frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (1920, 1080))
        mask = np.zeros_like(frame)
        height, width, _ = frame.shape
        cv2.ellipse(
            mask,
            (int(width/2), int(height/2)),
            (int(height/2.5), int(height/2)),
            0, 0, 360 , (255, 255,255), -1
            )
        
        mask = cv2.bitwise_not(mask)
        masked_background = cv2.bitwise_and(self.background, mask)
        masked_frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
        output = cv2.bitwise_or(masked_background, masked_frame)

        rgb_frame = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if (results.multi_face_landmarks and (not self.stopRecording)):
            self.timestamps+=[str(time.time())]
            cv2.imwrite(f"{self.name}/{self.counter}.png",frame)
            self.counter += 1

            if (self.counter >= self.max_frames):
                self.stopRecording = True
            
        if ((not self.recorded) and self.stopRecording):
            f = open(f"{self.name}/timestamps.txt","w")
            f.write(",".join(self.timestamps))
            f.close()
            

        ret1, jpeg = cv2.imencode('.jpg', output)
        return jpeg.tobytes()

# class Detection(object):
#     def __init__(self):
#         #%% User Settings
#         self.use_prerecorded		= True
#         self.fs					= 30  # Sampling Frequency

#         self.haar_cascade_path 	= "static/xml/haarcascade_frontalface_default.xml"
#         self.face_cascade 		= cv2.CascadeClassifier(self.haar_cascade_path)
#         tracker 			= cv2.TrackerKCF_create()
#         cap 				= utils.RecordingReader() if use_prerecorded else cv2.VideoCapture(0) 

#         window				= 300 # Number of samples to use for every measurement
#         skin_vec            = [0.3841,0.5121,0.7682]
#         B,G,R               = 0,1,2

#         found_face 	            = False
#         initialized_tracker		= False
#         face_box            	= []
#         mean_colors             = []
#         timestamps 	            = []

#         mean_colors_resampled   = np.zeros((3,1))
