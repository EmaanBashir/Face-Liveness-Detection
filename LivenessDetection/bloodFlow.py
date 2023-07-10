from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2,os,urllib.request,pickle
import numpy as np
from django.conf import settings
import os,time
from tqdm import tqdm
from scipy import signal
import mediapipe as mp
import math
import random
from . import bf_utils
import statistics
from collections import deque

class BloodFlowDetect(object):
    def __init__(self):

        self.use_prerecorded = False
        self.fs	= 30  # Sampling Frequency

        self.cap = cv2.VideoCapture(0)
        haar_cascade_path 	= "static/xml/haarcascade_frontalface_default.xml"
        self.face_cascade 		= cv2.CascadeClassifier(haar_cascade_path)
        self.tracker 			= cv2.TrackerKCF_create()
        self.cap 				= bf_utils.RecordingReader() if self.use_prerecorded else cv2.VideoCapture(0) 

        self.window				= 300 # Number of samples to use for every measurement
        self.skin_vec            = [0.3841,0.5121,0.7682]
        self.B,self.G,self.R               = 0,1,2

        self.found_face 	            = False
        self.initialized_tracker		= False
        self.face_box            	= []
        self.mean_colors             = []
        self.timestamps 	            = []

        self.mean_colors_resampled   = np.zeros((3,1))
        
        self.background = cv2.imread('static/images/background_new.png')
        self.background = cv2.resize(self.background, (1920, 1080))

        self.bpm_list = deque(maxlen=30)
        self.loops = 0

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        ret, frame = self.cap.read() 
        frame = cv2.flip(frame, 1)
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        if self.loops >= 25:
            # Try to update face location using tracker		
            if self.found_face and self.initialized_tracker :
                print("Tracking")
                self.found_face,self.face_box = self.tracker.update(frame)
                if not self.found_face:
                    print("Lost Face")

            # Try to detect new face		
            if not self.found_face:
                self.initialized_tracker = False
                print("Redetecing")
                self.faces = self.face_cascade.detectMultiScale(frame_gray, 1.3, 5)
                self.found_face = len(self.faces) > 0

            # Reset tracker
            if self.found_face and not self.initialized_tracker:			
                self.face_box = self.faces[0]
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(frame,tuple(self.face_box))			
                self.initialized_tracker = True

        # Measure face color
        if self.found_face and self.loops >= 25:
            self.face = bf_utils.crop_to_boundingbox(self.face_box,frame)
            if self.face.shape[0] > 0 and self.face.shape[1]>0:
                
                self.mean_colors += [self.face.mean(axis=0).mean(axis=0)] 
                self.timestamps  +=  [ret] if self.use_prerecorded else [time.time()]
                bf_utils.draw_face_roi(self.face_box,frame)
                t = np.arange(self.timestamps[0],self.timestamps[-1],1/self.fs)
                self.mean_colors_resampled = np.zeros((3,t.shape[0]))
                
                for color in [self.B,self.G,self.R]:
                    resampled = np.interp(t,self.timestamps,np.array(self.mean_colors)[:,color])
                    self.mean_colors_resampled[color] = resampled

        frame = cv2.resize(frame, (1920, 1080))
        mask = np.zeros_like(frame)
        height, width, _ = frame.shape
        cv2.ellipse(
            mask,
            (int(width/2), int(height/2)),
            (int(height/4), int(height/3)),
            0, 0, 360 , (255, 255,255), -1
            )
        
        mask = cv2.bitwise_not(mask)
        masked_background = cv2.bitwise_and(self.background, mask)
        masked_frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
        output = cv2.bitwise_or(masked_background, masked_frame)

        if self.loops < 25:
            print(self.loops)
            self.loops += 1
            ret1, jpeg = cv2.imencode('.jpg', output)
            return jpeg.tobytes()

        # Perform chrominance method
        if self.mean_colors_resampled.shape[1] > self.window:

            col_c = np.zeros((3,self.window))
            
            for col in [self.B,self.G,self.R]:
                col_stride 	= self.mean_colors_resampled[col,-self.window:]# select last samples
                y_ACDC 		= signal.detrend(col_stride/np.mean(col_stride))
                col_c[col] 	= y_ACDC * self.skin_vec[col]
                
            X_chrom     = col_c[self.R]-col_c[self.G]
            Y_chrom     = col_c[self.R] + col_c[self.G] - 2* col_c[self.B]
            Xf          = bf_utils.bandpass_filter(X_chrom) 
            Yf          = bf_utils.bandpass_filter(Y_chrom)
            Nx          = np.std(Xf)
            Ny          = np.std(Yf)
            alpha_CHROM = Nx/Ny
            
            x_stride   				= Xf - alpha_CHROM*Yf
            amplitude 				= np.abs( np.fft.fft(x_stride,self.window)[:int(self.window/2+1)])
            normalized_amplitude 	= amplitude/amplitude.max() #  Normalized Amplitude
            
            frequencies = np.linspace(0,self.fs/2,int(self.window/2) + 1) * 60
            bpm_index = np.argmax(normalized_amplitude)
            bpm       = frequencies[bpm_index]
            snr       = bf_utils.calculateSNR(normalized_amplitude,bpm_index)
            self.bpm_list.append(bpm)
            avg_bpm = statistics.mean(self.bpm_list)
            bf_utils.put_snr_bpm_onframe(bpm,snr,output, avg_bpm)
            
        ret1, jpeg = cv2.imencode('.jpg', output)
        return jpeg.tobytes()
