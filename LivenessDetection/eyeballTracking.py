from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2,os,urllib.request,pickle
import numpy as np
from django.conf import settings
import os

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
import time

#face_mesh object
mp_face_mesh = mp.solutions.face_mesh

#face_mesh object
mp_face_mesh = mp.solutions.face_mesh

#left eye and right eye indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

#left and right iris indices
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

rightEyeRL = [263] #right eye right most landmark
rightEyeLL = [362] #right eye left most landmark
lefteyeRL = [133] #left eye right most landmark
lefteyeLL = [33] #left eye left most landmark

#calculate the distance
def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

iris_positions = []
#find the position
def iris_position(iris_center, right_point, left_point):
    center_to_right_distance = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_distance/total_distance
    iris_position = ""
    if ratio <= 0.434:
        iris_position = "right"
        iris_positions.append("right")
    elif ratio > 0.434 and ratio <= 0.50:
        iris_position = "center"
        iris_positions.append("center")
    else:
        iris_position = "left"
        iris_positions.append("left")
    print(ratio)
    return iris_positions, iris_position, ratio

x_axisRange = 1920
y_axisRange = 1080
randomX=[]
randomY=[]

def randomNumber():
    again = True
    while (again):        
        rx = random.randint(0, x_axisRange)
        ry = random.randint(0, y_axisRange)
        if ((rx not in randomX) and (ry not in randomY) and ((rx < 300) or (rx > 1700) or (rx > 900 and rx < 1020))):
            randomX.append(rx)
            randomY.append(ry)
            again = False
    return rx, ry

circlePositions = []
def getCirclePosition(x,y):
    cir_position = ""
    if (700 <= x <= 1219):
        cir_position = "center"
        circlePositions.append("center")
    elif (0 <= x <= 699):
        cir_position = "left"
        circlePositions.append("left")
    else:
        cir_position = "right"
        circlePositions.append("right")
    return cir_position
        
class FaceDetect(object):
    def __init__(self):

        self.cap = cv2.VideoCapture(0)
        self.rx1, self.ry1 = randomNumber()
        self.rx2, self.ry2 = randomNumber()
        self.rx3, self.ry3 = randomNumber()
        self.rx4, self.ry4 = randomNumber()
        self.rx5, self.ry5 = randomNumber()
        self.circleMatchCounter = 0
        self.loopEnd = False
        self.circleMatches = 0
        self.wrongAnswers = 0
        self.maxCount = 20
        self.maxWrong = 50
        self.outputText = ""

        # self.background = cv2.imread('C:\\Users\\Dell\\OneDrive\\Desktop\\FYP\\WebApp\\LivenessDetectionApp\\LivenessDetection\\background.png')
        self.background = cv2.imread('static/images/background_new.png')

        self.background = cv2.resize(self.background, (1920, 1080))
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5    
        )

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.loopEnd = True  ### break
        #flip it so that its more like a mirror
        #1 -> mirror image
        frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (1920, 1080))
        mask = np.zeros_like(frame)
        height, width, _ = frame.shape
        cv2.ellipse(
            mask,
            (int(width/2), int(height/2)),
            (int(width/5), int(height/5)),
            0, 0, 360 , (255, 255,255), -1
            )
        
        mask = cv2.bitwise_not(mask)
        masked_background = cv2.bitwise_and(self.background, mask)
        masked_frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
        output = cv2.bitwise_or(masked_background, masked_frame)
        #it creates the image in bgr but mediapipe requires in RGB.
        rgb_frame = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        img_h, img_w = output.shape[:2]
        results = self.face_mesh.process(rgb_frame)
        cv2.putText(output, self.outputText, (760, 570), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
        if (results.multi_face_landmarks):
            #print(results.multi_face_landmarks[0].landmark)
            #get the pixel coordinate by multiplying the landmarks x and y with the frame width and height
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            
            #478 landmark points in it (due to the refine_landmarks)
            #print(mesh_points.shape)

            #draw the eye shape around the detected one
            #cv2.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 0,255), 1, cv2.LINE_AA)
            #cv2.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0, 0,255), 1, cv2.LINE_AA)

            #draw circles for iris - detected eyes
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv2.circle(output, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(output, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(output, mesh_points[rightEyeRL[0]], 3, (255, 255 , 255), -1, cv2.LINE_AA)
            cv2.circle(output, mesh_points[rightEyeLL[0]], 3, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(output, mesh_points[lefteyeRL[0]], 3, (255, 255 , 255), -1, cv2.LINE_AA)
            cv2.circle(output, mesh_points[lefteyeLL[0]], 3, (0, 255, 255), -1, cv2.LINE_AA)

            iris_array, iris_pos, ratio = iris_position(center_right, mesh_points[rightEyeRL[0]], mesh_points[rightEyeLL[0]])
            #iris_pos1, ratio1 = iris_position(center_left, mesh_points[lefteyeRL[0]], mesh_points[lefteyeLL[0]])
            print(iris_pos)

            if(self.circleMatchCounter == 0):
                cv2.circle(output, (self.rx1, self.ry1), radius=0, color=(255,0,0), thickness=20)
                circlePosition = getCirclePosition(self.rx1,self.ry1)
                if(iris_pos == circlePosition):
                    self.circleMatches += 1
                    print("circle 1 (" , self.rx1, self.ry1 , ") matched", self.circleMatches)
                else:
                    self.wrongAnswers += 1

            elif(self.circleMatchCounter == 1):
                cv2.circle(output, (self.rx2, self.ry2), radius=0, color=(255,0,0), thickness=20)
                circlePosition = getCirclePosition(self.rx2,self.ry2)
                if(iris_pos == circlePosition):
                    self.circleMatches += 1
                    print("circle 2 (" , self.rx2, self.ry2 , ") matched", self.circleMatches)
                else:
                    self.wrongAnswers += 1
                    
            elif(self.circleMatchCounter == 2):
                cv2.circle(output, (self.rx3, self.ry3), radius=0, color=(255,0,0), thickness=20)
                circlePosition = getCirclePosition(self.rx3,self.ry3)
                if(iris_pos == circlePosition):
                    self.circleMatches += 1
                    print("circle 3 (" , self.rx3, self.ry3 , ") matched", self.circleMatches)
                else:
                    self.wrongAnswers += 1

            elif(self.circleMatchCounter == 3):
                cv2.circle(output, (self.rx4, self.ry4), radius=0, color=(255,0,0), thickness=20)
                circlePosition = getCirclePosition(self.rx4,self.ry4)
                if(iris_pos == circlePosition):
                    self.circleMatches += 1
                    print("circle 4 (" , self.rx4, self.ry4 , ") matched", self.circleMatches)
                else:
                    self.wrongAnswers += 1

            elif(self.circleMatchCounter == 4):
                cv2.circle(output, (self.rx5, self.ry5), radius=0, color=(255,0,0), thickness=20)
                circlePosition = getCirclePosition(self.rx5,self.ry5)
                if(iris_pos == circlePosition):
                    self.circleMatches += 1
                    print("circle 5 (" , self.rx5, self.ry5 , ") matched", self.circleMatches)
                else:
                    self.wrongAnswers += 1

            elif(self.circleMatchCounter == 5):
                self.outputText = "Real Person"
                print("Real person!")
                self.loopEnd = True
                ### break

            if (self.circleMatches > self.maxCount):
                self.circleMatches = 0
                self.wrongAnswers = 0
                self.circleMatchCounter = self.circleMatchCounter + 1
            elif (self.wrongAnswers > self.maxWrong):
                self.loopEnd = True
                self.outputText = "Fake Person"
                print('Fakeeeeee!!!!!!!')

        ret1, jpeg = cv2.imencode('.jpg', output)
        return jpeg.tobytes(), self.loopEnd
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.loopEnd = True ### break
            
        


# class FaceDetect(object):
# 	def __init__(self):
# 		# initialize the video stream, then allow the camera sensor to warm up
# 		self.vs = VideoStream(src=0).start()
# 		# start the FPS throughput estimator
# 		self.fps = FPS().start()

# 	def __del__(self):
# 		cv2.destroyAllWindows()

# 	def get_frame(self):
# 		# grab the frame from the threaded video stream
# 		frame = self.vs.read()
# 		frame = cv2.flip(frame,1)

# 		# resize the frame to have a width of 600 pixels (while
# 		# maintaining the aspect ratio), and then grab the image
# 		# dimensions
# 		frame = imutils.resize(frame, width=600)
# 		(h, w) = frame.shape[:2]

# 		# construct a blob from the image
# 		imageBlob = cv2.dnn.blobFromImage(
# 			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
# 			(104.0, 177.0, 123.0), swapRB=False, crop=False)

# 		# # apply OpenCV's deep learning-based face detector to localize
# 		# # faces in the input image
# 		# detector.setInput(imageBlob)
# 		# detections = detector.forward()


# 		# # loop over the detections
# 		# for i in range(0, detections.shape[2]):
# 		# 	# extract the confidence (i.e., probability) associated with
# 		# 	# the prediction
# 		# 	confidence = detections[0, 0, i, 2]

# 		# 	# filter out weak detections
# 		# 	if confidence > 0.5:
# 		# 		# compute the (x, y)-coordinates of the bounding box for
# 		# 		# the face
# 		# 		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 		# 		(startX, startY, endX, endY) = box.astype("int")

# 		# 		# extract the face ROI
# 		# 		face = frame[startY:endY, startX:endX]
# 		# 		(fH, fW) = face.shape[:2]

# 		# 		# ensure the face width and height are sufficiently large
# 		# 		if fW < 20 or fH < 20:
# 		# 			continue

# 		# 		# construct a blob for the face ROI, then pass the blob
# 		# 		# through our face embedding model to obtain the 128-d
# 		# 		# quantification of the face
# 		# 		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
# 		# 			(96, 96), (0, 0, 0), swapRB=True, crop=False)
# 		# 		embedder.setInput(faceBlob)
# 		# 		vec = embedder.forward()

# 		# 		# perform classification to recognize the face
# 		# 		preds = recognizer.predict_proba(vec)[0]
# 		# 		j = np.argmax(preds)
# 		# 		proba = preds[j]
# 		# 		name = le.classes_[j]


# 		# 		# draw the bounding box of the face along with the
# 		# 		# associated probability
# 		# 		text = "{}: {:.2f}%".format(name, proba * 100)
# 		# 		y = startY - 10 if startY - 10 > 10 else startY + 10
# 		# 		cv2.rectangle(frame, (startX, startY), (endX, endY),
# 		# 			(0, 0, 255), 2)
# 		# 		cv2.putText(frame, text, (startX, y),
# 		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# 		# update the FPS counter
# 		self.fps.update()
# 		ret, jpeg = cv2.imencode('.jpg', frame)
# 		return jpeg.tobytes()
		