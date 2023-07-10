import numpy as np
import joblib
from skimage import feature as skif
import cv2


def lbp_histogram(image,P=8,R=1,method = 'nri_uniform'):

    #print("lbp_histogram | feature_extract.py")

    '''
    image: shape is N*M 
    '''
    lbp = skif.local_binary_pattern(image, P,R, method) # lbp.shape is equal image.shape
    # cv2.imwrite("lbp.png",lbp)
    max_bins = int(lbp.max() + 1) # max_bins is related P
    hist,_= np.histogram(lbp,   bins=max_bins, range=(0, max_bins))
    ##normed=True,
    return hist

# model = joblib.load("Face-anti-spoofing-based-on-color-texture-analysis-master/model.m")
# image = cv2.imread("Face-anti-spoofing-based-on-color-texture-analysis-master/real-time/my_fake.jpg")

# original_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
# #cv2.imshow('original_image',original_image)

# # Convert color image to grayscale for Viola-Jones
# grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #cv2.imshow('grayscale_image',grayscale_image)

# # Load the classifier and create a cascade object for face detection
# face_cascade = cv2.CascadeClassifier('Face-anti-spoofing-based-on-color-texture-analysis-master/haarcascade_frontalface_alt.xml')

# detected_faces = face_cascade.detectMultiScale(grayscale_image)

# for (column, row, width, height) in detected_faces:
#     cv2.rectangle(
#         original_image,
#         (column, row),
#         (column + width, row + height),
#         (0, 255, 0),
#         2
#     )
#     face_image = original_image[row: row+height, column: column+width]
#     print(face_image)
#     #cv2.imshow('face_image',face_image)

#size the image to 64x64 pixels
# final = cv2.resize(face_image, (112, 112)).cvtColor(image, cv2.COLOR_BGR2YCrCb)
# final = cv2.resize(face_image, (112, 112))
# final = cv2.cvtColor(final, cv2.COLOR_BGR2YCrCb)
#cv2.imshow('resized_image',final)

#cv2.waitKey(0)

##########################################################

# def predict(file_path):

#     model = joblib.load("LivenessDetection/models/model.m")
#     image = cv2.imread(file_path)

       
#     HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#colorspace
#     V = lbp_histogram(HSV_image[:,:,2])

#     RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#colorspace
#     R = lbp_histogram(RGB_image[:,:,0])
#     G = lbp_histogram(RGB_image[:,:,1])
#     RG=R-G
#     B = lbp_histogram(RGB_image[:,:,2])

#     feature = np.concatenate((RG,B,V))

#     if model.predict([feature])==1 :
#         output="fake"
#     elif model.predict([feature])==0 :
#         output="real"
    
#     print(output)

def predict(frame):

    # model = joblib.load("LivenessDetection/models/model.m")
    model = joblib.load("LivenessDetection/models/HSV-YCrCb.m")

    # HSV_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#colorspace
    # V = lbp_histogram(HSV_image[:,:,2])

    # RGB_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#colorspace
    # R = lbp_histogram(RGB_image[:,:,0])
    # G = lbp_histogram(RGB_image[:,:,1])
    # RG=R-G
    # B = lbp_histogram(RGB_image[:,:,2])

    # feature = np.concatenate((RG,B,V))


    color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)#colorspace
    h1 = lbp_histogram(color_image[:,:,0])
    h2 = lbp_histogram(color_image[:,:,1])
    h3 = lbp_histogram(color_image[:,:,2])

    color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#colorspace
    H1 = lbp_histogram(color_image[:,:,0])
    H2 = lbp_histogram(color_image[:,:,1])
    H3 = lbp_histogram(color_image[:,:,2])

    feature = np.concatenate((h1, h2, h3, H1, H2, H3)) 

    # 1 is fake
    # 0 is real
    return model.predict([feature])

def put_result_onframe(frame, result):
    text = f'Texture Analysis : {result}'
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (00, 350) 
    fontScale = 1
    color = (0, 0, 255) 
    thickness = 2
    cv2.putText(frame,text,org,font,fontScale,color,thickness)
