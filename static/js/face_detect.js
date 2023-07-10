import * as faceapi from 'face-api.js'

const MODEL_URL = '../../face_detection_models'

faceapi.loadSsdMobilenetv1Model(MODEL_URL)
faceapi.loadFaceLandmarkModel(MODEL_URL)
faceapi.loadFaceRecognitionModel(MODEL_URL)

const input = document.getElementById('myImage')
let fullFaceDescriptions = faceapi.detectAllFaces(input).withFaceLandmarks().withFaceDescriptors()
