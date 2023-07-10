const video = document.getElementById('inputVideo')

Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(model_url),
    faceapi.nets.faceLandmark68Net.loadFromUri(model_url),
]).then(startVideo)

let detections;

function startVideo() {
    navigator.getUserMedia(
        { video: {} },
        stream => {
            // Fix video inversion
            // video.style.transform = 'scaleX(-1)';
            video.srcObject = stream;
        },
        err => console.error(err)
    )
}

video.addEventListener('play', () => {
    const canvas = faceapi.createCanvasFromMedia(video)
    document.getElementById("webcamDiv").append(canvas)
    setInterval(async () => {
        const displaySize = { width: video.offsetWidth, height: video.offsetHeight }
        faceapi.matchDimensions(canvas, displaySize)
        detections = await faceapi.detectAllFaces(video,
            new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks()
        console.log(detections)
        // Code to draw the detection box on the screen
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
        const resizedDetections = faceapi.resizeResults(detections, displaySize)
        faceapi.draw.drawDetections(canvas, resizedDetections)
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)

    }, 100)
})

