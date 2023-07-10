const video = document.getElementById('inputVideo')

Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(model_url),
    faceapi.nets.faceLandmark68Net.loadFromUri(model_url),
]).then(startVideo)

let shouldRecord = true;
let mediaRecorder = null;
let chunks = [];
let detections;

function startVideo() {
    navigator.getUserMedia(
        { video: {} },
        stream => {
            // Fix video inversion
            video.style.transform = 'scaleX(-1)';
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
        // console.log(detections)
        //Code to draw the detection box on the screen
        // canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
        // const resizedDetections = faceapi.resizeResults(detections, displaySize)
        // faceapi.draw.drawDetections(canvas, resizedDetections)
        // faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)

        if (shouldRecord && detections.length == 1){
            try{
                await recordVideo();
            }catch(error){
                console.log(error);
            }
        }

    }, 100)
})

async function recordVideo(){
    if (mediaRecorder === null) {
        chunks = [];
        mediaRecorder = new MediaRecorder(video.srcObject);
        mediaRecorder.addEventListener('dataavailable', event => chunks.push(event.data));
        mediaRecorder.addEventListener('stop', () => {
            const blob = new Blob(chunks, { type: 'video/webm' });
            const videoUrl = URL.createObjectURL(blob);
            console.log('Recorded video URL:', videoUrl);
            const a = document.createElement('a');
            a.href = videoUrl;
            a.download = 'recorded-video.webm';
            document.body.appendChild(a);
            if (detections.length != 1) throw new Error("Detection is not valid");
            a.click();
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(videoUrl);
            }, 1000); // Remove the link after 1 second
            shouldRecord = false;
        });
        mediaRecorder.start();
        setTimeout(() => {
            mediaRecorder.stop();
            mediaRecorder = null;
        }, 5000); // Stop recording after 5 seconds
    }
}
