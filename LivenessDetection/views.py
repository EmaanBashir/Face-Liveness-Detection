from django.shortcuts import render
from django.shortcuts import redirect
from django.http.response import StreamingHttpResponse
from .eyeballTracking import FaceDetect
from .bloodFlow import BloodFlowDetect
# from.downloadframes(extra) import BloodFlowDetect
from .textureAnalysis import TextureDetect
from django.http import JsonResponse
from django.core.mail import send_mail
from django.http import HttpResponse
import os
import cv2
from django.core.cache import cache
from django.http import JsonResponse

# Create your views here.
def home(request):
    return render(request, "LivenessDetection/home.html", {'navbar' : 'home'})

def services(request):
    return render(request, "LivenessDetection/services.html", {'navbar' : 'services'})

def webcam(request):
    cache.set('result_available', False)
    cache.set('texture_result', None)
    cache.set('bloodflow_result', None)

    if request.method == 'POST':
        bloodflow_correct = request.POST.get('blood-flow', '')
        texture_correct = request.POST.get('texture', '')
        bloodflow_predicted = request.POST.get('bloodflow-actual', '') ### predicted
        texture_predicted = request.POST.get('texture-actual', '') ### predicted

        actual = int(bloodflow_predicted) if bloodflow_correct == 'yes' else (1 - int(bloodflow_predicted))
        
        fold_name = request.POST.get('fold-name', '')
        new_name = fold_name + '_' + str(actual)

        try:
            os.rename(fold_name, new_name)
        except FileNotFoundError:
            print(f"Folder not found.")
        except FileExistsError:
            print(f"A folder with the name already exists.")

        return render(request, "LivenessDetection/demo.html", {'navbar' : 'demo', 'type' : 'texture_feed'})

    return render(request, "LivenessDetection/demo.html", {'navbar' : 'demo', 'type' : 'texture_feed'})

def eyeball(request):
    return render(request, "LivenessDetection/demo.html", {'navbar' : 'demo', 'type' : 'video_feed'})

def bloodflow(request):
    cache.set('result_available', False)
    cache.set('texture_result', None)
    cache.set('bloodflow_result', None)
    return render(request, "LivenessDetection/demo.html", {'navbar' : 'demo', 'type' : 'bloodflow_feed'})

def usecase(request):
    return render(request, "LivenessDetection/usecase.html", {'navbar' : 'usecase'})

def aboutus(request):
    return render(request, "LivenessDetection/aboutus.html", {'navbar' : 'aboutus'})

def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name', '')
        email = request.POST.get('email', '')
        subject = request.POST.get('subject', '')
        message = request.POST.get('message', '')

        if name and email and subject and message:
            # Send email
            send_mail(
                subject,
                message,
                email,
                ['your-email@example.com'],
                fail_silently=True,
            )
            return render(request, "LivenessDetection/contact.html", {'alert' : 'Thanks for contacting us!'})
        else:
            return render(request, "LivenessDetection/contact.html", {'alert' : 'Please fill all the required fields!'})
    else:
        return render(request, "LivenessDetection/contact.html")

counter = 0
def gen(camera):
    while True:
        global counter
        frame, loopEnd = camera.get_frame()
        if (loopEnd):
            counter += 1
            if (counter == 5):
                break
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    
def video_feed(request):
    return StreamingHttpResponse(gen(FaceDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame')

def bf_gen(camera):
    while True:
        frame = camera.get_frame()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    
def bloodflow_feed(request):
    return StreamingHttpResponse(bf_gen(BloodFlowDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame')

def texture_gen(camera):
    cache.set('result_available', False)
    cache.set('texture_result', None)
    cache.set('bloodflow_result', None)
    counter = 0
    while True:
        frame = camera.get_frame()
        if camera.endloop:
            counter += 1
            if counter == 2:
                i = 0

                if not os.path.exists(camera.name):
                    os.mkdir(camera.name)

                for x in camera.all_frames:
                    # 0 is real
                    # 1 is fake
                    #filename of the form frame0_1 means texture algo predicted 1 for frame0
                    cv2.imwrite(f"{camera.name}/frame{i}_{x[1]}.png",x[0]) 
                    i += 1

                cache.set('result_available', True)
                cache.set('texture_result', camera.texture_result)
                cache.set('bloodflow_result', camera.bf_result)
                cache.set('fold_name', camera.name)
                break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
def texture_feed(request):
    folders = os.listdir('static/data')
    fold = 'static/data/video' + str(len(folders)) 

    return StreamingHttpResponse(texture_gen(TextureDetect(fold)),
					content_type='multipart/x-mixed-replace; boundary=frame')

def fetch_result(request):
    data = {'result_available': cache.get('result_available'), 'texture_result' : cache.get('texture_result'), 'bloodflow_result' : cache.get('bloodflow_result'), 'fold_name' : cache.get('fold_name')}
    return JsonResponse(data)



