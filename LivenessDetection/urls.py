from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name = 'home'),
    path('services/', views.services, name = 'services'),
    path('demo/', views.webcam,  name = 'webcam'),
     path('bloodflow/', views.bloodflow,  name = 'real-time-bloodflow'),
    path('eyeball/', views.eyeball,  name = 'eyeball'),
    path('usecase/', views.usecase, name = 'usecase'),
    path('aboutus/', views.aboutus, name = 'aboutus'),
    path('contact/', views.contact, name = 'contact'),
    path('video_feed/', views.video_feed, name = 'video_feed'),
    path('bloodflow_feed/', views.bloodflow_feed, name = 'bloodflow_feed'),
    path('texture_feed/', views.texture_feed, name = 'texture_feed'),
    path('fetch_result/', views.fetch_result, name='fetch-result'),
]