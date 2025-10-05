from django.urls import path
from . import views

app_name = 'segmentation'

urlpatterns = [
    path('', views.upload_image, name='upload'),
    path('result/<str:output_filename>/', views.view_result, name='result'),
]
