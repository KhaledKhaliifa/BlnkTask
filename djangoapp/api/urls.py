from django.urls import path
from . import views

urlpatterns = [
    path('ocrapi/', views.perform_ocr, name='ocrapi'),  # Define the endpoint here
]
