from django.shortcuts import render,redirect
from torchvision import transforms, models
import torchvision as tf
from PIL import Image
import torch
import json
import os
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from rest_framework import viewsets
from .models import *
from .serializer import *
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser


class ItemViewSet(viewsets.ModelViewSet):
     queryset = Item.objects.all()
     serializer_class = Itemserializer
     parser_classes = [MultiPartParser] 

def detect(request):
    image1 = Item.objects.filter().last()
    if not image1 or not image1.image:
        return Response("No image found", status=404)

    model = tf.models.resnet50(pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        image_path = image1.image.path
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  
    except Exception as e:
        return Response(f"Error processing image: {str(e)}", status=500)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class_index = torch.argmax(probabilities).item()

    try:
        imagenet_labels_path = os.path.join(settings.BASE_DIR, r'F:\MITHUN\ANDROID ( REACT-NATIVE )\objectdetection_ai\imagenet_class_index (1).json')
        with open(imagenet_labels_path) as f:
            class_labels = json.load(f)
    except Exception as e:
        return Response(f"Error loading class labels: {str(e)}", status=500)

    predicted_label = class_labels.get(str(predicted_class_index), ["Unknown", "Unknown"])[1]

    return Response(predicted_label)