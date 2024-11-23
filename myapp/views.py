from django.http import JsonResponse
from rest_framework.decorators import api_view
from django.core.files.storage import default_storage
from torchvision import transforms, models
from PIL import Image
import torch
from django.conf import settings
import os
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


@api_view(['POST'])
def detect_image(request):
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image provided'})

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),                
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = request.FILES['image']
    temp_path = default_storage.save(f'temp/{image.name}', image)
    image_full_path = os.path.join(settings.MEDIA_ROOT, temp_path)

    try:
        image = Image.open(image_full_path).convert("RGB")
    except Exception as e:
        return JsonResponse({'error': f'Invalid image file: {str(e)}'})

    try:
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
    except Exception as e:
        return JsonResponse({'error': f'Error during preprocessing: {str(e)}'})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_batch = input_batch.to(device)

    try:
        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_index = torch.argmax(probabilities).item()
    except Exception as e:
        return JsonResponse({'error': f'Error during model prediction: {str(e)}'})

    try:
        imagenet_labels_path = 'server/imagenet_class_index (1).json'
        with open(imagenet_labels_path) as f:
            class_labels = json.load(f)
    except Exception as e:
        return JsonResponse({'error': f'Error loading class labels: {str(e)}'})

    predicted_label = class_labels.get(str(predicted_class_index), ["Unknown", "Unknown"])[1]

    default_storage.delete(temp_path)
    redirect_url = 'https://en.wikipedia.org/wiki/'

    return JsonResponse({'result': predicted_label, 'url': redirect_url})
