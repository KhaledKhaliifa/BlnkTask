import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.apps import apps
from api.model.encoder_utils import  decode_characters
from api.image_preprocessing.preprocessing import preprocess_image

@api_view(['POST'])
def perform_ocr(request):
    try:
        # Load the model from the apps
        model = apps.get_app_config('api').ocr_model

        # Check if an image was uploaded
        if 'image' not in request.FILES:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the uploaded image
        uploaded_file = request.FILES['image']

        # Preprocess the image
        preprocessed_image = preprocess_image(uploaded_file)  # Ensure preprocess_image works with cv2 images

        # Predict using the model
        prediction = model.predict(np.expand_dims(preprocessed_image, axis=0)) 

        # Decode the predicted characters
        decoded_characters = decode_characters(prediction)

        return Response({'prediction': decoded_characters}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
