import numpy as np
import cv2
from rest_framework.response import Response
from rest_framework import status
import numpy as np
import cv2

def normalize_image(image, target_size=(140, 40)):
    """Normalizes the given image to values between 0 and 1

    Args:
        image (numpy.ndarray): The image to normalize

    Returns:    
        numpy.ndarray: The normalized image
    """
    image_resized = np.expand_dims(cv2.resize(image, target_size), axis=-1)
    image_normalized = image_resized / 255.0
    return image_normalized

def threshold_image(image):
    """Applies a 245 threshold on the image

    Args:
        image (numpy.ndarray): The grayscale image to threshold

    Returns:
        numpy.ndarray: The thresholded image
    """
    _, thresholded_image = cv2.threshold(image, 245, 255, cv2.THRESH_BINARY)
    image = thresholded_image 
    return image

def load_image(image):
    """Loads the given image from the files upload

    Args:
        image (File): The image to load

    Returns:
        numpy.ndarray: The loaded image
    """
    file_bytes = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  
    if image is None:
        return Response({'error': 'Failed to read the image using cv2'}, status=status.HTTP_400_BAD_REQUEST)
    return image

def preprocess_image(file):
    """Preprocesses the uploaded file

    Args:
        file (File): The file to preprocess

    Returns:
        numpy.ndarray: The preprocessed and normalized grayscale image
    """
    image = load_image(file)
    image = threshold_image(image)
    image = normalize_image(image)
    return image
