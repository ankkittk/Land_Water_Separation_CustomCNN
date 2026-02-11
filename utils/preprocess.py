import cv2
import numpy as np

def load_image(path):
    image = cv2.imread(path)
    return image

def to_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def resize_image(image, size = (256, 256)):
    resized = cv2.resize(image, size)
    return resized

def normalize_image(image):
    image = image.astype(np.float32) / 255.0
    return image

def preprocess_pipeline(path, size = (256, 256)):
    image = load_image(path)
    image = to_grayscale(image)
    image = resize_image(image)
    image = normalize_image(image)
    return image