import cv2
import numpy as np

from utils.preprocess import preprocess_pipeline
from cnn.cnn_model import CustomCNN
from segmentation.patch_segmenter import segment_image


def extract_boundary(original_bgr, mask):

    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw on original color image
    output = original_bgr.copy()
    cv2.drawContours(output, contours, -1, (0, 0, 255), 2)

    return output


def run_pipeline(image_path):

    original_bgr = cv2.imread(image_path)

    gray = preprocess_pipeline(image_path)

    h, w = gray.shape

    original_resized = cv2.resize(original_bgr, (w, h))

    model = CustomCNN()

    mask = segment_image(gray, original_bgr, model)

    result = extract_boundary(original_resized, mask)

    return original_resized, result
