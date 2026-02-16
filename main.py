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

    output = cv2.cvtColor(original_bgr, cv2.COLOR_GRAY2BGR)
    output = cv2.drawContours(output, contours, -1, (0, 0, 255), 2)

    return output


def run_pipeline(image_path):

    # Load original color image (for boundary drawing)
    original_bgr = cv2.imread(image_path)

    # Preprocess using utils
    gray = preprocess_pipeline(image_path)

    # Model
    model = CustomCNN()

    # Segmentation
    mask = segment_image(gray, model)

    # Boundary overlay
    result = extract_boundary(gray, mask)

    return result
