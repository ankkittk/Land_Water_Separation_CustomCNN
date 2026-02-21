import cv2
import numpy as np

def classify(mean_val,
             var_val,
             local_var,
             edge_density,
             blue_dominance,
             green_variation):

    score = 0

    # Texture smoothness (water tends to be smoother)
    if local_var < 0.02:
        score += 2

    if edge_density < 0.08:
        score += 2

    # CNN response smoothness
    if var_val < 3.0:
        score += 1

    # Blue dominance (helps blue ocean)
    if blue_dominance > 5:
        score += 1

    # Low green variation (land usually has high green variation)
    if green_variation < 500:
        score += 1

    if score >= 3:
        return "Water"
    else:
        return "Land"
