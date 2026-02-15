import cv2
import numpy as np

def classify(mean_value, variance_value, T_mean=1.1, T_var=1.5):

    """
    Simple deterministic rule:
    Water → low mean + low variance
    Land  → otherwise
    """

    if mean_value < T_mean and variance_value < T_var:
        return "Water"
    else:
        return "Land"