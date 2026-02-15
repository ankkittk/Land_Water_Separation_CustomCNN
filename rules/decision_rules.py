import cv2
import numpy as np

def classify(mean_value, variance_value, all_means, T_mean=1.28, T_var=4.22):

    """
    Simple deterministic rule:
    Water → low mean + low variance
    Land  → otherwise
    """

    #T_mean = all_means

    if mean_value < T_mean and variance_value < T_var:
        return "Water"
    else:
        return "Land"