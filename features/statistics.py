import cv2
import numpy as np

def compute_stats(feature_maps):
    """
    feature_maps: list of 2D numpy arrays
    returns:
        overall_mean
        overall_variance
    """

    means = []
    variances = []

    for fm in feature_maps:
        means.append(np.mean(fm))
        variances.append(np.var(fm))
    
    overall_mean = np.mean(means)
    overall_variance = np.mean(variances)

    return overall_mean, overall_variance, np.median(means)