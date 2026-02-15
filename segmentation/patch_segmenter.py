import cv2
import numpy as np

from features.statistics import compute_stats
from rules.decision_rules import classify

def segment_image(image, model, patch_size=16):

    h, w = image.shape
    mask = np.zeros((h, w))

    stride = int(patch_size / 4)

    for i in range(0, h, stride):
        for j in range(0, w, stride):

            patch = image[i:i+patch_size, j:j+patch_size]

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            feature_maps = model.forward(patch)

            mean_val, var_val, all_means = compute_stats(feature_maps)
            #print(mean_val, var_val, all_means)

            label = classify(mean_val, var_val, all_means)

            if label == "Water":
                mask[i:i+patch_size, j:j+patch_size] = 255

    return mask