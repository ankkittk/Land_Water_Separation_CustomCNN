import numpy as np

def relu(feature_map):
    relu_img = np.zeros(feature_map.shape)
    h, w = feature_map.shape
    for i in range(h):
        for j in range(w):
            relu_img[i][j] = max(0, feature_map[i][j])

    return relu_img