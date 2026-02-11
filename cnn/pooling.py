import numpy as np

def max_pooling(activated_map):
    h, w = activated_map.shape

    out_h = h // 2
    out_w = w // 2

    pooled_map = np.zeros((out_h, out_w))

    for i in range(0, h, 2):
        for j in range(0, w, 2):
            pooled_map[i // 2][j // 2] = max(
                activated_map[i][j],
                activated_map[i + 1][j],
                activated_map[i][j + 1],
                activated_map[i + 1][j + 1]
            )

    return pooled_map