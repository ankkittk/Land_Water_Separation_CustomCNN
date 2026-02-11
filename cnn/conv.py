import numpy as np

def correlation(padded, kernel, p, q):
    total = 0

    k_h, k_w = kernel.shape
    a = k_h // 2
    b = k_w // 2

    for i in range(-a, a + 1):
        for j in range(-b, b + 1):
            total += padded[p + i][q + j] * kernel[i + a][j + b]

    return total

def convolve(image, kernel):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    padded_h = img_h + 2 * pad_h
    padded_w = img_w + 2 * pad_w

    padded = np.zeros((padded_h, padded_w))

    for i in range(img_h):
        for j in range(img_w):
            padded[i + pad_h][j + pad_w] = image[i][j]

    output = np.zeros((img_h, img_w))

    for i in range(pad_h, pad_h + img_h):
        for j in range(pad_w, pad_w + img_w):
            output[i - pad_h][j - pad_w] = correlation(padded, kernel, i, j)

    return output