import numpy as np
from cnn.conv import convolve
from cnn.activation import relu


class CustomCNN:

    def __init__(self):

        # Multi-directional filter bank
        self.kernels = [

            # Sobel X
            np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]),

            # Sobel Y
            np.array([
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]
            ]),

            # Laplacian
            np.array([
                [0,  1, 0],
                [1, -4, 1],
                [0,  1, 0]
            ]),

            # Diagonal filter
            np.array([
                [2, -1, -1],
                [-1, 2, -1],
                [-1, -1, 2]
            ])
        ]

    def forward(self, image):

        feature_maps = []

        for kernel in self.kernels:
            conv = convolve(image, kernel)
            activated = relu(conv)
            feature_maps.append(np.abs(activated))

        return feature_maps
