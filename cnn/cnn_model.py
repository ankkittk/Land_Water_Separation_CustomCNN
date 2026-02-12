import cv2
import numpy as np
from cnn.conv import convolve
from cnn.activation import relu
from cnn.pooling import max_pooling


class CustomCNN:

    def __init__(self):

        # -------- Layer 1 Kernels --------
        self.layer1_kernels = [
            np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]),  # Horizontal edge

            np.array([
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]
            ])   # Vertical edge
        ]

        # -------- Layer 2 Kernels --------
        self.layer2_kernels = [
            np.array([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]),  # Aggregation filter

            np.array([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ])   # Structural emphasis
        ]


    def apply_filters(self, image, kernels):
        outputs = []
        for kernel in kernels:
            feature = convolve(image, kernel)
            outputs.append(feature)
        
        return outputs


    def forward(self, image):

        # ---------- Layer 1 ----------
        feature_maps1 = self.apply_filters(image, self.layer1_kernels)

        activated1 = []
        for fm in feature_maps1:
            activated1.append(relu(fm))

        pooled1 = []
        for fm in activated1:
            pooled1.append(max_pooling(fm))


        # ---------- Layer 2 ----------
        feature_maps2 = []
        for fm in pooled1:
            feature_maps2.extend(self.apply_filters(fm, self.layer2_kernels))

        activated2 = []
        for fm in feature_maps2:
            activated2.append(relu(fm))

        pooled2 = []
        for fm in activated2:
            pooled2.append(max_pooling(fm))


        return pooled2
