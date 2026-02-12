import cv2
import numpy as np
from cnn.conv import convolve
from cnn.activation import relu
from cnn.pooling import max_pooling

class CustomCNN:

    def __init__(self):
        self.kernel1 = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

    def forward(self, image):
        conv1 = convolve(image, self.kernel1)
        relu1 = relu(conv1)
        pool1 = max_pooling(relu1)

        return pool1