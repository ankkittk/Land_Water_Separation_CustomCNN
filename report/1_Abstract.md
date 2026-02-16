# Land–Water Segmentation Using a Custom CNN Architecture

## Abstract

This project presents a deterministic land–water segmentation framework using a custom-built Convolutional Neural Network (CNN) implemented without backpropagation. 
The CNN is used strictly as a hierarchical feature extractor with handcrafted filters, and final classification is performed using statistical rule-based thresholds.

The system leverages physical surface properties such as reflectance smoothness and edge density to distinguish between land and water regions. 
Patch-wise segmentation is performed using mean and variance statistics extracted from CNN feature maps. 
The final output highlights detected land–water boundaries on the original image.
