# Methodology

## 1. Preprocessing
- RGB image loading
- Grayscale conversion
- Resizing to fixed resolution
- Intensity normalization

## 2. Custom CNN Architecture
Two convolutional blocks are implemented manually:
- Layer 1: Edge detection filters (3×3)
- ReLU activation
- Max pooling (2×2)
- Layer 2: Texture aggregation filters (5×5)
- ReLU activation
- Max pooling

The CNN is implemented without training or backpropagation.

## 3. Patch-Based Segmentation
- Patch size: 16×16
- Stride: Patch size / 4
- For each patch:
  - Extract feature maps
  - Compute mean and variance
  - Apply deterministic thresholds

## 4. Boundary Extraction
Contours are extracted from binary masks and overlaid as red boundaries on the original image.
