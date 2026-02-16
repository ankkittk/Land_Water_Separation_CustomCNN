# Experiments and Analysis

## 1. Feature Map Visualization
Layer-wise outputs show:
- Strong responses on textured land
- Weak responses on smooth water surfaces

## 2. Mean-Variance Scatter Plot
Patches were plotted in (mean, variance) space.
Clear clustering behavior was observed:
- Water: Lower mean and lower variance
- Land: Higher mean and/or variance

## 3. Threshold Sensitivity
Optimal parameters determined experimentally:
- T_mean = 1.28
- T_var = 4.22

## 4. Patch Size Comparison
- 32×32 → Blocky segmentation
- 16×16 → Balanced performance
- 1×1 → No spatial context

## 5. Failure Cases
- Dark land patches occasionally misclassified as water.
