import numpy as np
import cv2
from features.statistics import compute_stats


def segment_image(gray_image, color_image, model, patch_size=16):

    h, w = gray_image.shape
    stride = int(patch_size / 4)

    water_score_map = np.zeros((h, w), dtype=np.float32)

    # ---------- PASS 1: Compute Water Score ----------
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):

            patch_gray = gray_image[i:i+patch_size, j:j+patch_size]
            patch_color = color_image[i:i+patch_size, j:j+patch_size]

            # Texture features from custom CNN
            feature_maps = model.forward(patch_gray)
            mean_val, var_val, _ = compute_stats(feature_maps)

            # Smoothness score (lower variance = smoother = more water-like)
            smoothness = 1.0 / (var_val + 1e-6)

            # Blue dominance
            B_mean = np.mean(patch_color[:, :, 0])
            R_mean = np.mean(patch_color[:, :, 2])
            blue_score = max(0, B_mean - R_mean)

            # Edge energy (Sobel gradient)
            gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
            gradient_energy = np.mean(np.sqrt(gx**2 + gy**2))

            # Final water score (tunable weights)
            water_score = (
                0.6 * smoothness +
                0.3 * blue_score -
                0.5 * gradient_energy
            )

            water_score_map[i:i+patch_size, j:j+patch_size] += water_score

    # ---------- Normalize Score ----------
    water_score_map = cv2.normalize(
        water_score_map,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    # ---------- Threshold ----------
    _, mask = cv2.threshold(
        water_score_map,
        0.5,     # can tune slightly (0.45–0.6)
        1,
        cv2.THRESH_BINARY
    )

    mask = (mask * 255).astype(np.uint8)

    # ---------- Keep Only Largest Connected Component ----------
    num_labels, labels = cv2.connectedComponents(mask)

    if num_labels > 1:
        areas = []
        for lbl in range(1, num_labels):
            area = np.sum(labels == lbl)
            areas.append((lbl, area))

        areas.sort(key=lambda x: x[1], reverse=True)

        largest_label = areas[0][0]

        clean_mask = np.zeros_like(mask)
        clean_mask[labels == largest_label] = 255
    else:
        clean_mask = mask

    # ---------- Morphological Smoothing ----------
    kernel = np.ones((7, 7), np.uint8)

    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    return clean_mask
