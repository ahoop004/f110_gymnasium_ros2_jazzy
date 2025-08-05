import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import maximum_filter

def extract_track_mask(img, threshold=127):
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)
    return (binary // 255).astype(np.uint8)

def get_centerline_from_mask(mask):
    # Distance transform: "ridge" is track center
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Find local maxima: these are centerline candidates
    maxima = (dist == maximum_filter(dist, size=15))
    centerline = np.column_stack(np.where(maxima & (dist > 0)))
    # (Optional) Order by projecting onto skeleton or contour—could add
    return centerline

def plot_results(img, mask, centerline):
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='gray')
    plt.imshow(mask, cmap='gray', alpha=0.3)
    plt.scatter(centerline[:,1], centerline[:,0], c='r', s=1)
    plt.title('Centerline Points')
    plt.gca().invert_yaxis()
    plt.show()

def save_centerline_csv(centerline, path='centerline.csv'):
    df = pd.DataFrame(centerline, columns=['y', 'x'])
    df.to_csv(path, index=False)
    print(f"Saved centerline points to {path}")

if __name__ == "__main__":
    # Use your uploaded image
    img_path = '/home/aaron/f110_gymnasium_ros2_jazzy/rl_training/maps/Shanghai_map.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = extract_track_mask(img)
    centerline = get_centerline_from_mask(mask)
    plot_results(img, mask, centerline)
    save_centerline_csv(centerline)
