import os
import cv2
import numpy as np

# Input/output folders
input_dir = '../model/test/test_data/R100H/inputcrop'
output_dir = '../outputs/edge_enhanced'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Processing {filename}...")

        # Load image
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not load {filename}")
            continue

        # --- Unsharp mask ---
        blurred = cv2.GaussianBlur(image, (9, 9), 10)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

        # --- Grayscale + blur ---
        gray_image = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.medianBlur(gray_image, 7)

        # --- CLAHE (Contrast Limited Adaptive Histogram Equalization) ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_image = clahe.apply(gray_image)

        # --- Edge detection ---
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=7)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=7)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        sobel_edges = np.uint8(np.absolute(sobel_edges))

        canny_edges = cv2.Canny(gray_image, 100, 250)

        # Combine + clip
        edges = cv2.addWeighted(sobel_edges, 0.6, canny_edges, 0.4, 0)
        edges = np.clip(edges, 0, 100).astype(np.uint8)

        # Blend edges into sharpened image
        enhanced_image = cv2.addWeighted(sharpened, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.2, 0)

        # Save result
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, enhanced_image)

print(" All images processed and saved to:", output_dir)

