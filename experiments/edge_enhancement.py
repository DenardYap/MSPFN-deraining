import os
import cv2
import numpy as np

# Updated input/output paths
input_dir = '../model/test/test_data/R100H'
output_dir = '../outputs/edge_enhanced'

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all images
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Processing {filename}...")

        # Load image
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping {filename}: could not load.")
            continue

        # Convert to grayscale and blur
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.medianBlur(gray_image, 7)

        # Canny edge detection
        canny_edges = cv2.Canny(gray_image, 100, 250)

        # Find short contours (likely rain)
        contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rain_mask = np.zeros_like(canny_edges)
        min_edge_length = 50
        for contour in contours:
            length = cv2.arcLength(contour, closed=False)
            if length < min_edge_length:
                cv2.drawContours(rain_mask, [contour], -1, 255, thickness=-1)

        # Darken rain areas
        darkened_image = image.copy()
        darkened_image[rain_mask == 255] = (darkened_image[rain_mask == 255] * 0.6).astype(np.uint8)

        # Sobel edge detection
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=7)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=7)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        sobel_edges = np.uint8(np.absolute(sobel_edges))

        # Combine edge maps
        edges = cv2.addWeighted(sobel_edges, 0.6, canny_edges, 0.4, 0)

        # Final enhancement
        enhanced_image = cv2.addWeighted(darkened_image, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.25, -30)

        # Save output
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, enhanced_image)

print(" Done processing all images!")
