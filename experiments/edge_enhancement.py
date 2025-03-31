import os
import cv2
import numpy as np

input_dir = '../model/test/test_data/R100H/inputcrop'
output_dir = '../outputs/edge_nlm_all'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing {filename}...")
        input_path = os.path.join(input_dir, filename)
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Could not load {filename}, skipping.")
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.medianBlur(gray_image, 7)

        canny_edges = cv2.Canny(gray_image, 100, 250)

        contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rain_mask = np.zeros_like(canny_edges)
        min_edge_length = 50
        for contour in contours:
            if cv2.arcLength(contour, closed=False) < min_edge_length:
                cv2.drawContours(rain_mask, [contour], -1, 255, thickness=-1)

        darkened_image = image.copy()
        darkened_image[rain_mask == 255] = (darkened_image[rain_mask == 255] * 0.6).astype(np.uint8)

        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=7)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=7)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        sobel_edges = np.uint8(np.absolute(sobel_edges))

        edges = cv2.addWeighted(sobel_edges, 0.6, canny_edges, 0.4, 0)

        enhanced_image = cv2.addWeighted(darkened_image, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.25, -30)

        denoised_image = cv2.bilateralFilter(enhanced_image, d=9, sigmaColor=75, sigmaSpace=75)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, denoised_image)

print("All images processed and saved")
