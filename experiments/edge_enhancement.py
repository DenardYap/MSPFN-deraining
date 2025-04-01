import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('1.png', cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Detect horizontal edges
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Detect vertical edges

edges = cv2.magnitude(sobel_x, sobel_y)
edges = np.uint8(np.absolute(edges))
enhanced_image = cv2.addWeighted(image, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.5, 0)

plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Enhanced Image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.title('Edge Enhanced Image')
plt.axis('off')
plt.savefig("1_enhanced.png")