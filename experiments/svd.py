import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("giraffe.png", cv2.IMREAD_GRAYSCALE)
image_rain = cv2.imread("giraffe_rain.png", cv2.IMREAD_GRAYSCALE)
U, S, Vt = np.linalg.svd(image, full_matrices=False)
U_rain, S_rain, Vt_rain = np.linalg.svd(image_rain, full_matrices=False)
# Plot singular values
plt.figure(figsize=(8, 5))
plt.plot(S_rain, 'ro-', label="Singular values of rainy image")

plt.subplot(1, 2, 1)
plt.plot(S, 'bo-', label="Original Image")
plt.plot(S, 'bo-', label="Singular Values of original image")
plt.xlabel("Index")
plt.ylabel("Singular Value Magnitude")
plt.title("Singular Values - Original Image")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(S_rain, 'ro-', label="Singular values of rainy image")
plt.xlabel("Index")
plt.ylabel("Singular Value Magnitude")
plt.title("Singular Values - Rainy Image")
plt.legend()
plt.grid()

# Show the plots
plt.tight_layout()
plt.show()


k = 100 

S_k = np.diag(S_rain[:k])
U_k = U_rain[:, :k]
Vt_k = Vt_rain[:k, :]

reconstructed_image = np.dot(U_k, np.dot(S_k, Vt_k))
reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
cv2.imwrite("reconstructed.jpg", reconstructed_image)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rain, cmap="gray")
plt.title("Rain Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap="gray")
plt.title(f"Reconstructed Image (k={k})")
plt.axis("off")

plt.tight_layout()
plt.show()

