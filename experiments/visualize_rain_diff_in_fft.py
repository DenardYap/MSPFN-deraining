import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 
import numpy as np 
from numpy.fft import fft2, fftshift

def generate_fft(img_path):
    
    img = cv2.imread(img_path)

    if img is None:  # If the image is not read successfully
        print(f"Error: Could not read image at {img_path}")
        raise
    
    b, g, r = cv2.split(img)
    fft_r = np.fft.fft2(r)
    fft_g = np.fft.fft2(g)
    fft_b = np.fft.fft2(b)
    fft_r_shifted = np.fft.fftshift(fft_r)
    fft_g_shifted = np.fft.fftshift(fft_g)
    fft_b_shifted = np.fft.fftshift(fft_b)
    mag_r, phase_r = np.abs(fft_r_shifted), np.angle(fft_r_shifted)
    mag_g, phase_g = np.abs(fft_g_shifted), np.angle(fft_g_shifted)
    mag_b, phase_b = np.abs(fft_b_shifted), np.angle(fft_b_shifted)
    mag_r_spectrum = np.log1p(mag_r)
    mag_g_spectrum = np.log1p(mag_g)
    mag_b_spectrum = np.log1p(mag_b)
    mag_r_spectrum = (mag_r_spectrum / np.max(mag_r_spectrum) * 255).astype(np.uint8)
    mag_g_spectrum = (mag_g_spectrum / np.max(mag_g_spectrum) * 255).astype(np.uint8)
    mag_b_spectrum = (mag_b_spectrum / np.max(mag_b_spectrum) * 255).astype(np.uint8)
    magnitude_spectrum = cv2.merge([mag_b_spectrum, mag_g_spectrum, mag_r_spectrum])
    return magnitude_spectrum

# Original FFT, light FFT, medium FFT, heavy FFT
fig, axes = plt.subplots(6, 4, figsize=(9, 18))

types = ["light", "medium", "heavy"]
prefixes = [0, 1, 2, 3, 4, 5]

# first display normal image and its fft
orig_fft_images = []
for idx, prefix in enumerate(prefixes): 
    fft_img = generate_fft(f"rain_images/light/{prefix}.jpg_derain.png")
    orig_fft_images.append(fft_img)
    ax = axes[idx, 0]
    ax.imshow(fft_img )
    ax.axis('off')  # Hide axes

for idx, type_ in enumerate(types):
    for jdx, prefix in enumerate(prefixes):
        ax = axes[jdx, idx + 1]
        fft_img = generate_fft(f"rain_images/{type_}/{prefix}.jpg_rain.png")
        ax.imshow(orig_fft_images[jdx] - fft_img)
        ax.axis('off')  # Hide axes

axes[0, 0].set_title("Normal Image FFT")
for idx, type_ in enumerate(types):
    axes[0, idx + 1].set_title(type_.capitalize() + " Difference")

plt.tight_layout()
# plt.show()
plt.savefig("fft_viz_diff.png")