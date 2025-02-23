import numpy as np
import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import wiener

def get_filter(rows, cols, D0=30):
    crow, ccol = rows // 2, cols // 2  # Center of the frequency domain

    x, y = np.ogrid[:rows, :cols]
    mask = np.exp(-((x - crow)**2 + (y - ccol)**2) / (2 * D0**2))

def butterworth_highpass_filter(shape, cutoff, order=2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
    d = np.sqrt(x**2 + y**2)
    return 1 / (1 + (cutoff / d) ** (2 * order))
def ideal_highpass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
    d = np.sqrt(x**2 + y**2)
    mask = np.ones((rows, cols))
    mask[d < cutoff] = 0  # Zero out low frequencies
    return mask

image = cv2.imread("giraffe_rain.png", cv2.IMREAD_GRAYSCALE)

filtered_image = wiener(image, (10, 10)) 
filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

# Display results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Filtered Image")
plt.imshow(filtered_image, cmap="gray")
plt.axis("off")


plt.savefig("filtered.png")


def generate_fft(image_path, output_image="fft_image.png", output_npz="fft_data.npz"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image. Check the path.")

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
    cv2.imwrite(output_image, magnitude_spectrum)
    np.savez(output_npz, mag_r=mag_r, phase_r=phase_r, mag_g=mag_g, phase_g=phase_g, mag_b=mag_b, phase_b=phase_b)

    print(f"FFT color image saved as: {output_image}")
    print(f"FFT data (magnitude & phase) saved as: {output_npz}")

def reconstruct_from_fft_filtered(input_npz="fft_data.npz", output_image="reconstructed_filtered.png", cutoff=30, order=1):
    data = np.load(input_npz)
    mag_r, phase_r = data["mag_r"], data["phase_r"]
    mag_g, phase_g = data["mag_g"], data["phase_g"]
    mag_b, phase_b = data["mag_b"], data["phase_b"]

    filter =  1 - butterworth_highpass_filter(mag_r.shape, cutoff, order)
    # filter = 1 - ideal_highpass_filter(mag_r.shape, cutoff)

    # Apply filter to magnitude
    mag_r_filtered = mag_r * filter
    mag_g_filtered = mag_g * filter
    mag_b_filtered = mag_b * filter

    # Reconstruct the complex FFT for each channel
    fft_r = mag_r_filtered * np.exp(1j * phase_r)
    fft_g = mag_g_filtered * np.exp(1j * phase_g)
    fft_b = mag_b_filtered * np.exp(1j * phase_b)

    # Perform inverse FFT for each channel
    ifft_r = np.fft.ifft2(np.fft.ifftshift(fft_r))
    ifft_g = np.fft.ifft2(np.fft.ifftshift(fft_g))
    ifft_b = np.fft.ifft2(np.fft.ifftshift(fft_b))

    # Get real values (discard imaginary part)
    r_channel = np.abs(ifft_r)
    g_channel = np.abs(ifft_g)
    b_channel = np.abs(ifft_b)

    # Normalize each channel to [0, 255]
    r_channel = (r_channel / np.max(r_channel) * 255).astype(np.uint8)
    g_channel = (g_channel / np.max(g_channel) * 255).astype(np.uint8)
    b_channel = (b_channel / np.max(b_channel) * 255).astype(np.uint8)

    # Merge the channels back into a color image
    reconstructed_image = cv2.merge([b_channel, g_channel, r_channel])

    # Save the reconstructed color image
    cv2.imwrite(output_image, reconstructed_image)

    print(f"Reconstructed color image with low-pass filter saved as: {output_image}")

def reconstruct_from_fft(input_npz="fft_data.npz", output_image="reconstructed_image.png"):
    # Load FFT data for each channel
    data = np.load(input_npz)
    mag_r, phase_r = data["mag_r"], data["phase_r"]
    mag_g, phase_g = data["mag_g"], data["phase_g"]
    mag_b, phase_b = data["mag_b"], data["phase_b"]

    # Reconstruct the complex FFT for each channel
    fft_r = mag_r * np.exp(1j * phase_r)
    fft_g = mag_g * np.exp(1j * phase_g)
    fft_b = mag_b * np.exp(1j * phase_b)

    # Perform inverse FFT for each channel
    ifft_r = np.fft.ifft2(np.fft.ifftshift(fft_r))
    ifft_g = np.fft.ifft2(np.fft.ifftshift(fft_g))
    ifft_b = np.fft.ifft2(np.fft.ifftshift(fft_b))

    # Get real values (discard imaginary part)
    r_channel = np.abs(ifft_r)
    g_channel = np.abs(ifft_g)
    b_channel = np.abs(ifft_b)

    # Normalize each channel to [0, 255]
    r_channel = (r_channel / np.max(r_channel) * 255).astype(np.uint8)
    g_channel = (g_channel / np.max(g_channel) * 255).astype(np.uint8)
    b_channel = (b_channel / np.max(b_channel) * 255).astype(np.uint8)

    # Merge the channels back into a color image
    reconstructed_image = cv2.merge([b_channel, g_channel, r_channel])

    # Save the reconstructed color image
    cv2.imwrite(output_image, reconstructed_image)

    print(f"Reconstructed color image saved as: {output_image}")

# generate_fft("giraffe.png", "giraffe_fft.jpg", "giraffe_fft.npz")
# generate_fft("giraffe_rain.png", "giraffe_rain_fft.jpg", "giraffe_rain_fft.npz")
# reconstruct_from_fft("giraffe_rain_fft.npz", "reconstructed_filtered_image.jpeg")
# reconstruct_from_fft_filtered("giraffe_rain_fft.npz", "reconstructed_filtered_image.jpeg")

import cv2
import numpy as np

# Load an image
image = cv2.imread('giraffe_rain.png')

# Apply Median Filter (kernel size 3, 5, 7, etc. depending on the amount of noise)
filtered_image = cv2.medianBlur(image, 5)
# filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
# filtered_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Display the filtered image
cv2.imshow('Median Filter', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
