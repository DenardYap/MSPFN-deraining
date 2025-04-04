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

def apply_wiener_color(img, kernel_size=(10, 10)):
    filtered_channels = []
    for i in range(3):  # loop over R, G, B
        channel = img[:, :, i]
        filtered_channel = wiener(channel, kernel_size)
        filtered_channels.append(filtered_channel)
    return cv2.merge(filtered_channels)

img = cv2.imread("/Users/gennadumont/Downloads/MSPFN-deraining/diffusion/giraffe_rain.png")

filtered_image = apply_wiener_color(img, (10, 10))

"""
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
"""

def generate_fft(image_path):
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
    phase_spectrum = cv2.merge([phase_r, phase_g, phase_b])
    orig_magnitude_spectrum = cv2.merge([mag_r, mag_g, mag_b])

    return magnitude_spectrum, phase_spectrum, orig_magnitude_spectrum

def generate_fft_from_image(img):
    if img is None:
        raise ValueError("Image is None")

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
    phase_spectrum = cv2.merge([phase_r, phase_g, phase_b])
    orig_magnitude_spectrum = cv2.merge([mag_r, mag_g, mag_b])

    return magnitude_spectrum, phase_spectrum, orig_magnitude_spectrum


def reconstruct_from_fft_filtered(input_npz="fft_data.npz", output_image="reconstructed_filtered.png", cutoff=30, order=1):
    data = np.load(input_npz)
    mag_r, phase_r = data["mag_r"], data["phase_r"]
    mag_g, phase_g = data["mag_g"], data["phase_g"]
    mag_b, phase_b = data["mag_b"], data["phase_b"]

    # Apply Gaussian low-pass filter
    filter = gennas_test(mag_r.shape, fraction=0.25, inner_value=1.0, outer_value=0.7) * \
                  gaussian_lowpass_filter(mag_r.shape, D0=60, min_value=0.3)


    mag_r_filtered = mag_r * filter
    mag_g_filtered = mag_g * filter
    mag_b_filtered = mag_b * filter

    # Save FFT visualization *after* applying the filter
    save_fft_visualization(mag_r_filtered, mag_g_filtered, mag_b_filtered, output_image="rain_fft_filtered.jpg")

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




#######EVERYTHING UNDER THIS WAS ADDED BY GENNA AND COULD BE WRONG 

def gaussian_lowpass_filter(shape, D0, min_value=0.3):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.ogrid[:rows, :cols]
    distance_squared = (x - crow)**2 + (y - ccol)**2
    base_filter = np.exp(-distance_squared / (2 * (D0**2)))
    
    # Scale so it ranges from [min_value, 1] instead of [0, 1]
    return base_filter * (1 - min_value) + min_value


def save_fft_visualization(mag_r, mag_g, mag_b, output_image="fft_filtered.jpg"):
    mag_r_spectrum = np.log1p(mag_r)
    mag_g_spectrum = np.log1p(mag_g)
    mag_b_spectrum = np.log1p(mag_b)
    mag_r_spectrum = (mag_r_spectrum / np.max(mag_r_spectrum) * 255).astype(np.uint8)
    mag_g_spectrum = (mag_g_spectrum / np.max(mag_g_spectrum) * 255).astype(np.uint8)
    mag_b_spectrum = (mag_b_spectrum / np.max(mag_b_spectrum) * 255).astype(np.uint8)
    spectrum_image = cv2.merge([mag_b_spectrum, mag_g_spectrum, mag_r_spectrum])
    cv2.imwrite(output_image, spectrum_image)

def gennas_test(image_shape, fraction=0.25, inner_value=1.0, outer_value=0.3):
    """
    Creates a filter where the center box has `inner_value` and the rest has `outer_value`.

    Parameters:
        image_shape: tuple like (rows, cols) or (rows, cols, channels)
        fraction: how large the center box should be (relative to image size)
        inner_value: value inside the box (usually 1)
        outer_value: value outside the box (e.g., 0.7)

    Returns:
        filter: 2D array of same shape as image, with two distinct values
    """
    rows, cols = image_shape[:2]
    box_h = int(rows * fraction)
    box_w = int(cols * fraction)
    
    start_row = (rows - box_h) // 2
    start_col = (cols - box_w) // 2

    # Start everything at outer_value
    filter = np.full((rows, cols), outer_value, dtype=float)

    # Set center box to inner_value
    filter[start_row:start_row + box_h, start_col:start_col + box_w] = inner_value

    return filter

def vertical_average_filter(image, kernel_size=5):
    """
    Applies a vertical average (box) filter to the image.
    Only vertical blurring is applied — horizontal direction is untouched.

    Parameters:
        image: 2D (grayscale) or 3D (color) NumPy array
        kernel_size: number of vertical pixels to average (should be odd)

    Returns:
        filtered_image: vertically averaged image
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    # Create a vertical kernel (e.g., [1, 1, 1, ..., 1]T)
    kernel = np.ones((kernel_size, 1), dtype=np.float32) / kernel_size

    # Apply filter using OpenCV
    return cv2.filter2D(image, -1, kernel)

"""
image = cv2.imread("rain.png")
mask_filter = gennas_test(image.shape, fraction=0.25, inner_value=1.0, outer_value=0.7)



# Generate FFT data and images
generate_fft("clean.png", "clean_fft.jpg", "clean_fft.npz")
generate_fft("rain.png", "rain_fft.jpg", "rain_fft.npz")

# Reconstruct rain image using Gaussian low-pass filter
reconstruct_from_fft_filtered("rain_fft.npz", "rain_filtered.png")

# Load and convert images to RGB
def load_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

clean = load_rgb("clean.png")
clean_fft = load_rgb("clean_fft.jpg")
rain = load_rgb("rain.png")
rain_fft = load_rgb("rain_fft.jpg")
rain_filtered = load_rgb("rain_filtered.png")
rain_fft_filtered = load_rgb("rain_fft_filtered.jpg")

# Plot side-by-side
plt.figure(figsize=(10, 8))

titles = [
    "Clean Image", "Clean FFT",
    "Rain Image", "Rain FFT",
    "Rain Filtered (Time Domain)", "Rain FFT (after Gaussian Filter)"
]
images = [
    clean, clean_fft,
    rain, rain_fft,
    rain_filtered, rain_fft_filtered
]

for i in range(6):
    plt.subplot(3, 2, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i])
    plt.axis("off")

plt.tight_layout()
plt.show()"""