import cv2
import numpy as np
import matplotlib.pyplot as plt

def directional_filter_color(image, angle_range=(85, 95)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    f = np.fft.fft2(l)
    fshift = np.fft.fftshift(f)

    rows, cols = l.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)

    for y in range(rows):
        for x in range(cols):
            angle = np.degrees(np.arctan2(y - crow, x - ccol))
            angle = (angle + 360) % 180
            if angle_range[0] <= angle <= angle_range[1]:
                mask[y, x] = 0

    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    l_filtered = np.abs(np.fft.ifft2(f_ishift))
    l_filtered = cv2.normalize(l_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    lab_filtered = cv2.merge((l_filtered, a, b))
    color_filtered = cv2.cvtColor(lab_filtered, cv2.COLOR_LAB2BGR)

    return color_filtered

def refine_with_edge_enhancement(image):
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

    return denoised_image

def selective_brighten(image, threshold=170, brighten_factor=1.5):
    image_float = image.astype(np.float32)
    intensity = np.mean(image_float, axis=2)
    mask = intensity < threshold
    mask_3d = np.stack([mask]*3, axis=-1)
    image_float[mask_3d] *= brighten_factor
    image_float = np.clip(image_float, 0, 255)
    return image_float.astype(np.uint8)

def smart_sharpen(image, contrast_thresh=10, brightness_thresh=230, amount=1.0):
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect edges (you could also try cv2.Laplacian or cv2.Sobel here)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    contrast_mask = np.abs(laplacian) > contrast_thresh
    
    # Optionally avoid sharpening very bright areas (e.g. rain streaks)
    brightness_mask = gray < brightness_thresh

    # Combine masks
    final_mask = np.logical_and(contrast_mask, brightness_mask)

    # Expand mask to 3 channels
    mask_3d = np.stack([final_mask]*3, axis=-1)

    # Unsharp mask-style sharpening
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)

    # Apply sharpening only where mask is True
    result = np.where(mask_3d, sharpened, image)

    return result.astype(np.uint8)

def enhance_details_bilateral(image, sigma_color=30, sigma_space=30, detail_boost=1.5):
    # Smooth base layer
    base = cv2.bilateralFilter(image, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    # Detail layer = original - base
    detail = cv2.subtract(image, base)

    # Boost detail and add back
    boosted = cv2.addWeighted(base, 1.0, detail, detail_boost, 0)

    return boosted

def horizontal_blur(image, kernel_width=15):
    return cv2.blur(image, ksize=(kernel_width, 1))


image_path = "../model/test/test_data/R100H/inputcrop/1.png"
original = cv2.imread(image_path)

filtered_freq = directional_filter_color(original)
combined_result = refine_with_edge_enhancement(filtered_freq)
brightened_result = selective_brighten(combined_result, threshold=100, brighten_factor=1.4)
enhance_details_result = enhance_details_bilateral(brightened_result, sigma_color=30, sigma_space=30, detail_boost=1.5)
horiz_blurred = horizontal_blur(brightened_result, kernel_width=5)
brightened_result = selective_brighten(combined_result, threshold=80, brighten_factor=8)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(horiz_blurred, cv2.COLOR_BGR2RGB))
plt.title("Combined Result")
plt.axis('off')

plt.tight_layout()
plt.show()






#THESE ARE PREVIOUS METHODS THAT I WANT TO KEEP JUST IN CASE 

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def directional_filter_color(image, angle_range=(85, 95)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # FFT and shift
    f = np.fft.fft2(l)
    fshift = np.fft.fftshift(f)

    # Create a directional mask
    rows, cols = l.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)

    for y in range(rows):
        for x in range(cols):
            angle = np.degrees(np.arctan2(y - crow, x - ccol))
            angle = (angle + 360) % 180  # Wrap to [0, 180)
            if angle_range[0] <= angle <= angle_range[1]:
                mask[y, x] = 0

    # Apply directional mask
    fshift_filtered = fshift * mask

    # Inverse FFT to reconstruct filtered luminance
    f_ishift = np.fft.ifftshift(fshift_filtered)
    l_filtered = np.abs(np.fft.ifft2(f_ishift))
    l_filtered = cv2.normalize(l_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Merge back with original color channels
    lab_filtered = cv2.merge((l_filtered, a, b))
    color_filtered = cv2.cvtColor(lab_filtered, cv2.COLOR_LAB2BGR)

    return color_filtered

# Load image
image_path = "../model/test/test_data/R100H/inputcrop/1.png"
original = cv2.imread(image_path)
filtered = directional_filter_color(original, angle_range=(85, 95))  # Targeting vertical rain

# Display
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
plt.title("Directional Filter (Color Preserved)")
plt.axis('off')
plt.tight_layout()
plt.show()
"""

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "../model/test/test_data/R100H/inputcrop/1.png"
original = cv2.imread(image_path)

if original is None:
    print(f"Could not load {original}, skipping.")

gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gray_image = cv2.medianBlur(gray_image, 7)

canny_edges = cv2.Canny(gray_image, 100, 250)

contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rain_mask = np.zeros_like(canny_edges)
min_edge_length = 50
for contour in contours:
    if cv2.arcLength(contour, closed=False) < min_edge_length:
        cv2.drawContours(rain_mask, [contour], -1, 255, thickness=-1)

darkened_image = original.copy()
darkened_image[rain_mask == 255] = (darkened_image[rain_mask == 255] * 0.6).astype(np.uint8)

sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=7)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=7)
sobel_edges = cv2.magnitude(sobel_x, sobel_y)
sobel_edges = np.uint8(np.absolute(sobel_edges))

edges = cv2.addWeighted(sobel_edges, 0.6, canny_edges, 0.4, 0)

enhanced_image = cv2.addWeighted(darkened_image, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.25, -30)

denoised_image = cv2.bilateralFilter(enhanced_image, d=9, sigmaColor=75, sigmaSpace=75)


# Display
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
plt.title("edge enhancement and bilateral")
plt.axis('off')
plt.tight_layout()
plt.show()
  """