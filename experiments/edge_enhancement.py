import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
from image_similarity_measures.quality_metrics import fsim


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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    contrast_mask = np.abs(laplacian) > contrast_thresh
    brightness_mask = gray < brightness_thresh
    final_mask = np.logical_and(contrast_mask, brightness_mask)
    mask_3d = np.stack([final_mask]*3, axis=-1)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    result = np.where(mask_3d, sharpened, image)
    return result.astype(np.uint8)

def enhance_details_bilateral(image, sigma_color=30, sigma_space=30, detail_boost=1.5):
    base = cv2.bilateralFilter(image, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    detail = cv2.subtract(image, base)
    boosted = cv2.addWeighted(base, 1.0, detail, detail_boost, 0)
    return boosted

def horizontal_blur(image, kernel_width=15):
    return cv2.blur(image, ksize=(kernel_width, 1))

def suppress_high_freq_fft_Y(rain_img, suppression_strength=0.8):
    rain_ycrcb = cv2.cvtColor(rain_img, cv2.COLOR_BGR2YCrCb)
    rain_Y, rain_Cr, rain_Cb = cv2.split(rain_ycrcb)

    fft_Y = np.fft.fftshift(np.fft.fft2(rain_Y))
    mag = np.abs(fft_Y)
    phase = np.angle(fft_Y)

    rows, cols = rain_Y.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    max_dist = np.sqrt(crow**2 + ccol**2)
    suppression_mask = 1 - suppression_strength * (distance / max_dist)
    suppression_mask = np.clip(suppression_mask, 0, 1)

    new_mag = mag * suppression_mask

    suppressed_fft = new_mag * np.exp(1j * phase)
    new_Y = np.fft.ifft2(np.fft.ifftshift(suppressed_fft))
    new_Y = np.abs(new_Y).astype(np.uint8)

    new_ycrcb = cv2.merge([new_Y, rain_Cr, rain_Cb])
    final = cv2.cvtColor(new_ycrcb, cv2.COLOR_YCrCb2BGR)
    return final


rain_path = "/Users/gennadumont/Downloads/MSPFN-deraining/model/test/test_data/R100H/inputcrop/1.png"
clean_path = "/Users/gennadumont/Downloads/MSPFN-deraining/model/test/test_data/R100H/cleancrop/1.png"

rain = cv2.imread(rain_path)
if rain is None:
    raise FileNotFoundError(f"Failed to load image from {rain_path}")
else:
    print("Loaded image shape:", rain.shape)

#input_dir = '../model/test/test_data/R100H/inputcrop'
#output_dir = '../outputs/edge_directional_bilateral'

filtered_freq = directional_filter_color(rain)
combined_result = refine_with_edge_enhancement(filtered_freq)
enhance_details_result = enhance_details_bilateral(combined_result, sigma_color=30, sigma_space=30, detail_boost=1.5)
brightened_result = selective_brighten(enhance_details_result, threshold=100, brighten_factor=1.4)
horiz_blurred = horizontal_blur(brightened_result, kernel_width=5)
brightened_result = selective_brighten(horiz_blurred, threshold=80, brighten_factor=1.1)


"""
def calculate_psnr(clean_path, rain_path):
    clean = cv2.imread(clean_path)
    rain = cv2.imread(rain_path)

    mse = np.mean((clean - rain) ** 2)
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

psnr_value = calculate_psnr(clean_path, rain_path)
print(f"PSNR: {psnr_value:.2f} dB")
"""



def calculate_fsim(image_path1, image_path2):
    """Calculates the FSIM (Feature Similarity Index) between two images."""
    img1 = np.array(Image.open(image_path1).convert('RGB'))
    img2 = np.array(Image.open(image_path2).convert('RGB'))
    return fsim(img1, img2)

def calculate_psnr(image_path1, image_path2):
    """Calculates the PSNR (Peak Signal-to-Noise Ratio) between two grayscale images."""
    img1 = np.array(Image.open(image_path1).convert('L'))  # Convert to grayscale
    img2 = np.array(Image.open(image_path2).convert('L'))
    return psnr(img1, img2)

def calculate_ssim(image_path1, image_path2):
    """Calculates the SSIM (Structural Similarity Index) between two grayscale images."""
    img1 = np.array(Image.open(image_path1).convert('L'))  # Convert to grayscale
    img2 = np.array(Image.open(image_path2).convert('L'))
    return ssim(img1, img2)

fsim_value = calculate_fsim(clean_path, rain_path)
psnr_value = calculate_psnr(clean_path, rain_path)
ssim_value = calculate_ssim(clean_path, rain_path)

print(f"FSIM: {fsim_value}")
print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}")


"""
for filename in tqdm(os.listdir(input_dir)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        original = cv2.imread(input_path)
        if original is None:
            print(f"Skipping {filename} (failed to load).")
            continue

        # Apply your pipeline
        filtered_freq = directional_filter_color(original)
        combined_result = refine_with_edge_enhancement(filtered_freq)
        brightened = selective_brighten(combined_result, threshold=100, brighten_factor=1.4)
        enhanced = enhance_details_bilateral(brightened, sigma_color=30, sigma_space=30, detail_boost=1.5)
        horiz_blurred = horizontal_blur(enhanced, kernel_width=5)
        final = selective_brighten(horiz_blurred, threshold=80, brighten_factor=8)

        # Save the result
        cv2.imwrite(output_path, final)
"""
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(rain, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(brightened_result, cv2.COLOR_BGR2RGB))
plt.title("Directional + Edge Enhancement + Bilateral + Brightening Result")
plt.axis('off')

plt.tight_layout()
plt.show()

