# import cv2
# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim

# def load_image(path, grayscale=True):
#     img = cv2.imread(path)
#     if grayscale:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img

# # FSIM implementation based on phase congruency and gradient magnitude
# def fsim(img1, img2):
#     from skimage.filters import sobel
#     from skimage import img_as_float
#     from scipy.ndimage import gaussian_filter

#     img1, img2 = img_as_float(img1), img_as_float(img2)

#     # Compute Phase Congruency (Approximation)
#     def phase_congruency(img):
#         return sobel(img)

#     PC1, PC2 = phase_congruency(img1), phase_congruency(img2)
#     GM1, GM2 = sobel(img1), sobel(img2)

#     # Similarity map based on phase congruency and gradient magnitude
#     T1, T2 = 0.85, 160
#     S_PC = (2 * PC1 * PC2 + T1) / (PC1**2 + PC2**2 + T1)
#     S_G = (2 * GM1 * GM2 + T2) / (GM1**2 + GM2**2 + T2)

#     # FSIM calculation
#     FSIM_map = S_PC * S_G
#     return np.mean(FSIM_map)

# # Load images
# generated_img = load_image("giraffe.png")
# ground_truth_img = load_image("giraffe_rain.png")

# # Compute PSNR, SSIM, and FSIM
# psnr_value = psnr(ground_truth_img, generated_img, data_range=255)
# ssim_value = ssim(ground_truth_img, generated_img, data_range=255)
# fsim_value = fsim(ground_truth_img, generated_img)

# # Print results
# print(f"PSNR: {psnr_value:.2f} dB")
# print(f"SSIM: {ssim_value:.4f}")
# print(f"FSIM: {fsim_value:.4f}")
from image_similarity_measures.quality_metrics import fsim
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
import numpy as np

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

if __name__ == "__main__":

    image_path1 = 'giraffe.png'
    image_path2 = 'giraffe_rain.png'

    fsim_value = calculate_fsim(image_path1, image_path2)
    psnr_value = calculate_psnr(image_path1, image_path2)
    ssim_value = calculate_ssim(image_path1, image_path2)

    print(f"FSIM: {fsim_value}")
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")