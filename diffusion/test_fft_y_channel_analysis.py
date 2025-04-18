# test_fft_y_channel_analysis.py
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from fft_helpers import generate_fft_YCRCB, normalize_diff

# === File Paths ===
gt_path = "/Users/gennadumont/Downloads/MSPFN-deraining/model/test/test_data/R100H/cleancrop/1.png"
rain_path = "/Users/gennadumont/Downloads/MSPFN-deraining/model/test/test_data/R100H/inputcrop/1.png"

# === Load FFTs for YCrCb ===
mag_gt, phase_gt, raw_mag_gt = generate_fft_YCRCB(gt_path)
mag_rain, phase_rain, raw_mag_rain = generate_fft_YCRCB(rain_path)

# === Compute Diffs ===
diff_mag = raw_mag_gt - raw_mag_rain
diff_phase = phase_gt - phase_rain

# === Normalize for Display ===
norm_diff_mag = normalize_diff(diff_mag)
norm_diff_phase = normalize_diff(diff_phase)

# === Reconstruct using Y-channel diff only ===
def reconstruct_Y_only(groundtruth_path, rain_path):
    gt_img = cv2.imread(groundtruth_path)
    rain_img = cv2.imread(rain_path)

    gt_ycrcb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCrCb)
    rain_ycrcb = cv2.cvtColor(rain_img, cv2.COLOR_BGR2YCrCb)

    gt_Y, _, _ = cv2.split(gt_ycrcb)
    rain_Y, rain_Cr, rain_Cb = cv2.split(rain_ycrcb)

    # FFT on both Y channels
    fft_gt_Y = np.fft.fftshift(np.fft.fft2(gt_Y))
    fft_rain_Y = np.fft.fftshift(np.fft.fft2(rain_Y))

    # Separate mag and phase
    mag_gt = np.abs(fft_gt_Y)
    phase_rain = np.angle(fft_rain_Y)
    mag_rain = np.abs(fft_rain_Y)

    # Calculate mag difference
    mag_diff = mag_gt - mag_rain
    recon_mag = mag_rain + mag_diff

    # Combine new mag with original rain phase
    combined_fft = recon_mag * np.exp(1j * phase_rain)
    combined_fft = np.fft.ifft2(np.fft.ifftshift(combined_fft))
    new_Y = np.abs(combined_fft).astype(np.uint8)

    # Reconstruct image with modified Y and original Cr/Cb
    new_ycrcb = cv2.merge([new_Y, rain_Cr, rain_Cb])
    new_bgr = cv2.cvtColor(new_ycrcb, cv2.COLOR_YCrCb2BGR)

    # Show result
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rain_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Rain Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(new_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Reconstructed w/ Y Diff Only")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    cv2.imwrite("reconstructed_Y_only.png", new_bgr)
    print("Saved: reconstructed_Y_only.png")


# === Run it ===
reconstruct_Y_only(gt_path, rain_path)
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

def suppress_high_freq_fft_Y(rain_path, suppression_strength=2):
    rain_img = cv2.imread(rain_path)
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
    new_bgr = cv2.cvtColor(new_ycrcb, cv2.COLOR_YCrCb2BGR)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rain_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Rainy Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(new_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Rain Reduced (FFT Suppression)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    cv2.imwrite("rain_reduced_fft_only.png", new_bgr)
    print("Saved: rain_reduced_fft_only.png")

suppress_high_freq_fft_Y("/Users/gennadumont/Downloads/MSPFN-deraining/model/test/test_data/R100H/inputcrop/1.png")
