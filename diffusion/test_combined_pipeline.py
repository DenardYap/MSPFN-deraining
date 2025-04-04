# test_combined_pipeline.py

import cv2
import matplotlib.pyplot as plt
from directional_filter import DirectionalFilter
from fft_and_filter import generate_fft
from fft_helpers import reconstruct_from_fft_return

# === File Paths ===
gt_path = "/Users/gennadumont/Downloads/MSPFN-deraining/model/test/test_data/R100H/cleancrop/1.png"
rain_path = "/Users/gennadumont/Downloads/MSPFN-deraining/model/test/test_data/R100H/inputcrop/1.png"

# === Step 1: Apply Directional Filter ===
df = DirectionalFilter()
rain_img = cv2.imread(rain_path)
filtered_img = df.apply_filter(rain_img)

# === Step 2: Run FFT Analysis ===
# Get magnitude and phase from both GT and filtered rain image
mag_gt, phase_gt, raw_mag_gt = generate_fft(gt_path)
mag_filtered, phase_filtered, raw_mag_filtered = generate_fft(filtered_img)

# === Step 3: Compare FFT Differences ===
diff_mag = raw_mag_gt - raw_mag_filtered
diff_phase = phase_gt - phase_filtered

# === Step 4: Reconstruct from Filtered FFT ===
reconstructed = reconstruct_from_fft_return(
    raw_mag_filtered[:, :, 2], raw_mag_filtered[:, :, 1], raw_mag_filtered[:, :, 0],
    phase_filtered[:, :, 2], phase_filtered[:, :, 1], phase_filtered[:, :, 0],
)

# === Step 5: Display ===
plt.figure(figsize=(15, 6))

# Original Rain Image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(rain_img, cv2.COLOR_BGR2RGB))
plt.title("Rain Image")
plt.axis("off")

# After Directional Filter
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
plt.title("Directional Filter Output")
plt.axis("off")

# Reconstructed from Filtered FFT
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
plt.title("Reconstructed from FFT")
plt.axis("off")

plt.tight_layout()
plt.show()
