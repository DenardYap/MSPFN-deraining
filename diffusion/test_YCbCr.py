from fft_helpers import * 
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from helpers.process import * 
from helpers.get_stats import * 
import numpy as np 
"""
Issues
2. For this particular image 
3. Maybe need to ifftshift 

diff_mag_gt 0.767193347595782
diff_mag_gt 0.7176361798597586
but the predicted image range all the way from -1 to 1
# investigate why we never see normalized of 1 

"""
epoch = 37
diff_npz = f"801_YCrCb_{epoch}.npz"
stats = get_stats_YCrCb("statistics/diff_fft_statistics_log_YCrCb.csv")
data = np.load(diff_npz)["data"]
data = data[0, :, :] # get rid of batch
# nan_indices = np.where(np.isnan(diff_mag))
print(data.shape)

diff_mag = data[0, :, :]
print("Diff mag min before unnormalization:", np.min(diff_mag, axis= (0, 1)))
print("Diff mag max before unnormalization:", np.max(diff_mag, axis= (0, 1)))
# diff_mag = unnormalize_mag(diff_mag, stats)
diff_phase = data[1, :, :]
print("Diff phase min before unnormalization:", np.min(diff_phase, axis= (0, 1)))
print("Diff phase max before unnormalization:", np.max(diff_phase, axis= (0, 1)))
diff_phase *= 2 * np.pi
diff_mag = unnormalize_mag_YCrCb(diff_mag, stats)
print("Diff mag min after unnormalization:", np.min(diff_mag, axis= (0, 1)))
print("Diff mag max after unnormalization:", np.max(diff_mag, axis= (0, 1)))
print("Diff phase max after unnormalization:", np.max(diff_phase, axis= (0, 1)))
print("Diff phase min after unnormalization:", np.min(diff_phase, axis= (0, 1)))

fig, axs = plt.subplots(4, 1, figsize=(8, 16))  # 2 rows and 2 columns

gt = "dataset/images/801.jpg"
rain = "dataset/images_rain/801_1.jpg"

groundtruth_mag_norm, groundtruth_phase, groundtruth_mag =  generate_fft_YCRCB(gt, resize_shape=152)
rain_mag_norm, rain_phase, rain_mag =  generate_fft_YCRCB(rain, resize_shape=152)

diff_mag_unnorm = groundtruth_mag - rain_mag
# diff_mag_unnorm = robust_normalize(diff_mag_unnorm)

diff_mag_gt = signed_log_scale(diff_mag_unnorm)
diff_phase_gt = groundtruth_phase - rain_phase
axs[0].imshow(diff_mag)
axs[0].set_title("Predicted Diff (Mag) of FFT")
axs[0].axis("off")

axs[1].imshow(diff_mag_gt)
axs[1].set_title("Groundtruth Diff (Mag) of FFT")
axs[1].axis("off")

axs[2].imshow(diff_phase)
axs[2].set_title("Predicted Diff (Phase) of FFT")
axs[2].axis("off")

axs[3].imshow(diff_phase_gt)
axs[3].set_title("Groundtruth Diff (Phase) of FFT")
axs[3].axis("off")
output_name = f"prediction_YCrCb_epoch_{epoch}.png"
plt.savefig(output_name)


mag_Y = diff_mag
phase_Y = diff_phase

fig, axes = plt.subplots(1, 2, figsize=(12, 9))  # 3 rows, 2 columns

axes[0].hist(mag_Y.flatten(), bins=100, color='red', alpha=0.7)
axes[0].set_title("Distribution of Mag (Y Channel)")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Frequency")
axes[1].hist(phase_Y.flatten(), bins=100, color='red', alpha=0.7)
axes[1].set_title("Distribution of Phase (Y Channel)")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Frequency")


plt.tight_layout()
plt.savefig(f"predicted_YCrCb_dist_norm_epoch_{epoch}.png")
