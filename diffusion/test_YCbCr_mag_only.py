from fft_helpers import * 
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from helpers.process import * 
from helpers.get_stats import * 
import numpy as np 
epoch = 68
diff_npz = f"801_UNet2_{epoch}.npz"
stats = get_stats_YCrCb("statistics/diff_fft_statistics_log_YCrCb.csv")
data = np.load(diff_npz)["data"]
data = data[0, :, :] # get rid of batch
# nan_indices = np.where(np.isnan(diff_mag))
print(data)

diff_mag = data[0, :, :]
diff_mag = unnormalize_mag_YCrCb(diff_mag, stats)

fig, axs = plt.subplots(2, 1, figsize=(8, 16))  # 2 rows and 2 columns

gt = "dataset/images/801.jpg"
rain = "dataset/images_rain/801_1.jpg"

groundtruth_mag_norm, groundtruth_phase, groundtruth_mag =  generate_fft_YCRCB(gt, resize_shape=128)
rain_mag_norm, rain_phase, rain_mag =  generate_fft_YCRCB(rain, resize_shape=128)

diff_mag_unnorm = groundtruth_mag - rain_mag

diff_mag_gt = signed_log_scale(diff_mag_unnorm)
diff_phase_gt = groundtruth_phase - rain_phase
axs[0].imshow(diff_mag, cmap="gray")
axs[0].set_title("Predicted Diff (Mag) of FFT")
axs[0].axis("off")

axs[1].imshow(diff_mag_gt[:, :, 0], cmap="gray")
axs[1].set_title("Groundtruth Diff (Mag) of FFT")
axs[1].axis("off")

output_name = f"prediction_UNet2_{epoch}.png"
plt.savefig(output_name)
