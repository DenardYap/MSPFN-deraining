import numpy as np 
from fft_helpers import * 
import matplotlib.pyplot as plt 
from helpers.get_stats import *
from helpers.process import * 

gt = "dataset/images/801.jpg"
rain = "dataset/images_rain/801_1.jpg"

img = cv2.imread(gt)
b, g, r = cv2.split(img)
fft_r = np.fft.fft2(r)
fft_g = np.fft.fft2(g)
fft_b = np.fft.fft2(b)
plt.figure(figsize=(12, 5))
mag_r, phase_r = np.abs(fft_r), np.angle(fft_r)
mag_g, phase_g = np.abs(fft_g), np.angle(fft_g)
mag_b, phase_b = np.abs(fft_b), np.angle(fft_b)

# Comment out these to visualize how the distribution look like without log
mag_r = signed_log_scale(mag_r)
mag_g = signed_log_scale(mag_g)
mag_b = signed_log_scale(mag_b)

fig, axes = plt.subplots(3, 2, figsize=(12, 9))  # 3 rows, 2 columns

axes[0, 0].hist(mag_r.flatten(), bins=100, color='red', alpha=0.7)
axes[0, 0].set_title("Distribution of Mag (Red Channel)")
axes[0, 0].set_xlabel("Value")
axes[0, 0].set_ylabel("Frequency")
axes[0, 1].hist(phase_r.flatten(), bins=100, color='red', alpha=0.7)
axes[0, 1].set_title("Distribution of Phase (Red Channel)")
axes[0, 1].set_xlabel("Value")
axes[0, 1].set_ylabel("Frequency")

axes[1, 0].hist(mag_g.flatten(), bins=100, color='green', alpha=0.7)
axes[1, 0].set_title("Distribution of Mag (Green Channel)")
axes[1, 0].set_xlabel("Value")
axes[1, 0].set_ylabel("Frequency")
axes[1, 1].hist(phase_g.flatten(), bins=100, color='green', alpha=0.7)
axes[1, 1].set_title("Distribution of Phase (Green Channel)")
axes[1, 1].set_xlabel("Value")
axes[1, 1].set_ylabel("Frequency")

axes[2, 0].hist(mag_b.flatten(), bins=100, color='blue', alpha=0.7)
axes[2, 0].set_title("Distribution of Mag (Blue Channel)")
axes[2, 0].set_xlabel("Value")
axes[2, 0].set_ylabel("Frequency")
axes[2, 1].hist(phase_b.flatten(), bins=100, color='blue', alpha=0.7)
axes[2, 1].set_title("Distribution of Phase (Blue Channel)")
axes[2, 1].set_xlabel("Value")
axes[2, 1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("dist.png")