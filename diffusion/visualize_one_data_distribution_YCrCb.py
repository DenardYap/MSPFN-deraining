import numpy as np 
from fft_helpers import * 
import matplotlib.pyplot as plt 
from helpers.get_stats import *
from helpers.process import * 

gt = "dataset/images/801.jpg"
rain = "dataset/images_rain/801_1.jpg"

img = cv2.imread(rain)

Y, Cr, Cb = cv2.split(img)
fft_Y = np.fft.fft2(Y)
fft_Cr = np.fft.fft2(Cr)
fft_Cb = np.fft.fft2(Cb)
fft_Y_shifted = np.fft.fftshift(fft_Y)
fft_Cr_shifted = np.fft.fftshift(fft_Cr)
fft_Cb_shifted = np.fft.fftshift(fft_Cb)

mag_Y, phase_Y = np.abs(fft_Y_shifted), np.angle(fft_Y_shifted)
mag_Cr, phase_Cr = np.abs(fft_Cr_shifted), np.angle(fft_Cr_shifted)
mag_Cb, phase_Cb = np.abs(fft_Cb_shifted), np.angle(fft_Cb_shifted)

# Comment out these to visualize how the distribution look like without log
mag_Y = signed_log_scale(mag_Y)
mag_Cr = signed_log_scale(mag_Cr)
mag_Cb = signed_log_scale(mag_Cb)

fig, axes = plt.subplots(3, 2, figsize=(12, 9))  # 3 rows, 2 columns

axes[0, 0].hist(mag_Y.flatten(), bins=100, color='red', alpha=0.7)
axes[0, 0].set_title("Distribution of Mag (Y Channel)")
axes[0, 0].set_xlabel("Value")
axes[0, 0].set_ylabel("Frequency")
axes[0, 1].hist(phase_Y.flatten(), bins=100, color='red', alpha=0.7)
axes[0, 1].set_title("Distribution of Phase (Y Channel)")
axes[0, 1].set_xlabel("Value")
axes[0, 1].set_ylabel("Frequency")

axes[1, 0].hist(mag_Cr.flatten(), bins=100, color='green', alpha=0.7)
axes[1, 0].set_title("Distribution of Mag (Cr Channel)")
axes[1, 0].set_xlabel("Value")
axes[1, 0].set_ylabel("Frequency")
axes[1, 1].hist(phase_Cr.flatten(), bins=100, color='green', alpha=0.7)
axes[1, 1].set_title("Distribution of Phase (Cr Channel)")
axes[1, 1].set_xlabel("Value")
axes[1, 1].set_ylabel("Frequency")

axes[2, 0].hist(mag_Cb.flatten(), bins=100, color='blue', alpha=0.7)
axes[2, 0].set_title("Distribution of Mag (Cb Channel)")
axes[2, 0].set_xlabel("Value")
axes[2, 0].set_ylabel("Frequency")
axes[2, 1].hist(mag_Cb.flatten(), bins=100, color='blue', alpha=0.7)
axes[2, 1].set_title("Distribution of Phase (Cb Channel)")
axes[2, 1].set_xlabel("Value")
axes[2, 1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("dist_YCrCb.png")