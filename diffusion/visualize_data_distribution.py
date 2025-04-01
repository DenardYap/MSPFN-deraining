import numpy as np 
from fft_helpers import * 
import matplotlib.pyplot as plt 
from helpers.get_stats import *
from helpers.process import * 
import os 

rain_mag_data = [] 
rain_phase_data = [] 
diff_mag_data = [] 
diff_phase_data = [] 

gt = "dataset/images/801.jpg"
rain = "dataset/images_rain/801_1.jpg"

diff_stats = get_stats("statistics/diff_fft_statistics_log.csv")
rain_stats = get_stats("statistics/rain_fft_statistics_log.csv")

groundtruth_folder = "dataset/images"
rain_folder = "dataset/images_rain"
rain_image_paths = os.listdir(rain_folder)

# for rains
rain_mins = np.array([float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')])  
rain_maxs = np.array([float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')])  
rain_sum_6_channels = np.zeros(6)
rain_sum_sq_6_channels = np.zeros(6)  

# for diffs
diff_mins = np.array([float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')])  
diff_maxs = np.array([float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')])  
diff_sum_6_channels = np.zeros(6)
diff_sum_sq_6_channels = np.zeros(6)  

pixel_count = 0 
for i, rain_image_path in enumerate(rain_image_paths):
    print(i)
    rain_id, ext = rain_image_path.split('.')
    groundtruth_id, _ = rain_id.split('_')
    rain_filepath = os.path.join(rain_folder, rain_image_path)
    groundtruth_filepath = os.path.join(groundtruth_folder, groundtruth_id + "." + ext)

    # First get the unnormalized magnitude and phase for the groundtruth and rain images
    _, gt_phase, gt_mag_unnorm = generate_fft(groundtruth_filepath, 152)
    _, rain_phase, rain_mag_unnorm = generate_fft(rain_filepath, 152)

    diff_phase = gt_phase - rain_phase
    # Step 1. Get the difference in the original scale 
    diff_mag_unnorm = gt_mag_unnorm - rain_mag_unnorm
    # Step 2. Scale the log values down
    diff_mag_log = signed_log_scale(diff_mag_unnorm) 
    rain_mag_log = signed_log_scale(rain_mag_unnorm)
    diff_mag_log = normalize_mag(diff_mag_log, diff_stats, True)
    rain_mag_log = normalize_mag(rain_mag_log, rain_stats, True)
    rain_mag_data.append(rain_mag_log)
    diff_mag_data.append(diff_mag_log)
    rain_phase_data.append(rain_phase / np.pi)
    diff_phase_data.append(diff_phase / (2 * np.pi))

rain_mag_data = np.array(rain_mag_data)
rain_phase_data = np.array(rain_phase_data)
diff_mag_data = np.array(diff_mag_data)
diff_phase_data = np.array(diff_phase_data)

rain_mag_r = rain_mag_data[:, :, :, 0]
rain_mag_g = rain_mag_data[:, :, :, 1]
rain_mag_b = rain_mag_data[:, :, :, 2]
rain_phase_r = rain_phase_data[:, :, :, 0]
rain_phase_g = rain_phase_data[:, :, :, 1]
rain_phase_b = rain_phase_data[:, :, :, 2]

diff_mag_r = diff_mag_data[:, :, :, 0]
diff_mag_g = diff_mag_data[:, :, :, 1]
diff_mag_b = diff_mag_data[:, :, :, 2]
diff_phase_r = diff_phase_data[:, :, :, 0]
diff_phase_g = diff_phase_data[:, :, :, 1]
diff_phase_b = diff_phase_data[:, :, :, 2]
fig, axes = plt.subplots(2, 6, figsize=(20, 9))  

axes[0, 0].hist(rain_mag_r.flatten(), bins=200, color='red', alpha=0.7)
axes[0, 0].set_title("rain_mag_r")
axes[0, 0].set_xlabel("Value")
axes[0, 0].set_ylabel("Frequency")
axes[0, 1].hist(rain_mag_g.flatten(), bins=200, color='green', alpha=0.7)
axes[0, 1].set_title("rain_mag_g")
axes[0, 1].set_xlabel("Value")
axes[0, 1].set_ylabel("Frequency")
axes[0, 2].hist(rain_mag_b.flatten(), bins=200, color='blue', alpha=0.7)
axes[0, 2].set_title("rain_mag_b")
axes[0, 2].set_xlabel("Value")
axes[0, 2].set_ylabel("Frequency")
axes[0, 3].hist(rain_phase_r.flatten(), bins=200, color='red', alpha=0.7)
axes[0, 3].set_title("rain_phase_r")
axes[0, 3].set_xlabel("Value")
axes[0, 3].set_ylabel("Frequency")
axes[0, 4].hist(rain_phase_g.flatten(), bins=200, color='green', alpha=0.7)
axes[0, 4].set_title("rain_phase_g")
axes[0, 4].set_xlabel("Value")
axes[0, 4].set_ylabel("Frequency")
axes[0, 5].hist(rain_phase_b.flatten(), bins=200, color='blue', alpha=0.7)
axes[0, 5].set_title("rain_phase_b")
axes[0, 5].set_xlabel("Value")
axes[0, 5].set_ylabel("Frequency")

axes[1, 0].hist(diff_mag_r.flatten(), bins=200, color='red', alpha=0.7)
axes[1, 0].set_title("diff_mag_r")
axes[1, 0].set_xlabel("Value")
axes[1, 0].set_ylabel("Frequency")
axes[1, 1].hist(diff_mag_g.flatten(), bins=200, color='green', alpha=0.7)
axes[1, 1].set_title("diff_mag_g")
axes[1, 1].set_xlabel("Value")
axes[1, 1].set_ylabel("Frequency")
axes[1, 2].hist(diff_mag_b.flatten(), bins=200, color='blue', alpha=0.7)
axes[1, 2].set_title("diff_mag_b")
axes[1, 2].set_xlabel("Value")
axes[1, 2].set_ylabel("Frequency")
axes[1, 3].hist(diff_phase_r.flatten(), bins=200, color='red', alpha=0.7)
axes[1, 3].set_title("diff_phase_r")
axes[1, 3].set_xlabel("Value")
axes[1, 3].set_ylabel("Frequency")
axes[1, 4].hist(diff_phase_g.flatten(), bins=200, color='green', alpha=0.7)
axes[1, 4].set_title("diff_phase_g")
axes[1, 4].set_xlabel("Value")
axes[1, 4].set_ylabel("Frequency")
axes[1, 5].hist(diff_phase_b.flatten(), bins=200, color='blue', alpha=0.7)
axes[1, 5].set_title("diff_phase_b")
axes[1, 5].set_xlabel("Value")
axes[1, 5].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("all_data_dist_norm.png")