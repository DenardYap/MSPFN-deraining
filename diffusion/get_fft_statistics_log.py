"""
Generate FFT of all rain and diff images after log transform, then get the 
mean, std, max, and min of each of the following channels

Mag_R Mag_G Mag_B Phase_R Phase_G Phase_B
"""

from fft_helpers import * 
import os 
import sys 
import csv 
from collections import defaultdict
from helpers.process import * 

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
    _, gt_phase, gt_mag_unnorm = generate_fft(groundtruth_filepath)
    _, rain_phase, rain_mag_unnorm = generate_fft(rain_filepath)

    diff_phase = gt_phase - rain_phase
    # Step 1. Get the difference in the original scale 
    diff_mag_unnorm = gt_mag_unnorm - rain_mag_unnorm
    # Step 2. Scale the log values down
    diff_mag_log = signed_log_scale(diff_mag_unnorm)
    rain_mag_log = signed_log_scale(rain_mag_unnorm)
    
    diff_mag_and_phase = concat_mag_and_phase(diff_mag_log, diff_phase)
    rain_mag_and_phase = concat_mag_and_phase(rain_mag_log, rain_phase)

    assert diff_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {diff_mag_and_phase.shape[-1]}"
    assert rain_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {rain_mag_and_phase.shape[-1]}"

    # 6 channels = mag_r, mag_g, mag_b, phase_r, phase_g, phase_b
    rain_sum_6_channels += np.sum(rain_mag_and_phase, axis=(0, 1))
    rain_sum_sq_6_channels += np.sum(rain_mag_and_phase ** 2, axis=(0, 1)) 

    diff_sum_6_channels += np.sum(diff_mag_and_phase, axis=(0, 1))
    diff_sum_sq_6_channels += np.sum(diff_mag_and_phase ** 2, axis=(0, 1)) 

    rain_mins = np.minimum(rain_mins, rain_mag_and_phase.min(axis=(0, 1)))
    rain_maxs = np.maximum(rain_maxs, rain_mag_and_phase.max(axis=(0, 1)))
    diff_mins = np.minimum(diff_mins, diff_mag_and_phase.min(axis=(0, 1)))
    diff_maxs = np.maximum(diff_maxs, diff_mag_and_phase.max(axis=(0, 1)))
    pixel_count += rain_mag_and_phase.shape[0] * rain_mag_and_phase.shape[1]

rain_mean = rain_sum_6_channels / pixel_count  
rain_std = np.sqrt((rain_sum_sq_6_channels / pixel_count) - rain_mean ** 2) 
diff_mean = diff_sum_6_channels / pixel_count  
diff_std = np.sqrt((diff_sum_sq_6_channels / pixel_count) - diff_mean ** 2) 

def construct_stats(maxs, mins, means, stds):
    stats = {}
    stats["mag_R_max"] = float(maxs[0])
    stats["mag_G_max"] = float(maxs[1])
    stats["mag_B_max"] = float(maxs[2])
    stats["phase_R_max"] = float(maxs[3])
    stats["phase_G_max"] = float(maxs[4])
    stats["phase_B_max"] = float(maxs[5])
    stats["mag_R_min"] = float(mins[0])
    stats["mag_G_min"] = float(mins[1])
    stats["mag_B_min"] = float(mins[2])
    stats["phase_R_min"] = float(mins[3])
    stats["phase_G_min"] = float(mins[4])
    stats["phase_B_min"] = float(mins[5])
    stats["mag_R_mean"] = float(means[0])
    stats["mag_G_mean"] = float(means[1])
    stats["mag_B_mean"] = float(means[2])
    stats["phase_R_mean"] = float(means[3])
    stats["phase_G_mean"] = float(means[4])
    stats["phase_B_mean"] = float(means[5])
    stats["mag_R_std"] = float(stds[0])
    stats["mag_G_std"] = float(stds[1])
    stats["mag_B_std"] = float(stds[2])
    stats["phase_R_std"] = float(stds[3])
    stats["phase_G_std"] = float(stds[4])
    stats["phase_B_std"] = float(stds[5])
    keys = []
    vals = [] 

    for key, val in stats.items():
        keys.append(key)
        vals.append(val)

    return stats, [keys, vals]

rain_stats, rain_data = construct_stats(rain_maxs, rain_mins, rain_mean, rain_std)
diff_stats, diff_data = construct_stats(diff_maxs, diff_mins, diff_mean, diff_std)
rain_stats["number_of_images"] = len(rain_image_paths)
rain_stats["number_of_pixels"] = pixel_count
diff_stats["number_of_images"] = len(rain_image_paths)
diff_stats["number_of_pixels"] = pixel_count

print("================== STATS FOR RAIN IMAGES ==================")
print(rain_stats)
print("================== STATS FOR DIFF IMAGES ==================")
print(diff_stats)


output_csv_rain = "statistics/rain_fft_statistics_log.csv"
output_csv_diff = "statistics/diff_fft_statistics_log.csv"

with open(output_csv_rain, mode='w', newline='') as f: 
    csv_writer = csv.writer(f)
    csv_writer.writerows(rain_data)

with open(output_csv_diff, mode='w', newline='') as f: 
    csv_writer = csv.writer(f)
    csv_writer.writerows(diff_data)

print(f"Wrote csv files to {output_csv_diff} and {output_csv_rain}")
