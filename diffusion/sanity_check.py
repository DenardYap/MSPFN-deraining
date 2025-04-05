from fft_helpers import * 
import matplotlib.pyplot as plt 
import numpy as np 
import csv 
from helpers.process import * 

diff_stats_csv_file = "statistics/diff_fft_statistics_log.csv"
rain_stats_csv_file = "statistics/rain_fft_statistics_log.csv"
reshape_size = 152
column_names = [
    "mag_R_max", "mag_G_max", "mag_B_max",
    "phase_R_max", "phase_G_max", "phase_B_max",
    "mag_R_min", "mag_G_min", "mag_B_min",
    "phase_R_min", "phase_G_min", "phase_B_min",
    "mag_R_mean", "mag_G_mean", "mag_B_mean",
    "phase_R_mean", "phase_G_mean", "phase_B_mean",
    "mag_R_std", "mag_G_std", "mag_B_std",
    "phase_R_std", "phase_G_std", "phase_B_std"
]

resize_shape = 128

with open(diff_stats_csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    row = next(reader)  # Read the only data row
    diff_stats = {key: float(value) for key, value in zip(column_names, row)}

with open(rain_stats_csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    row = next(reader)  # Read the only data row
    rain_stats = {key: float(value) for key, value in zip(column_names, row)}


def _normalize(data, stats):
    """Normalize data using min-max scaling."""
    normalized_data = np.zeros((resize_shape, resize_shape, 3), dtype=np.float32)
    
    for i, key in enumerate(["mag_R", "mag_G", "mag_B"]):
        min_val = float(stats[f"{key}_min"])  # Extract the min value
        max_val = float(stats[f"{key}_max"])  # Extract the max value
        
        normalized_data[..., i] = 2 * ((data[..., i] - min_val) / (max_val - min_val + 1e-8)) - 1

    # sanity check 
    # print("min", np.min(normalized_data, axis=(0, 1))) # should be -1
    # print("max", np.max(normalized_data, axis=(0, 1))) # should be 1
    return normalized_data
    
def get_data(image_path, image_rain_path):

    _, gt_phase, gt_mag_unnorm = generate_fft(image_path, resize_shape=resize_shape)
    _, rain_phase, rain_mag_unnorm = generate_fft(image_rain_path, resize_shape=resize_shape)

    diff_phase = (gt_phase - rain_phase) / (2 * np.pi)
    rain_phase /= np.pi

    # Step 1. Get the difference in the original scale 
    diff_mag_unnorm = gt_mag_unnorm - rain_mag_unnorm
    # Step 2. Scale the log values down
    diff_mag_log = signed_log_scale(diff_mag_unnorm)
    # Step 3: Normalize to [-1, 1]
    diff_mag_norm = _normalize(diff_mag_log, diff_stats)

    rain_mag_log = signed_log_scale(rain_mag_unnorm)
    rain_mag_norm = _normalize(rain_mag_log, rain_stats)        

    # Concat them to form a WxHx6 matrix
    # The first 3 rows (magnitude) will be in the range [-1, 1]
    # The next 3 rows (phase/angle) will be in the range [-2π, 2π]
    diff_mag_and_phase = concat_mag_and_phase(diff_mag_norm, diff_phase)
    rain_mag_and_phase = concat_mag_and_phase(rain_mag_norm, rain_phase)
    assert diff_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {diff_mag_and_phase.shape[-1]}"
    assert rain_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {rain_mag_and_phase.shape[-1]}"


    diff_mag_and_phase = np.float32(diff_mag_and_phase)
    rain_mag_and_phase = np.float32(rain_mag_and_phase) 
    
    return diff_mag_and_phase, rain_mag_and_phase, \
            concat_mag_and_phase(diff_mag_log, diff_phase * 2 * np.pi),  \
            concat_mag_and_phase(rain_mag_log, rain_phase * np.pi)

gt_file = "dataset/images/801.jpg"
rain_file = "dataset/images_rain/801_1.jpg"
diff, rain, diff_log, rain_log = get_data(gt_file, rain_file)
print(np.max(diff, axis=(0, 1)))
print(np.min(diff, axis=(0, 1)))
print(np.max(rain, axis=(0, 1)))
print(np.min(rain, axis=(0, 1)))
diff_unnorm = unnormalize_mag_and_phase(diff, diff_stats, True)
rain_unnorm = unnormalize_mag_and_phase(rain, rain_stats, True)
diff_mag = diff_log[:, :, :3]
diff_phase = diff_log[:, :, 3:]
rain_mag = rain_log[:, :, :3]
rain_phase = rain_log[:, :, 3:]


diff_mag_unnorm = diff_unnorm[:, :, :3]
diff_phase_unnorm = diff_unnorm[:, :, 3:]
rain_mag_unnorm = rain_unnorm[:, :, :3]
rain_phase_unnorm = rain_unnorm[:, :, 3:]

fig, axs = plt.subplots(2, 4, figsize=(16, 16)) 

axs[0, 0].imshow(diff_mag)
axs[0, 0].set_title("Normalized Diff (Mag) of FFT")
axs[0, 0].axis("off")

axs[0, 1].imshow(diff_phase)
axs[0, 1].set_title("Normalized Diff (Phase) of FFT")
axs[0, 1].axis("off")

axs[0, 2].imshow(rain_mag)
axs[0, 2].set_title("Normalized Rain (Mag) of FFT")
axs[0, 2].axis("off")

axs[0, 3].imshow(rain_phase)
axs[0, 3].set_title("Normalized Rain (Phase) of FFT")
axs[0, 3].axis("off")

axs[1, 0].imshow(diff_mag_unnorm)
axs[1, 0].set_title("Unnormalized Diff (Mag) of FFT")
axs[1, 0].axis("off")

axs[1, 1].imshow(diff_phase_unnorm)
axs[1, 1].set_title("Unnormalized Diff (Phase) of FFT")
axs[1, 1].axis("off")

axs[1, 2].imshow(rain_mag_unnorm)
axs[1, 2].set_title("Unnormalized Rain (Mag) of FFT")
axs[1, 2].axis("off")

axs[1, 3].imshow(rain_phase_unnorm)
axs[1, 3].set_title("Unnormalized Rain (Phase) of FFT")
axs[1, 3].axis("off")

plt.savefig("sanity_check.png")