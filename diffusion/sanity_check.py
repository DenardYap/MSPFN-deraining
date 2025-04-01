from fft_helpers import * 
import matplotlib.pyplot as plt 
import numpy as np 
import csv 
from helpers.process import * 

diff_stats_csv_file = "statistics/diff_fft_statistics.csv"
rain_stats_csv_file = "statistics/rain_fft_statistics.csv"
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
    normalized_data = np.zeros((reshape_size, reshape_size, 6), dtype=np.float32)
    
    for i, key in enumerate(["mag_R", "mag_G", "mag_B", "phase_R", "phase_G", "phase_B"]):
        min_val = float(stats[f"{key}_min"])  # Extract the min value
        max_val = float(stats[f"{key}_max"])  # Extract the max value
        
        normalized_data[..., i] = 2 * ((data[..., i] - min_val) / (max_val - min_val + 1e-8)) - 1

    return normalized_data
    
def get_data(image_path, image_rain_path):

    groundtruth_mag, groundtruth_phase, rain_mag, rain_phase = \
            generate_fft_for_preprocessing(image_path, image_rain_path, reshape_size)
    
    groundtruth_mag_and_phase = concat_mag_and_phase(groundtruth_mag, groundtruth_phase)
    rain_mag_and_phase = concat_mag_and_phase(rain_mag, rain_phase)
    assert groundtruth_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {groundtruth_mag_and_phase.shape[-1]}"
    assert rain_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {rain_mag_and_phase.shape[-1]}"
    diff_mag_and_phase = groundtruth_mag_and_phase - rain_mag_and_phase
    # print("diffmax",np.max(rain_mag_and_phase))
    # print("diffmin", np.min(rain_mag_and_phase))
    diff_mag_and_phase = _normalize(diff_mag_and_phase, diff_stats)
    rain_mag_and_phase = _normalize(rain_mag_and_phase, rain_stats)

    # (152, 152, 6) -> (6, 152, 152)
    diff_mag_and_phase = np.transpose(diff_mag_and_phase, (2, 0, 1))
    rain_mag_and_phase = np.transpose(rain_mag_and_phase, (2, 0, 1))
    diff_mag_and_phase = np.float32(diff_mag_and_phase)
    rain_mag_and_phase = np.float32(rain_mag_and_phase)
    return diff_mag_and_phase, rain_mag_and_phase

gt_file = "dataset/images/801.jpg"
rain_file = "dataset/images_rain/801_1.jpg"
diff, rain = get_data(gt_file, rain_file)
# (6, 152, 152) -> (152, 152, 6)
diff = unnormalize(diff, diff_stats)
rain = unnormalize(rain, rain_stats)
diff = np.transpose(diff, (1, 2, 0))
rain = np.transpose(rain, (1, 2, 0))
print(np.max(diff, axis=(0, 1)))
print(np.min(diff, axis=(0, 1)))
diff_mag = diff[:, :, :3]
diff_phase = diff[:, :, 3:]
rain_mag = rain[:, :, :3]
rain_phase = rain[:, :, 3:]

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

print("diff_mag", np.max(diff_mag))
print("diff_mag", np.min(diff_mag))

print("diff_phase", np.max(diff_phase))
print("diff_phase", np.min(diff_phase))

print("rain_mag", np.max(rain_mag))
print("rain_mag", np.min(rain_mag))

print("rain_phase", np.max(rain_phase))
print("rain_phase", np.min(rain_phase))


plt.savefig("sanity_check.png")