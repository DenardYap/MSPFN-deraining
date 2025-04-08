"""
Derain an image!
"""

from modules import UNet_conditional, EMA
import numpy as np 
from fft_helpers import * 
from helpers.process import * 
import os 
import matplotlib.pyplot as plt 
import torch 
from ddpm_conditional import Diffusion
import csv 

diff_stats_csv_file = "statistics/diff_fft_statistics_log.csv"
rain_stats_csv_file = "statistics/rain_fft_statistics_log.csv"

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
    
def get_condition(rain_image_path, resize_shape, stats):
    _, phase, mag_unnorm = generate_fft(rain_image_path, resize_shape)

    mag_log = signed_log_scale(mag_unnorm)
    # mag_log = np.transpose(mag_log, (2, 0, 1))
    # mag_norm = normalize_mag(mag_log, stats)
    # mag_norm = np.transpose(mag_norm, (1, 2, 0))
    mag_and_phase = concat_mag_and_phase(mag_log, phase)
    mag_and_phase_transposed = np.transpose(mag_and_phase, (2, 0, 1))
    y = mag_and_phase_transposed
    # Should be in [-1, 1], for log data it's focused around ~-0.5 - ~0.5
    print("Max:", np.max(y, axis=(1, 2)))
    print("Min:", np.min(y, axis=(1, 2)))

    return y


# def get_condition(rain_image_path, resize_shape, stats):
#     _, phase, mag = generate_fft(rain_image_path, resize_shape)
#     mag_and_phase = concat_mag_and_phase(mag, phase)
#     mag_and_phase_transposed = np.transpose(mag_and_phase, (2, 0, 1))
#     print("mag_and_phase_transposed.shape", mag_and_phase_transposed.shape)
#     y = normalize(mag_and_phase_transposed, stats)
#     print("Max:", np.max(y, axis=(1, 2)))
#     print("Min:", np.min(y, axis=(1, 2)))

#     return y

if __name__ == "__main__":

    image_id = "801"
    rain_image_path = f"dataset/images_rain/{image_id}_1.jpg"
    gt_image_path = f"dataset/images/{image_id}.jpg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    img_size = 152
    print(f"Deraining {rain_image_path}, targeted image size: {img_size}")
    model = UNet_conditional(img_size).to(device)
    diffusion = Diffusion(img_size=img_size, device=device)

    best_epoch = 7
    weights_path = f"weights/{best_epoch}.pth"
    ckpt = torch.load(weights_path)
    model.load_state_dict(ckpt)
    print("Loaded model.")
    
    # n = 1
    y = torch.from_numpy(get_condition(rain_image_path, img_size, rain_stats)).float()
    y = y.unsqueeze(0)
    # y = np.float32(y)
    # y = y.to(model.device)
    x = diffusion.sample(model, 1, y, cfg_scale=0)
    x = x.cpu().numpy()
    print(x)
    np.savez(f"{image_id}_diff_fft_unnorm_epoch{best_epoch}.npz", data=x)
    print(x.shape)
