"""
Derain an image!
"""

from modules import UNet_conditional, EMA, UNet_conditional_YCrCb
import numpy as np 
from fft_helpers import * 
from helpers.process import * 
import os 
import matplotlib.pyplot as plt 
import torch 
from ddpm_conditional import Diffusion
import csv 

diff_stats_csv_file = "statistics/diff_fft_statistics_log_YCrCb.csv"
rain_stats_csv_file = "statistics/rain_fft_statistics_log_YCrCb.csv"

column_names = [
    "mag_Y_max", "mag_Cr_max", "mag_Cb_max",
    "phase_Y_max", "phase_Cr_max", "phase_Cb_max",
    "mag_Y_min", "mag_Cr_min", "mag_Cb_min",
    "phase_Y_min", "phase_Cr_min", "phase_Cb_min",
    "mag_Y_mean", "mag_Cr_mean", "mag_Cb_mean",
    "phase_Y_mean", "phase_Cr_mean", "phase_Cb_mean",
    "mag_Y_std", "mag_Cr_std", "mag_Cb_std",
    "phase_Y_std", "phase_Cr_std", "phase_Cb_std"
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
    _, phase, mag_unnorm = generate_fft_YCRCB(rain_image_path, resize_shape)
    print(mag_unnorm.shape)
    print(phase.shape)
    mag_unnorm =  np.expand_dims(mag_unnorm[:, :, 0], axis=-1)
    phase = np.expand_dims(phase[:, :, 0], axis=-1)
    print(mag_unnorm.shape)
    print(phase.shape)
    phase /= np.pi
    mag_log = signed_log_scale(mag_unnorm)
    # mag_log = np.transpose(mag_log, (2, 0, 1))
    mag_norm = normalize_mag_YCrCb(mag_log, stats, True)
    # mag_norm = np.transpose(mag_norm, (1, 2, 0))
    mag_and_phase = concat_mag_and_phase(mag_norm, phase)
    print(mag_and_phase.shape)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"
    img_size = 128
    print(f"Deraining {rain_image_path}, targeted image size: {img_size}")
    model = UNet_conditional_YCrCb(img_size).to(device)
    diffusion = Diffusion(img_size=img_size, device=device)

    best_epoch = 37
    weights_path = f"weights_YCrCb_ca/{best_epoch}.pth"
    ckpt = torch.load(weights_path)
    model.load_state_dict(ckpt)
    print("Loaded model.")
    
    # n = 1
    y = torch.from_numpy(get_condition(rain_image_path, img_size, rain_stats)).float()
    y = y.unsqueeze(0)
    # y = np.float32(y)
    # y = y.to(model.device)
    x = diffusion.sample_YCrCb(model, 1, y, cfg_scale=0)
    x = x.cpu().numpy()
    print(x)
    print("x min:", np.min(x, axis= (0, 1)))
    print("x max:", np.max(x, axis= (0, 1)))
    np.savez(f"{image_id}_YCrCb_{best_epoch}.npz", data=x)
    print(x.shape)
