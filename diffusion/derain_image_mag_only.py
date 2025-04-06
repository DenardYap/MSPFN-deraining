"""
Derain an image!
"""

from modules import UNet_Mag_Only, EMA, UNet_conditional_YCrCb
import numpy as np 
from fft_helpers import * 
from helpers.process import * 
import os 
import matplotlib.pyplot as plt 
import torch 
from ddpm_unconditional import Diffusion
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

if __name__ == "__main__":

    image_id = "801"
    rain_image_path = f"dataset/images_rain/{image_id}_1.jpg"
    gt_image_path = f"dataset/images/{image_id}.jpg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"
    img_size = 128
    print(f"Deraining {rain_image_path}, targeted image size: {img_size}")
    model = UNet_Mag_Only(img_size).to(device)
    diffusion = Diffusion(img_size=img_size, device=device)
    weights_folder = "weights_uncond_mag_only"
    best_epoch = 31
    weights_path = f"{weights_folder}/{best_epoch}.pth"
    ckpt = torch.load(weights_path)
    model.load_state_dict(ckpt)
    print("Loaded model.")
    
    # n = 1
    x = diffusion.sample_YCrCb(model, 1)
    x = x.cpu().numpy()
    print(x)
    print("x min:", np.min(x, axis= (0, 1)))
    print("x max:", np.max(x, axis= (0, 1)))
    np.savez(f"{image_id}_YCrCb_mag_only_{best_epoch}.npz", data=x)
    print(x.shape)
