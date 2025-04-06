"""
Derain an image!
"""

from modules import UNet2
import numpy as np 
from fft_helpers import * 
from helpers.process import * 
import os 
import matplotlib.pyplot as plt 
import torch 
from ddpm_conditional_mag_only import Diffusion
import csv 
import matplotlib.pyplot as plt 
from fft_helpers import * 
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from helpers.process import * 
from helpers.get_stats import * 
import numpy as np 
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_condition(rain_image_path, resize_shape, stats):
    _, phase, mag_unnorm = generate_fft_YCRCB(rain_image_path, resize_shape)
    mag_unnorm =  np.expand_dims(mag_unnorm[:, :, 0], axis=-1)
    phase = np.expand_dims(phase[:, :, 0], axis=-1)
    phase /= np.pi
    mag_log = signed_log_scale(mag_unnorm)
    # mag_log = np.transpose(mag_log, (2, 0, 1))
    mag_norm = normalize_mag_YCrCb(mag_log, stats, True)
    # mag_norm = np.transpose(mag_norm, (1, 2, 0))
    mag_and_phase = concat_mag_and_phase(mag_norm, phase)
    mag_and_phase_transposed = np.transpose(mag_and_phase, (2, 0, 1))
    y = mag_and_phase_transposed[[0],:, :]
    # Should be in [-1, 1], for log data it's focused around ~-0.5 - ~0.5
    print("Max:", np.max(y, axis=(1, 2)))
    print("Min:", np.min(y, axis=(1, 2)))

    return y


diff_stats = get_stats_YCrCb("statistics/diff_fft_statistics_log_YCrCb.csv")
rain_stats = get_stats_YCrCb("statistics/rain_fft_statistics_log_YCrCb.csv")
img_size = 128

def derain_image(image_id, idx, model):

    rain_image_path = f"dataset/images_rain/{image_id}_1.jpg"
    gt_image_path = f"dataset/images/{image_id}.jpg"

    print(f"Deraining {rain_image_path}...")

    y = torch.from_numpy(get_condition(rain_image_path, img_size, rain_stats)).float()
    y = y.unsqueeze(0)
    
    x = diffusion.sample_YCrCb(model, 1, y, cfg_scale=0)
    x = x.cpu().numpy()
    x = x[0, :, :] 
    diff_mag = unnormalize_mag_YCrCb(x, diff_stats)
    diff_mag = np.transpose(diff_mag, (1, 2, 0))

    fig, axs = plt.subplots(2, 4, figsize=(10, 8))  # 2 rows and 2 columns

    groundtruth_mag_norm, groundtruth_phase, groundtruth_mag =  generate_fft_YCRCB(gt_image_path, resize_shape=128)
    rain_mag_norm, rain_phase, rain_mag =  generate_fft_YCRCB(rain_image_path, resize_shape=128)

    diff_mag_unnorm = groundtruth_mag - rain_mag

    diff_mag_gt = signed_log_scale(diff_mag_unnorm)
    diff_phase_gt = groundtruth_phase - rain_phase

    axs[0, 0].imshow(groundtruth_mag_norm[:, :, 0], cmap="gray")
    axs[0, 0].set_title("Groundtruth image's FFT")
    axs[0, 0].axis("off")


    axs[0, 1].imshow(rain_mag_norm[:, :, 0], cmap="gray")
    axs[0, 1].set_title("Rain image's FFT")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(diff_mag_gt[:, :, 0], cmap="gray")
    axs[0, 2].set_title("Groundtruth Diff of FFT")
    axs[0, 2].axis("off")

    axs[0, 3].imshow(diff_mag, cmap="gray")
    axs[0, 3].set_title("Predicted Diff of FFT")
    axs[0, 3].axis("off")

    
    gt_img = cv2.imread(gt_image_path)
    rain_img = cv2.imread(rain_image_path)
    gt_img = cv2.resize(gt_img, (128, 128))
    rain_img = cv2.resize(rain_img, (128, 128))
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)  
    rain_img = cv2.cvtColor(rain_img, cv2.COLOR_BGR2RGB)  

    axs[1, 0].imshow(gt_img)
    axs[1, 0].set_title("Groundtruth image")
    axs[1, 0].axis("off")
    
    rain_mag_Y = rain_mag[:, :, [0]]
    derained_image_Y = reconstruct_prediction(rain_mag_Y, diff_mag)
    derained_image = reconstruct_from_fft_YCrCb_return(derained_image_Y[:, :, 0], rain_mag[:, :, 1], rain_mag[:, :, 2], 
                                rain_phase[:, :, 0], rain_phase[:, :, 1], rain_phase[:, :, 2])

    derained_image = cv2.cvtColor(derained_image, cv2.COLOR_BGR2RGB)  
    axs[1, 1].imshow(derained_image, cmap="gray")
    axs[1, 1].set_title("Derained image")
    axs[1, 1].axis("off")


    axs[1, 2].imshow(rain_img)
    axs[1, 2].set_title("Rain image")
    axs[1, 2].axis("off")

    axs[1, 3].axis('off')

    plt.suptitle(f"Diffusion deraining on Y channel, Magnitude only | Image Id: {image_id}", fontsize=16)

    output_name = f"derained_results/{image_id}_{best_epoch}_{idx}.png"
    plt.savefig(output_name)
    
if __name__ == "__main__":

    model = UNet2().to(device)
    diffusion = Diffusion(img_size=img_size, device=device)
    best_epoch = 170
    weights_path = f"all_weights/weights_mag_only/{best_epoch}.pth"
    ckpt = torch.load(weights_path)
    model.load_state_dict(ckpt)
    print("Loaded model.")

    image_ids = ["801", "802", "803", "804", "805", "806"]
    
    trial = 10 
    for image_id in image_ids:
        for idx in range(trial):
            derain_image(image_id, idx, model)