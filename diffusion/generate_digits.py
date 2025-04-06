"""
Derain an image!
"""

from modules import UNet_MNIST
import numpy as np 
from fft_helpers import * 
from helpers.process import * 
import os 
import matplotlib.pyplot as plt 
import torch 
from ddpm_conditional_MNIST import Diffusion
import csv 
import matplotlib.pyplot as plt

if __name__ == "__main__":

    image_id = "801"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"
    img_size = 28
    model = UNet_MNIST().to(device)
    diffusion = Diffusion(img_size=img_size, device=device)
    weights_folder = "all_weights/weights_MNIST/0.0005"
    best_epoch = 116
    weights_path = f"{weights_folder}/{best_epoch}.pth"
    ckpt = torch.load(weights_path)
    model.load_state_dict(ckpt)
    print("Loaded model.")
    
    # n = 1
    y = torch.tensor([i for i in range(10)]).to(device)
    x = diffusion.sample(model, 10, y, cfg_scale=0)
    x = x.cpu().numpy()

    print(x.shape)
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))  # 10 images in a row

    for idx in range(10):
        img = x[idx][0]  # shape: (28, 28) after removing channel dim
        label = y[idx]
        axes[idx].imshow(img, cmap="gray")
        axes[idx].axis("off")
        axes[idx].set_title(f"{label}", fontsize=10)
    plt.tight_layout()
    output_path = "generated_digits.png"
    plt.savefig(output_path)
    print("done")