import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from fft_dataset import FFTDataset
from fft_dataset_YCrCb import FFTDataset_YCrCb

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    dataset = FFTDataset(args.dataset_path, args.diff_stats_csv_file, args.rain_stats_csv_file, args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def get_data_YCrCb(args):
    dataset = FFTDataset_YCrCb(args.dataset_path, args.diff_stats_csv_file, args.rain_stats_csv_file, args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def get_data_YCrCb_mag_only(args):
    dataset = FFTDataset_YCrCb(args.dataset_path, args.diff_stats_csv_file, args.rain_stats_csv_file, args.image_size, mag_only=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)