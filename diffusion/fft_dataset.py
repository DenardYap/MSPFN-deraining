import csv
import json
from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np
import os 
from fft_helpers import * 
import cv2

class FFTDataset(Dataset):
    def __init__(self, csv_file, reshape_size=128, statistics=None, transforms=None, permute=True):
        """
        Data loader for FFT images

        Args:
            csv_file (str): Path to the csv file containing paths and to the groundtruth and rain images
            statistics (dict): max and min of the rain images and diff images 
            transforms : Pytorch transformations
        """
        self.reshape_size = reshape_size
        self.data = [] # filepath,id,npz_filepath
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                image_path, image_rain_path = row
                self.data.append((image_path, image_rain_path))  # Store as a tuple

        self.statistics = statistics
        self.transforms = transforms
        self.permute = permute


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, image_rain_path = self.data[idx]

        groundtruth_mag, groundtruth_phase, rain_mag, rain_phase = \
                generate_fft_for_preprocessing(image_path, image_rain_path, self.reshape_size)
        
        groundtruth_mag_and_phase = concat_mag_and_phase(groundtruth_mag, groundtruth_phase)
        rain_mag_and_phase = concat_mag_and_phase(rain_mag, rain_phase)
        assert groundtruth_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {groundtruth_mag_and_phase.shape[-1]}"
        assert rain_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {rain_mag_and_phase.shape[-1]}"
        diff_mag_and_phase = groundtruth_mag_and_phase - rain_mag_and_phase
        if self.transforms:
            diff_mag_and_phase = self.transforms(diff_mag_and_phase)
            rain_mag_and_phase = self.transforms(rain_mag_and_phase)

        if self.permute:
            # The UNet expect data to be in (C, W, H), but original data is (W, H, C)
            diff_mag_and_phase = np.transpose(diff_mag_and_phase, (2, 0, 1))
            rain_mag_and_phase = np.transpose(rain_mag_and_phase, (2, 0, 1))
        diff_mag_and_phase = np.float32(diff_mag_and_phase)
        rain_mag_and_phase = np.float32(rain_mag_and_phase)
        return diff_mag_and_phase, rain_mag_and_phase

# label_csv = "ids_to_pcs_with_labels.csv"

# dataset = PointCloudDataset(label_csv)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# # Iterate through the dataloader
# for _, pc, label in dataloader:
#     print("PC Shape:", pc.shape)
#     print("Label:", label)
