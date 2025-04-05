import csv
import json
from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np
import os 
from fft_helpers import * 
import cv2
from helpers.process import * 

class FFTDataset_YCrCb(Dataset):
    def __init__(self, csv_file, 
                 diff_stats_csv_file="diff_fft_statistics_log_YCrCb.csv", 
                 rain_stats_csv_file="rain_fft_statistics_log_YCrCb.csv", 
                 resize_shape=32, 
                 permute=True,
                 mag_only=False):
        """
        Data loader for FFT images

        Args:
            csv_file (str): Path to the csv file containing paths and to the groundtruth and rain images
            diff_stats_csv_file (str): Path to the statistics of the difference in FFT (e.g max, min, mean, std, etc)
            rain_stats_csv_file (str): Path to the statistics of the rain FFT (e.g max, min, mean, std, etc)
            resize_shape (int) : The size to resize the image into -> e.g if 128 all images will be resized to 128x128
            permute (bool) : Whether to swap (W, H, C) to (C, W, H) 
            mag_only (bool) : Whether to only return magnitude in training
        """
        self.mag_only = mag_only
        self.resize_shape = resize_shape
        self.data = []
        # first read in the image path and its rain counterpart
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                image_path, image_rain_path = row
                self.data.append((image_path, image_rain_path))  # Store as a tuple
        
        # Then we grab the statistics for normalization purposes
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
            self.diff_stats = {key: float(value) for key, value in zip(column_names, row)}

        with open(rain_stats_csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            row = next(reader)  # Read the only data row
            self.rain_stats = {key: float(value) for key, value in zip(column_names, row)}

        self.permute = permute


    def _normalize(self, data, stats):
        """Normalize data using min-max scaling."""
        normalized_data = np.zeros((self.resize_shape, self.resize_shape, 3), dtype=np.float32)
        
        for i, key in enumerate(["mag_Y", "mag_Cr", "mag_Cb"]):
            min_val = float(stats[f"{key}_min"])  # Extract the min value
            max_val = float(stats[f"{key}_max"])  # Extract the max value
            
            normalized_data[..., i] = 2 * ((data[..., i] - min_val) / (max_val - min_val + 1e-8)) - 1

        # sanity check 
        # print("min", np.min(normalized_data, axis=(0, 1))) # should be -1
        # print("max", np.max(normalized_data, axis=(0, 1))) # should be 1
        return normalized_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        The problem with FFT data is that they have very High-Dynamic Range (HDR)
        So, we need to first log1p the data. 
        In machine learning training it's usually best for data to be in the range 
        [-1, 1], so we need to further normalized these log data to [-1, 1]

        The other problem with difference is that they might include negative values,
        which mean the log is not defined, so we need to signed_log_scale as defined 
        in helpers/process.py.

        We need to do the difference between the groundtruth and the rain before 
        we log, because that's the "True difference". Doing difference after normalization
        changed the meaning of the difference and the mode might not be able to learn the 
        actual distribution.
        """

        image_path, image_rain_path = self.data[idx]

        # First get the unnormalized magnitude and phase for the groundtruth and rain images
        # _, gt_phase, gt_mag_unnorm = generate_fft_YCRCB_training(image_path, resize_shape=self.resize_shape)
        # _, rain_phase, rain_mag_unnorm = generate_fft_YCRCB_training(image_rain_path, resize_shape=self.resize_shape)
        _, gt_phase, gt_mag_unnorm = generate_fft_YCRCB(image_path, resize_shape=self.resize_shape)
        _, rain_phase, rain_mag_unnorm = generate_fft_YCRCB(image_rain_path, resize_shape=self.resize_shape)

        # Normalize phase to be in within [-1, 1]
        diff_phase = (gt_phase - rain_phase) / (2 * np.pi)
        rain_phase /= np.pi

        # Step 1. Get the difference in the original scale 
        diff_mag_unnorm = gt_mag_unnorm - rain_mag_unnorm
        # Step 2. Perform a log normalization 
        diff_mag_log = signed_log_scale(diff_mag_unnorm)
        # Step 3: Normalize to [-1, 1]
        diff_mag_norm = self._normalize(diff_mag_log, self.diff_stats)

        # Do the same for rain 
        rain_mag_log = signed_log_scale(rain_mag_unnorm)
        rain_mag_norm = self._normalize(rain_mag_log, self.rain_stats)        

        # Concat them to form a WxHx6 matrix
        if self.mag_only:
            diff_mag_and_phase = diff_mag_norm
            rain_mag_and_phase = rain_mag_norm
            # Only need the Y channel

            diff_mag_and_phase = diff_mag_and_phase[:, :, [0]]  # make it into shape (W, H, 1)
            rain_mag_and_phase = rain_mag_and_phase[:, :, [0]]  # make it into shape (W, H, 1)
        else:
            diff_mag_and_phase = np.concatenate([diff_mag_norm, diff_phase], axis=-1)  
            rain_mag_and_phase = np.concatenate([rain_mag_norm, rain_phase], axis=-1)  
            assert diff_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {diff_mag_and_phase.shape[-1]}"
            assert rain_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {rain_mag_and_phase.shape[-1]}"

            # Only need the Y channel
            diff_mag_and_phase = diff_mag_and_phase[:, :, [0, 3]]  # make it into shape (W, H, 2)
            rain_mag_and_phase = rain_mag_and_phase[:, :, [0, 3]]  # make it into shape (W, H, 2)

        if self.permute:
            # The UNet expect data to be in (2, W, H), but original data is (W, H, 2)
            diff_mag_and_phase = np.transpose(diff_mag_and_phase, (2, 0, 1))
            rain_mag_and_phase = np.transpose(rain_mag_and_phase, (2, 0, 1))

        diff_mag_and_phase = np.float32(diff_mag_and_phase)
        rain_mag_and_phase = np.float32(rain_mag_and_phase) 
        # print(diff_mag_and_phase.shape)
        # print(rain_mag_and_phase.shape)
        # print(np.max(diff_mag_and_phase, axis=(1, 2)))
        # print(np.min(diff_mag_and_phase, axis=(1, 2)))
        # print(np.max(rain_mag_and_phase, axis=(1, 2)))
        # print(np.min(rain_mag_and_phase, axis=(1, 2)))
        
        return diff_mag_and_phase, rain_mag_and_phase

    # def __getitem__(self, idx):
    #     image_path, image_rain_path = self.data[idx]

    #     # First get the unnormalized magnitude and phase for the groundtruth and rain images
    #     groundtruth_mag, groundtruth_phase, rain_mag, rain_phase = \
    #             generate_fft_for_preprocessing(image_path, image_rain_path, self.reshape_size)
        
    #     # Concat them to form a WxHx6 matrix
    #     groundtruth_mag_and_phase = concat_mag_and_phase(groundtruth_mag, groundtruth_phase)
    #     rain_mag_and_phase = concat_mag_and_phase(rain_mag, rain_phase)
    #     assert groundtruth_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {groundtruth_mag_and_phase.shape[-1]}"
    #     assert rain_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {rain_mag_and_phase.shape[-1]}"
    #     diff_mag_and_phase = groundtruth_mag_and_phase - rain_mag_and_phase

    #     diff_mag_and_phase = self._normalize(diff_mag_and_phase, self.diff_stats)
    #     rain_mag_and_phase = self._normalize(rain_mag_and_phase, self.rain_stats)

    #     if self.permute:
    #         # The UNet expect data to be in (C, W, H), but original data is (W, H, C)
    #         diff_mag_and_phase = np.transpose(diff_mag_and_phase, (2, 0, 1))
    #         rain_mag_and_phase = np.transpose(rain_mag_and_phase, (2, 0, 1))
    #     diff_mag_and_phase = np.float32(diff_mag_and_phase)
    #     rain_mag_and_phase = np.float32(rain_mag_and_phase)
    #     return diff_mag_and_phase, rain_mag_and_phase

# label_csv = "ids_to_pcs_with_labels.csv"

# dataset = PointCloudDataset(label_csv)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# # Iterate through the dataloader
# for _, pc, label in dataloader:
#     print("PC Shape:", pc.shape)
#     print("Label:", label)
