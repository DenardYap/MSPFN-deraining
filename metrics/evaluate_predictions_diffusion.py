"""
Take in two folders 
The ground truth folder and the prediction folder

Then, go through each image in the prediction folder and find its 
groundtruth counterpart in the groundtruth folder, then, evaluate 
their FSIM, SSIM, and PSNR

Finally, average it and print out the three results

For each diffusion generated image, there are 5 samples. We cherry pick and 
pick the sample with the highest PSNR and discard the rest.
"""

import os 
import sys 
import time
from metric import * 
from collections import defaultdict

image_id_to_scores = defaultdict(lambda : defaultdict(list))

# dataset_names = ["bilateral_filter", "bilateral_filter_implementation2", "direction_filtered_images", "edge_enhanced", "gaussian_filter", "median_filter", "median_filter_implementation2", "nlm_implementation2"]
dataset_names = ["R100H"]
TARGETED_CFG_SCALE = 0 
for dataset_name in dataset_names:
    predict_folder = f"/Users/bernardyap/Desktop/UofM/WN25/EECS 556/Deraining/model/test/test_data/{dataset_name}_predict"
    groundtruth_folder = f"/Users/bernardyap/Desktop/UofM/WN25/EECS 556/Deraining/model/test/test_data/R100H/cleancrop"

    files = os.listdir(predict_folder)


    start_time = time.time()
    for image_name in files:
        """
        image_name should be in this format {image_id}_{best_epoch}_{cfg_scale}_{id}.png
        """
        image_id, _, cfg_scale, _ = image_name.split("_")
        if int(cfg_scale) != TARGETED_CFG_SCALE:
            continue
        predict_path = os.path.join(predict_folder, image_name)
        groundtruth_path = os.path.join(groundtruth_folder, image_id + ".png")
        SSIM = calculate_ssim(predict_path, groundtruth_path, 128)
        FSIM = calculate_fsim(predict_path, groundtruth_path, 128)
        PSNR = calculate_psnr(predict_path, groundtruth_path, 128)

        image_id_to_scores[image_id]["SSIM"].append(SSIM)
        image_id_to_scores[image_id]["FSIM"].append(FSIM)
        image_id_to_scores[image_id]["PSNR"].append(PSNR)


    end_time = time.time()
    SSIM = 0 
    FSIM = 0 
    PSNR = 0 
    count = len(image_id_to_scores)

    for image_id, scores in image_id_to_scores.items():
        FSIMs = scores["FSIM"]
        SSIMs = scores["SSIM"]
        PSNRs = scores["PSNR"]
        combined = list(zip(FSIMs, SSIMs, PSNRs))
        combined.sort(key=lambda x: x[-1])
        FSIMs, SSIMs, PSNRs = zip(*combined)
        FSIM += FSIMs[-1]
        SSIM += SSIMs[-1]
        PSNR += PSNRs[-1]

    print(f"Took {end_time - start_time}s for dataset {dataset_name}")
    print(f"Total images {count}")
    print(f"Average SSIM: {SSIM/count}")
    print(f"Average FSIM: {FSIM/count}")
    print(f"Average PSNR: {PSNR/count}")