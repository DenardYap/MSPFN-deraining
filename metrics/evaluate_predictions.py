"""
Take in two folders 
The ground truth folder and the prediction folder

Then, go through each image in the prediction folder and find its 
groundtruth counterpart in the groundtruth folder, then, evaluate 
their FSIM, SSIM, and PSNR

Finally, average it and print out the three results
"""

import os 
import sys 
import time
from metric import * 

# dataset_names = ["bilateral_filter", "bilateral_filter_implementation2", "direction_filtered_images", "edge_enhanced", "gaussian_filter", "median_filter", "median_filter_implementation2", "nlm_implementation2"]
dataset_names = ["R100H"]
for dataset_name in dataset_names:
    predict_folder = f"/Users/bernardyap/Desktop/UofM/WN25/EECS 556/Deraining/model/test/test_data/{dataset_name}_predict"
    groundtruth_folder = f"/Users/bernardyap/Desktop/UofM/WN25/EECS 556/Deraining/model/test/test_data/R100H/cleancrop"

    files = os.listdir(predict_folder)
    count = len(files)

    SSIM = 0 
    FSIM = 0 
    PSNR = 0 

    start_time = time.time()
    for image_name in files:
        # print(image_name)
        predict_path = os.path.join(predict_folder, image_name)
        groundtruth_path = os.path.join(groundtruth_folder, image_name)
        SSIM += calculate_ssim(predict_path, groundtruth_path, 128)
        FSIM += calculate_fsim(predict_path, groundtruth_path, 128)
        PSNR += calculate_psnr(predict_path, groundtruth_path, 128)

    end_time = time.time()
    print(f"Took {end_time - start_time}s for dataset {dataset_name}")
    print(f"Total images {count}")
    print(f"Average SSIM: {SSIM/count}")
    print(f"Average FSIM: {FSIM/count}")
    print(f"Average PSNR: {PSNR/count}")