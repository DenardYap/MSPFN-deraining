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

dataset_name = "R100H"
predict_folder = f"/Users/bernardyap/Desktop/UofM/WN25/EECS 556/Deraining/model/test/test_data/{dataset_name}_predict"
groundtruth_folder = f"/Users/bernardyap/Desktop/UofM/WN25/EECS 556/Deraining/model/test/test_data/{dataset_name}/cleancrop"

files = os.listdir(predict_folder)
count = len(files)

SSIM = 0 
FSIM = 0 
PSNR = 0 

start_time = time.time()
for image_name in files:
    print(image_name)
    predict_path = os.path.join(predict_folder, image_name)
    groundtruth_path = os.path.join(groundtruth_folder, image_name)
    SSIM += calculate_ssim(predict_path, groundtruth_path)
    FSIM += calculate_fsim(predict_path, groundtruth_path)
    PSNR += calculate_psnr(predict_path, groundtruth_path)

end_time = time.time()
print(f"Took {end_time - start_time}s")
print(f"Total images {count}")
print(f"Average SSIM: {SSIM/count}")
print(f"Average FSIM: {FSIM/count}")
print(f"Average PSNR: {PSNR/count}")