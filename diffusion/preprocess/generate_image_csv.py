"""
Generate a .csv file with these columns
image_filepath, image_rain_filepath
"""

import csv 
import os 

data = [["image_filepath", "image_rain_filepath"]]

groundtruth_folder = "/home/bernerd/eecs556/dataset/images"
rain_folder = "/home/bernerd/eecs556/dataset/images_rain"
rain_image_paths = os.listdir(rain_folder)

for i, rain_image_path in enumerate(rain_image_paths):
    
    rain_id, ext = rain_image_path.split('.')
    groundtruth_id, _ = rain_id.split('_')
    rain_filepath = os.path.join(rain_folder, rain_image_path)
    groundtruth_filepath = os.path.join(groundtruth_folder, groundtruth_id + "." + ext)
    data.append([groundtruth_filepath, rain_filepath])


output_filename = "data.csv"
with open(output_filename, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(data)

print("All done")