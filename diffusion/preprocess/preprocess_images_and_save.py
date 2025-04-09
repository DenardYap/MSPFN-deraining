import os 
import sys
import csv
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fft_helpers import * 

data = [["groundtruth_id", "rain_id", "diff_npz_filepath", "groundtruth_npz_filepath", "rain_npz_filepath", "groundtruth_filepath", "rain_filepath"]]

npz_save_folder = "/Users/gennadumont/Downloads/MSPFN-deraining/model/gennamodel"

os.makedirs(os.path.join(npz_save_folder, "groundtruth_mag_and_phase"), exist_ok=True)
os.makedirs(os.path.join(npz_save_folder, "rain_mag_and_phase"), exist_ok=True)
os.makedirs(os.path.join(npz_save_folder, "diff_mag_and_phase"), exist_ok=True)
rain_folder = "/Users/gennadumont/Downloads/MSPFN-deraining/model/test/test_data/R100H/inputcrop"
groundtruth_folder = "/Users/gennadumont/Downloads/MSPFN-deraining/model/test/test_data/R100H/cleancrop"

rain_image_paths = os.listdir(rain_folder)

for i, rain_image_path in enumerate(rain_image_paths):
    if not rain_image_path.endswith(".png"):
        continue  # skip non-image files just in case

    rain_id, ext = os.path.splitext(rain_image_path)
    groundtruth_id = rain_id  # filenames are the same

    rain_filepath = os.path.join(rain_folder, rain_image_path)
    groundtruth_filepath = os.path.join(groundtruth_folder, groundtruth_id + ext)

    groundtruth_mag, groundtruth_phase, rain_mag, rain_phase = \
        generate_fft_for_preprocessing(groundtruth_filepath, rain_filepath)

    groundtruth_mag_and_phase = concat_mag_and_phase(groundtruth_mag, groundtruth_phase)
    rain_mag_and_phase = concat_mag_and_phase(rain_mag, rain_phase)
    assert groundtruth_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {groundtruth_mag_and_phase.shape[-1]}"
    assert rain_mag_and_phase.shape[-1] == 6, f"Last dimension is not 6, it is {rain_mag_and_phase.shape[-1]}"
    diff_mag_and_phase = groundtruth_mag_and_phase - rain_mag_and_phase

    groundtruth_npz_filepath = os.path.join(npz_save_folder, f"groundtruth_mag_and_phase/{groundtruth_id}.npz")
    rain_npz_filepath = os.path.join(npz_save_folder, f"rain_mag_and_phase/{rain_id}.npz")
    diff_npz_filepath = os.path.join(npz_save_folder, f"diff_mag_and_phase/{rain_id}.npz")

    np.savez(groundtruth_npz_filepath, mag_and_phase=groundtruth_mag_and_phase)
    np.savez(rain_npz_filepath, mag_and_phase=rain_mag_and_phase)
    np.savez(diff_npz_filepath, mag_and_phase=diff_mag_and_phase)

    data.append([groundtruth_id, rain_id, diff_npz_filepath, groundtruth_npz_filepath,
                 rain_npz_filepath, groundtruth_filepath, rain_filepath])

    print(f"Processed: {rain_id}.png")

output_filename = "data.csv"
with open(output_filename, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(data)

print("All done 🚀")
