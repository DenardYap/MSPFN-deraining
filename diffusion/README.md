# Greatlakes

## Enter interactive mode for GPU training
salloc --job-name=interactive --cpus-per-task=4 --nodes=1 --mem=20G --time=08:00:00 --account=eecs556w25_class --partition=spgpu --gres=gpu:1

Once you ran this command, you will have access to the Greatlakes' GPU. Then, in the same terminal, 
run python ddpm_conditional.py to train the model. 

# Training
ddpm_conditional.py is responsible for training the model, `you might want to modify the parameters in launch()` before running the code. To run the code, first request for GPU (see above, the Greatlakes section), then run `python ddpm_conditional.py`. 

# Dataloader
To modify the dataloader, modify the get_data() function in utils.py. For now, we are using FFTDataset to get the data. 
You can refer to the FFTDataset class to change how we access the data (e.g., change it to access real and imaginary parts instead of magnitude and phase).

# Normalization 
Since the FFT components are very large, we need to first log-normalize the data by using the signed_log_scale() function in helpers/process.py. Note: since FFT have negative values we need to do a signed log normalization instead of normal log normalizatio.

Also, since machine learning model trains better in the [-1, 1] range, we also need to normalize the log-normalized data in the [-1, 1] range by using this formula 

x_normalized = 2 * ((x_log - min_val) / (max_val - min_val + 1e-8)) - 1

# Reverse normalization
To reverse the normalization, you can reverse the formula above by doing 

x_log = (x_normalized + 1)/2 * (max_val - min_val + 1e-8) + min_val

Then, to reverse the log, call signed_log_inverse function in helpers/process.py.

# Statistics
To compute the statistics, refer to `get_fft_statistics_log.py` (for data after log transformation) or `get_fft_statistics.py` (for data before log transformation, you probably don't want to do this).

## Alternatively, use sbatch for better synchronization
In the main directory, run `sbatch train.sh`

# Credit:
Most of the diffusion code is copied from:
https://github.com/dome272/Diffusion-Models-pytorch

# Training flow 

See training_pipeline.png for a visualization. 

The diffusion model will learn to generate the 'image' of the difference in FFT for both magnitude and phase .

In the dataloader, we take in the groundtruth and rain images, generate FFT for both, concatenate
their magnitude and phase into an WxHx6 vectors. Then, we also take the difference by subtracting
groundtruth FFT with rain FFT, which also result in an WxHx6 matrix. Lastly, both difference_fft 
and rain_fft are returned. 

The difference_fft is what the diffusion model will learn to generate, the rain_fft will be used
as the condition for the diffusion model. 

During sampling, we feed the FFT of the rain image into the model to sample a diff_fft, then
we can use add the diff_fft to the FFT of the rain image to reconstruct the image. 


## Before training 
Preprocess the images into a .csv file with these columns
image_filepath, image_rain_filepath

Then, this .csv file will later be used by the DataLoader to load in FFT 'images'. 

# Stats 

================== STATS FOR RAIN IMAGES ==================
{'mag_R_max': 135855217.0, 
 'mag_G_max': 156903143.0, 
 'mag_B_max': 148352405.0, 
 'phase_R_max': 3.141592653589793, 
 'phase_G_max': 3.141592653589793, 
 'phase_B_max': 3.141592653589793, 
 'mag_R_min': 0.26717484204398306, 
 'mag_G_min': 0.3930467039383175, 
 'mag_B_min': 0.1844549793101568, 
 'phase_R_min': -3.141592653589793, 
 'phase_G_min': -3.141592653589793, 
 'phase_B_min': -3.141592653589793, 
 'mag_R_mean': 8814.03041683331, 
 'mag_G_mean': 8864.51467626992, 
 'mag_B_mean': 8940.354325505254, 
 'phase_R_mean': 2.471104472274917e-06, 
 'phase_G_mean': 4.4572740376683465e-06, 
 'phase_B_mean': 2.692936397761373e-06, 
 'mag_R_std': 66290.74666972898, 
 'mag_G_std': 74220.80913376767, 
 'mag_B_std': 74752.1538781648, 
 'phase_R_std': 1.8129752834502697, 
 'phase_G_std': 1.8131023785194367, 
 'phase_B_std': 1.8126965600959166, 
 'number_of_images': 2800, 
 'number_of_pixels': 608967730}

================== STATS FOR DIFF IMAGES ==================
{'mag_R_max': 4149842.8029009253, 
 'mag_G_max': 3182684.735184118, 
 'mag_B_max': 3333820.9423187114, 
 'phase_R_max': 6.283185307179586, 
 'phase_G_max': 6.283185307179586, 
 'phase_B_max': 6.283185307179586, 
 'mag_R_min': -27906869.0, 
 'mag_G_min': -23951451.0, 
 'mag_B_min': -24936859.0, 
 'phase_R_min': -6.283185307179586, 
 'phase_G_min': -6.283185307179586, 
 'phase_B_min': -6.283185307179586, 
 'mag_R_mean': 34.63145890222781, 
 'mag_G_mean': 135.34962504670045, 
 'mag_B_mean': 144.18493092819386, 
 'phase_R_mean': 7.067668323583826e-07, 
 'phase_G_mean': 7.428790062749291e-07, 
 'phase_B_mean': -4.5398161494717937e-07, 
 'mag_R_std': 10419.150481819577, 
 'mag_G_std': 9409.519604082185, 
 'mag_B_std': 9443.189321412372, 
 'phase_R_std': 1.606171782862127, 
 'phase_G_std': 1.6038872204529218, 
 'phase_B_std': 1.6265938990072801, 
 'number_of_images': 2800, 
 'number_of_pixels': 608967730}

# TODOs:
Preprocessing 
- write code to preprocess all images and construct the .csv files - Done
- Normalization : need to normalize all magnitude by first logging it and normalize it to [-1, 1]

Training 
- Write and proofread training architecture ~ 90%
- Get training code done - Done

Postprocessing 
- After generating the fft we need to denormalize it 
- After denormalizinng the fft we need to apply on the derained image's fft and get the reconstructed image  

## Worthy experiments:
1. Instead of generating Nx6 channels, just generate the FFT (both real and imaginary part separately)
2. Instead of generating the difference in FFT, just try to generate the original FFT conditioned on the 
   rain FFT

## Ignore this for now 

## Description

### IMPORTANT
All npz files have a "mag_and_phase" vector that contains the Nx6 mag_and_phase vector of the 
image, where the first Nx3 is the magnitude, and the next Nx3 is the phase

groundtruth_id:           The prefix of the groundtruth image, e.g 995.jpg will have an id of 995
rain_id:                  The prefix of the rain image, e.g 995_11.jpg will have an id of 995_11
diff_npz_filepath:        The npz filepath for difference in FFT (magnitude and phase) 
                          between the groundtruth and its rain counterpart, this is what 
                          will be noised by the cosine scheduler and the diffusion model 
                          will learn how to generate this given the FFT of the rain image
groundtruth_npz_filepath: The npz filepath of the FFT of the groundtruth image (magnitude and phase) 
rain_npz_filepath:        The npz filepath for the FFT of the rained image (magnitude and phase)
groundtruth_filepath:     The filepath to the groundtruth image
rain_filepath:            The filepath to the rain image

