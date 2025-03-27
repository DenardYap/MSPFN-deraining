# Greatlakes

## Enter interactive mode for GPU training
salloc --job-name=interactive --cpus-per-task=4 --nodes=1 --mem=80G --time=01:00:00 --account=eecs556w25_class --partition=spgpu --gres=gpu:1

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

