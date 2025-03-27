"""
Take the images and its rained counterpart (images and images_rain)
Convert both images into FFT (fft_original and fft_rain), then take the difference between 
the rain FFT and the original FFT (fft_diff), this will be the image that is fed to 
the diffusion pipeline (the one that will be noised). fft_rain will be used for conditioning 
during diffusion training.

Also save the fft_original, fft_rain, and fft_diff into its respective folders in dataset 
(with the same names as the image name in /dataset/images). 

Lastly, save all these into a .csv folder to be used in the dataloader.
"""
import csv 
import os 
