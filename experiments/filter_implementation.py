import numpy as np
import os
import cv2
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
from skimage.restoration import wiener
from skimage import img_as_float


import matplotlib.pyplot as plt
import math

# choose filter selection:
section = "Guassian"
# section = "Median"
# section = "Bilateral"
# section = "Non-Local Means"
# section = "Wiener"

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(
            -((x - (size // 2))**2 + (y - (size // 2))**2) / (2 * sigma**2)
        ),
        (size, size)
    )
    return kernel / np.sum(kernel)

def apply_gaussian_filter(image, kernel):
    height, width, channels = image.shape
    pad_size = kernel.shape[0] // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    output_image = np.zeros_like(image)

    for i in range(channels):
        for y in range(height):
            for x in range(width):
                region = padded_image[y:y+kernel.shape[0], x:x+kernel.shape[1], i]
                output_image[y, x, i] = np.sum(region * kernel)
    
    return output_image

def median_filter(image, kernel_size=3):
    height, width, channels = image.shape
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)
    
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                # region of interest (neighborhood) 
                region = padded_image[i:i+kernel_size, j:j+kernel_size, c]
                # median of the region for this channel
                filtered_image[i, j, c] = np.median(region)
    
    return filtered_image

def bilateral_filter(img, d, sigma_color, sigma_space):
    img = np.float32(img)
    height, width, channels = img.shape
    filtered_img = np.zeros_like(img)

    # initial Gaussian kernel for spatial distance
    half_d = d // 2
    spatial_kernel = np.zeros((d, d), np.float32)
    for i in range(d):
        for j in range(d):
            spatial_kernel[i, j] = math.exp(-((i - half_d) ** 2 + (j - half_d) ** 2) / (2 * sigma_space ** 2))
    
    spatial_kernel /= spatial_kernel.sum()

    for y in range(height):
        for x in range(width):
            #  window boundaries
            x_min = max(x - half_d, 0)
            x_max = min(x + half_d + 1, width)
            y_min = max(y - half_d, 0)
            y_max = min(y + half_d + 1, height)
            
            region = img[y_min:y_max, x_min:x_max] # region of interest
            # color difference (intensity difference kernel)
            region_color_diff = region - img[y, x] 
            # intensity difference kernel (sum squared differences across channels)
            intensity_kernel = np.exp(-np.sum(region_color_diff ** 2, axis=2) / (2 * sigma_color ** 2))
            
            # spatial kernel * intensity kernel = weight kernel
            weight_kernel = spatial_kernel[(y_min - y + half_d):(y_max - y + half_d), 
                                           (x_min - x + half_d):(x_max - x + half_d)] * intensity_kernel
            
            weight_kernel /= weight_kernel.sum()
            # Apply the weighted sum to the region (weighted average)
            weighted_sum = np.sum(region * weight_kernel[..., np.newaxis], axis=(0, 1))
            
            filtered_img[y, x] = np.clip(weighted_sum, 0, 255)
    
    return np.uint8(filtered_img)

def non_local_means(img, h=10, template_window_size=7, search_window_size=21):
    img = np.float32(img)
    height, width, channels = img.shape
    denoised_img = np.zeros_like(img)
    half_template = template_window_size // 2
    half_search = search_window_size // 2

    for c in range(channels):
        # process each channel independently
        channel = img[:, :, c]
        denoised_channel = np.zeros_like(channel)

        for i in range(half_template, height - half_template):
            for j in range(half_template, width - half_template):
                # define the search window around the pixel (i, j)
                search_window = channel[max(i - half_search, 0): min(i + half_search + 1, height),
                                        max(j - half_search, 0): min(j + half_search + 1, width)]
                
                # define the template window around the pixel (i, j)
                template_window = channel[i - half_template:i + half_template + 1, j - half_template:j + half_template + 1]

                # Compute weights for all pixels in the search window based on similarity to the template window
                weights = np.zeros_like(search_window, dtype=np.float32)
                for x in range(search_window.shape[0] - template_window.shape[0] + 1):
                    for y in range(search_window.shape[1] - template_window.shape[1] + 1):
                        # Extract sub-region of search window that is the same size as the template window
                        sub_region = search_window[x:x + template_window.shape[0], y:y + template_window.shape[1]]
                        
                        # squared Euclidean distance between the template window and sub-region
                        squared_diff = np.sum((sub_region - template_window) ** 2)
                        
                        # weights for the sub-region
                        weights[x, y] = np.exp(-squared_diff / (h**2))

                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    weights /= weight_sum
                else:
                    weights = np.ones_like(weights)  # fallback: if no similar pixels, just average

                # get denoised value for the pixel (i, j)
                denoised_value = np.sum(weights * search_window[:weights.shape[0], :weights.shape[1]])
                denoised_channel[i, j] = denoised_value

        # assign the denoised channel back to the denoised image
        denoised_img[:, :, c] = denoised_channel

    return np.uint8(denoised_img)

def wiener_filter(img, noise_var, signal_var):
    if len(img.shape) == 3:
        channels = cv2.split(img)
        filtered_channels = []
        
        # apply Wiener filter to each channel separately
        for channel in channels:
            channel_gray = np.float32(channel)
            
            # FFt image and shift - zero freq to center
            img_f = np.fft.fft2(channel_gray)
            img_f_shifted = np.fft.fftshift(img_f)  

            H = np.abs(img_f_shifted) ** 2 / (np.abs(img_f_shifted) ** 2 + noise_var / signal_var)
            
            filtered_f = img_f_shifted * H
            
            # IFFT to get denoised image (shift needed)
            filtered_img = np.fft.ifftshift(filtered_f) 
            denoised_img = np.fft.ifft2(filtered_img)
            
            denoised_img_real = np.abs(denoised_img)
            
            filtered_channels.append(np.uint8(denoised_img_real))
        
        # merge the filtered channels back into one image
        filtered_image = cv2.merge(filtered_channels)
        return filtered_image
    else:
        # print("Input image must be a color image with 3 channels (BGR).")
        return None

def plotter(image, filtered_image):
    # Plotting original and filtered images side by side
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Filtered image
    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')

    plt.show()

# Example usage with OpenCV to load an image
if __name__ == "__main__":
    folder = True
    file = False

    current_directory = os.getcwd()

    input_folder = f"{current_directory}/model/test/test_data/R100H/inputcrop"  # input folder
    output_folder = os.path.join(current_directory, f'output_images_izzi_{section}') # output folder

    # create output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if folder:
        # list of image files for input 
        image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            if section == "Guassian":
                sigma = 1.5  # std for Gaussian kernel
                kernel_size = 5
                kernel = gaussian_kernel(kernel_size, sigma)

                # apply Guassian filter
                filtered_image = apply_gaussian_filter(image, kernel) # Izzi implementation (1)
                guassian_filtered = gaussian_filter(image, sigma=sigma) # library implementation (2)

                # plotter(image, filtered_image)

                print("Library Implementation Pixel Value at (50, 50):", guassian_filtered[50, 50])
                print("Implemented Pixel Value at (50, 50):", filtered_image[50, 50])

            if section == "Median":
                # apply median filter (3x3 kernel)
                # filtered_image = cv2.medianBlur(image, 3)
                filtered_image = median_filter(image, kernel_size=3) # Izzi implementation (1)
                median_filtered = cv2.medianBlur(image, 3) # library implementation (2)

                # plotter(image, filtered_image)

                print("Library Implementation Pixel Value at (50, 50):", median_filtered[50, 50])
                print("Implemented Pixel Value at (50, 50):", filtered_image[50, 50])

            if section == "Bilateral":
                d = 15  # ciameter of the neighborhood
                sigma_color = 75  # std - intensity difference
                sigma_space = 75  # std - spatial distance

                # apply bilateral filter
                filtered_image = bilateral_filter(image, d, sigma_color, sigma_space) # Izzi implementation (1)
                bilateral_filtered = cv2.bilateralFilter(image, d=15, sigmaColor=75, sigmaSpace=75) # library implementation (2)
             
                # plotter(image, filtered_image)

                print("Library Implementation Pixel Value at (50, 50):", bilateral_filtered[50, 50])
                print("Implemented Pixel Value at (50, 50):", filtered_image[50, 50])

            if section == "Non-Local Means":
                h = 10

                # apply NLM filter
                filtered_image = non_local_means(image, h=h) # Izzi implementation (1)
                nlm_filtered = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21) # library implementation (2)
                
                # plotter(image, filtered_image)

                print("Library Implementation Pixel Value at (50, 50):", nlm_filtered[50, 50])
                print("Implemented Pixel Value at (50, 50):", filtered_image[50, 50])
            
            if section == "Wiener":
                noise_var = 0.01  # noise variance
                signal_var = 1.0  # signal variance (>1 - depending on image quality)

                # apply Wiener Filter
                filtered_image = wiener_filter(image, noise_var, signal_var) # Izzi implementation (1)
                # image_float = img_as_float(image)  # convert the image to float [0, 1]
                # filtered_image = wiener(image_float, (5, 5)) # library implementation (2)
   
                # plotter(image, filtered_image)

                print("Library Implementation Pixel Value at (50, 50):", bilateral_filtered[50, 50])
                print("Implemented Pixel Value at (50, 50):", filtered_image[50, 50])

            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, filtered_image)

            print(f"Processed and saved: {image_file}")


    # print("Original Pixel Value at (50, 50):", image[50, 50])
    # print("Filtered Pixel Value at (50, 50):", filtered_image[50, 50])