import numpy as np
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def generate_fft(img_path, resize_shape=None):
    
    img = cv2.imread(img_path)
    
    if img is None:  # If the image is not read successfully
        raise Exception(f"Error: Could not read image at {img_path}")
    
    if resize_shape:
        img = cv2.resize(img, (resize_shape, resize_shape))

    b, g, r = cv2.split(img)
    fft_r = np.fft.fft2(r)
    fft_g = np.fft.fft2(g)
    fft_b = np.fft.fft2(b)
    fft_r_shifted = np.fft.fftshift(fft_r)
    fft_g_shifted = np.fft.fftshift(fft_g)
    fft_b_shifted = np.fft.fftshift(fft_b)
    mag_r, phase_r = np.abs(fft_r_shifted), np.angle(fft_r_shifted)
    mag_g, phase_g = np.abs(fft_g_shifted), np.angle(fft_g_shifted)
    mag_b, phase_b = np.abs(fft_b_shifted), np.angle(fft_b_shifted)
    mag_r_spectrum = np.log1p(mag_r)
    mag_g_spectrum = np.log1p(mag_g)
    mag_b_spectrum = np.log1p(mag_b)
    # mag_r_spectrum = (mag_r_spectrum / np.max(mag_r_spectrum) * 255).astype(np.uint8)
    # mag_g_spectrum = (mag_g_spectrum / np.max(mag_g_spectrum) * 255).astype(np.uint8)
    # mag_b_spectrum = (mag_b_spectrum / np.max(mag_b_spectrum) * 255).astype(np.uint8)
    # NOTE: OpenCV uses BGR convention
    magnitude_spectrum = cv2.merge([mag_r_spectrum, mag_g_spectrum, mag_b_spectrum])
    phase_spectrum = cv2.merge([phase_r, phase_g, phase_b])
    orig_magnitude_spectrum = cv2.merge([mag_r, mag_g, mag_b])
    return magnitude_spectrum, phase_spectrum, orig_magnitude_spectrum

def generate_fft_YCRCB(img_path, resize_shape=None):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    if img is None:  # If the image is not read successfully
        raise Exception(f"Error: Could not read image at {img_path}")
    
    if resize_shape:
        img = cv2.resize(img, (resize_shape, resize_shape))

    Y, Cr, Cb = cv2.split(img)
    fft_Y = np.fft.fft2(Y)
    fft_Cr = np.fft.fft2(Cr)
    fft_Cb = np.fft.fft2(Cb)
    fft_Y_shifted = np.fft.fftshift(fft_Y)
    fft_Cr_shifted = np.fft.fftshift(fft_Cr)
    fft_Cb_shifted = np.fft.fftshift(fft_Cb)
    mag_Y, phase_Y = np.abs(fft_Y_shifted), np.angle(fft_Y_shifted)
    mag_Cr, phase_Cr = np.abs(fft_Cr_shifted), np.angle(fft_Cr_shifted)
    mag_Cb, phase_Cb = np.abs(fft_Cb_shifted), np.angle(fft_Cb_shifted)
    mag_Y_spectrum = np.log1p(mag_Y)
    mag_Cr_spectrum = np.log1p(mag_Cr)
    mag_Cb_spectrum = np.log1p(mag_Cb)
    # mag_Y_spectrum = (mag_Y_spectrum / np.max(mag_Y_spectrum) * 255).astype(np.uint8)
    # mag_Cr_spectrum = (mag_Cr_spectrum / np.max(mag_Cr_spectrum) * 255).astype(np.uint8)
    # mag_Cr_spectrum = (mag_Cr_spectrum / np.max(mag_Cr_spectrum) * 255).astype(np.uint8)
    # NOTE: OpenCV uses BGR convention
    magnitude_spectrum = cv2.merge([mag_Y_spectrum, mag_Cr_spectrum, mag_Cb_spectrum])
    phase_spectrum = cv2.merge([phase_Y, phase_Cr, phase_Cb])
    orig_magnitude_spectrum = cv2.merge([mag_Y, mag_Cr, mag_Cb])
    return magnitude_spectrum, phase_spectrum, orig_magnitude_spectrum

def generate_fft_YCRCB_training(img_path, resize_shape=None):
    
    img = cv2.imread(img_path)

    if img is None:  # If the image is not read successfully
        raise Exception(f"Error: Could not read image at {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if resize_shape:
        img = cv2.resize(img, (resize_shape, resize_shape))

    Y, _, _ = cv2.split(img)
    fft_Y = np.fft.fft2(Y)
    fft_Y_shifted = np.fft.fftshift(fft_Y)
    mag_Y, phase_Y = np.abs(fft_Y_shifted), np.angle(fft_Y_shifted)
    mag_Y_spectrum = np.log1p(mag_Y)
    magnitude_spectrum = cv2.merge([mag_Y_spectrum])
    phase_spectrum = cv2.merge([phase_Y])
    orig_magnitude_spectrum = cv2.merge([mag_Y])
    return magnitude_spectrum, phase_spectrum, orig_magnitude_spectrum

def normalize_mag(mag):
    mag_spectrum = np.log1p(mag)
    mag_spectrum = (mag_spectrum / np.max(mag_spectrum) * 255).astype(np.uint8)
    return mag_spectrum

def generate_fft_diff(orig_img_path, rain_img_path, output_npz="diff.npz"):
        
    orig_fft, _, orig_unnormalized = generate_fft(orig_img_path)
    rain_fft, _, rain_unnormalized = generate_fft(rain_img_path)
    diff_unnoramlized = orig_unnormalized - rain_unnormalized
    np.savez(output_npz, mag_r=diff_unnoramlized[:, :, 2], mag_g=diff_unnoramlized[:, :, 1], mag_b=diff_unnoramlized[:, :, 0])

    return orig_fft - rain_fft

def generate_fft_for_preprocessing(orig_img_path, rain_img_path, resize_shape=None):
    """
    orig_img_path (str): File path of the original image
    rain_img_path (str): File path of the rain image
    output_npz 
    """
    _, orig_phase, orig_unnormalized = generate_fft(orig_img_path, resize_shape=resize_shape)
    _, rain_phase, rain_unnormalized = generate_fft(rain_img_path, resize_shape=resize_shape)

    return orig_unnormalized, orig_phase, rain_unnormalized, rain_phase

def concat_mag_and_phase(magnitude, phase):
    """
    magnitude: An WxHx3 vector, the magnitude of the image
    phase:     An WxHx3 vector, the phase of the image

    return:    An WxHx6 vector by concatenating magnitude and phase together 
    """

    if magnitude.shape != phase.shape:
        raise ValueError("Magnitude and phase must have the same shape.")
    
    return np.concatenate([magnitude, phase], axis=-1)  

def generate_fft_and_save_to_npz(image_path, output_image, output_npz):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image. Check the path.")

    b, g, r = cv2.split(img)

    fft_r = np.fft.fft2(r)
    fft_g = np.fft.fft2(g)
    fft_b = np.fft.fft2(b)
    fft_r_shifted = np.fft.fftshift(fft_r)
    fft_g_shifted = np.fft.fftshift(fft_g)
    fft_b_shifted = np.fft.fftshift(fft_b)
    mag_r, phase_r = np.abs(fft_r_shifted), np.angle(fft_r_shifted)
    mag_g, phase_g = np.abs(fft_g_shifted), np.angle(fft_g_shifted)
    mag_b, phase_b = np.abs(fft_b_shifted), np.angle(fft_b_shifted)
    mag_r_spectrum = np.log1p(mag_r)
    mag_g_spectrum = np.log1p(mag_g)
    mag_b_spectrum = np.log1p(mag_b)
    mag_r_spectrum = (mag_r_spectrum / np.max(mag_r_spectrum) * 255).astype(np.uint8)
    mag_g_spectrum = (mag_g_spectrum / np.max(mag_g_spectrum) * 255).astype(np.uint8)
    mag_b_spectrum = (mag_b_spectrum / np.max(mag_b_spectrum) * 255).astype(np.uint8)
    magnitude_spectrum = cv2.merge([mag_b_spectrum, mag_g_spectrum, mag_r_spectrum])
    cv2.imwrite(output_image, magnitude_spectrum)
    np.savez(output_npz, mag_r=mag_r, phase_r=phase_r, mag_g=mag_g, phase_g=phase_g, mag_b=mag_b, phase_b=phase_b)

    print(f"FFT color image saved as: {output_image}")
    print(f"FFT data (magnitude & phase) saved as: {output_npz}")

def reconstruct_from_rain_and_diff_mag_only(rain_img_path, diff_npz):
    """
    """
    _, phase, mag = generate_fft(rain_img_path)
    # BGR -> RGB 
    mag = mag[..., ::-1]

    # diff_data is what will be learned by the diffusion model
    diff_data = np.load(diff_npz)

    diff_mag_r   = diff_data["mag_r"]  
    diff_mag_g   = diff_data["mag_g"]  
    diff_mag_b   = diff_data["mag_b"]  
    # new_mag = mag 
    mag[:, :, 0] += diff_mag_r
    mag[:, :, 1] += diff_mag_g
    mag[:, :, 2] += diff_mag_b

    reconstruct_from_fft(mag[:, :, 0], mag[:, :, 1], 
                         mag[:, :, 2], phase[:, :, 0], 
                         phase[:, :, 1], phase[:, :, 2], "recon_from_rain.png")
    
    

def normalize_diff(diff):
    # Normalize the difference to the range [0, 255]
    diff_min = np.min(diff)
    diff_max = np.max(diff)
    normalized_diff = (diff - diff_min) / (diff_max - diff_min) * 255
    return normalized_diff.astype(np.uint8)

def normalize_diff_float(diff):
    # Normalize the difference to the range [0, 255]
    diff_min = np.min(diff)
    diff_max = np.max(diff)
    normalized_diff = (diff - diff_min) / (diff_max - diff_min) 
    return normalized_diff

def reconstruct_from_rain_and_diff_mag_and_phase(groundtruth_image_path, rain_img_path, output_name="viz.png"):
    """
    First generate FFT for both images

    Output the photo as with 

    Original Image, Rain Image, 
    FFT (mag) of Original Image, FFT (mag) of Rain Image 
    FFT (phase) of Original Image, FFT (phase) of Rain Image 
    Diff (mag) of FFT, Diff (phase) of FFT
      reconstructed image from 
    """ 

    # Magnitude are ints, so it has to be in range [0, 255] 
    magnitude_spectrum_gt, phase_spectrum_gt, orig_magnitude_spectrum_gt = generate_fft(groundtruth_image_path)
    # convert BGR to RGB
    magnitude_spectrum_gt = magnitude_spectrum_gt[..., ::-1]
    orig_magnitude_spectrum_gt = orig_magnitude_spectrum_gt[..., ::-1]
    magnitude_spectrum_rain, phase_spectrum_rain, orig_magnitude_spectrum_rain = generate_fft(rain_img_path)
    magnitude_spectrum_rain = magnitude_spectrum_rain[..., ::-1]
    orig_magnitude_spectrum_rain = orig_magnitude_spectrum_rain[..., ::-1]
    # Phases are floats, so it has to be in range [0, 1] 
    phase_for_display_gt = (phase_spectrum_gt - phase_spectrum_gt.min()) / (phase_spectrum_gt.max() - phase_spectrum_gt.min()) 
    phase_for_display_rain = (phase_spectrum_rain - phase_spectrum_rain.min()) / (phase_spectrum_rain.max() - phase_spectrum_rain.min()) 
    fig, axs = plt.subplots(6, 2, figsize=(8, 30))  # 2 rows and 2 columns

    # Original and rain Image 
    img = cv2.imread(groundtruth_image_path)
    gt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    axs[0, 0].imshow(gt_img)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    img = cv2.imread(rain_img_path)
    rain_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    axs[0, 1].imshow(rain_img)
    axs[0, 1].set_title("Rain Image")
    axs[0, 1].axis("off")

    # Mag spectrum
    axs[1, 0].imshow(magnitude_spectrum_gt, cmap='gray')
    axs[1, 0].set_title("Magnitude Spectrum (Original)")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(magnitude_spectrum_rain, cmap='gray')
    axs[1, 1].set_title("Magnitude Spectrum (Rain)")
    axs[1, 1].axis("off")

    # Phase spectrum
    axs[2, 0].imshow(phase_for_display_gt, cmap='gray')
    axs[2, 0].set_title("Phase Spectrum (Original)")
    axs[2, 0].axis("off")

    axs[2, 1].imshow(phase_for_display_rain, cmap='gray')
    axs[2, 1].set_title("Phase Spectrum (Rain)")
    axs[2, 1].axis("off")

    # Diff in FFT
    diff_mag = orig_magnitude_spectrum_gt - orig_magnitude_spectrum_rain
    diff_phase = phase_spectrum_gt - phase_spectrum_rain
    diff_mag_norm = normalize_diff(diff_mag)
    diff_phase_norm = normalize_diff(diff_phase)

    axs[3, 0].imshow(diff_mag_norm, cmap='gray')
    axs[3, 0].set_title("Diff (Mag) of FFT")
    axs[3, 0].axis("off")

    axs[3, 1].imshow(diff_phase_norm, cmap='gray')
    axs[3, 1].set_title("Diff (Phase) of FFT")
    axs[3, 1].axis("off")

    # Reconstruction results
    # TODO: maybe also try phase only
    recon_mag = diff_mag + orig_magnitude_spectrum_gt
    recon_phase = phase_spectrum_rain
    reconstructed_image = reconstruct_from_fft_return(recon_mag[:, :, 0], recon_mag[:, :, 1], 
                         recon_mag[:, :, 2], recon_phase[:, :, 0], 
                         recon_phase[:, :, 1], recon_phase[:, :, 2])
    cv2.imwrite("recon_from_mag.png", reconstructed_image)
    reconstructed_image_rgb = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB)
    axs[4, 0].imshow(reconstructed_image_rgb)
    axs[4, 0].set_title("Reconstructed Image Mag only")
    axs[4, 0].axis("off")
    diff_image_gt = reconstructed_image_rgb - gt_img

    recon_mag = diff_mag + orig_magnitude_spectrum_gt
    recon_phase = diff_phase + phase_spectrum_rain
    reconstructed_image = reconstruct_from_fft_return(recon_mag[:, :, 0], recon_mag[:, :, 1], 
                         recon_mag[:, :, 2], recon_phase[:, :, 0], 
                         recon_phase[:, :, 1], recon_phase[:, :, 2])
    cv2.imwrite("recon_from_mag_and_phase.png", reconstructed_image)
    reconstructed_image_rgb = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB)
    
    axs[4, 1].imshow(reconstructed_image_rgb)
    axs[4, 1].set_title("Reconstructed Image both Mag and Phase")
    axs[4, 1].axis("off")

    diff_image_rain = reconstructed_image_rgb - gt_img
 
    axs[5, 0].imshow(diff_image_gt)
    axs[5, 0].set_title("Diff Image (Mag only)")
    axs[5, 0].axis("off")

    axs[5, 1].imshow(diff_image_rain)
    axs[5, 1].set_title("Diff Image (Mag and phase)")
    axs[5, 1].axis("off")

    plt.savefig(output_name)


def reconstruct_from_fft(mag_r, mag_g, mag_b, phase_r, phase_g, phase_b, output_image):
    fft_r = mag_r * np.exp(1j * phase_r)
    fft_g = mag_g * np.exp(1j * phase_g)
    fft_b = mag_b * np.exp(1j * phase_b)

    # Perform inverse FFT for each channel
    ifft_r = np.fft.ifft2(np.fft.ifftshift(fft_r))
    ifft_g = np.fft.ifft2(np.fft.ifftshift(fft_g))
    ifft_b = np.fft.ifft2(np.fft.ifftshift(fft_b))

    # Get real values (discard imaginary part)
    r_channel = np.abs(ifft_r)
    g_channel = np.abs(ifft_g)
    b_channel = np.abs(ifft_b)

    # Normalize each channel to [0, 255]
    r_channel = (r_channel / np.max(r_channel) * 255).astype(np.uint8)
    g_channel = (g_channel / np.max(g_channel) * 255).astype(np.uint8)
    b_channel = (b_channel / np.max(b_channel) * 255).astype(np.uint8)

    # Merge the channels back into a color image
    reconstructed_image = cv2.merge([b_channel, g_channel, r_channel])

    # Save the reconstructed color image
    cv2.imwrite(output_image, reconstructed_image)

    print(f"Reconstructed color image saved as: {output_image}")

def reconstruct_from_fft_return(mag_r, mag_g, mag_b, phase_r, phase_g, phase_b):
    fft_r = mag_r * np.exp(1j * phase_r)
    fft_g = mag_g * np.exp(1j * phase_g)
    fft_b = mag_b * np.exp(1j * phase_b)

    # Perform inverse FFT for each channel
    ifft_r = np.fft.ifft2(np.fft.ifftshift(fft_r))
    ifft_g = np.fft.ifft2(np.fft.ifftshift(fft_g))
    ifft_b = np.fft.ifft2(np.fft.ifftshift(fft_b))

    # Get real values (discard imaginary part)
    r_channel = np.abs(ifft_r)
    g_channel = np.abs(ifft_g)
    b_channel = np.abs(ifft_b)

    # Normalize each channel to [0, 255]
    r_channel = (r_channel / np.max(r_channel) * 255).astype(np.uint8)
    g_channel = (g_channel / np.max(g_channel) * 255).astype(np.uint8)
    b_channel = (b_channel / np.max(b_channel) * 255).astype(np.uint8)

    # Merge the channels back into a color image
    reconstructed_image = cv2.merge([b_channel, g_channel, r_channel])

    return reconstructed_image

def reconstruct_from_fft_npz(input_npz, output_image):
    # Load FFT data for each channel
    data = np.load(input_npz)
    mag_r, phase_r = data["mag_r"], data["phase_r"]
    mag_g, phase_g = data["mag_g"], data["phase_g"]
    mag_b, phase_b = data["mag_b"], data["phase_b"]
    print("mag_r", np.max(mag_r))
    print("mag_g", np.max(mag_g))
    print("mag_b", np.max(mag_b))

    # Reconstruct the complex FFT for each channel
    fft_r = mag_r * np.exp(1j * phase_r)
    fft_g = mag_g * np.exp(1j * phase_g)
    fft_b = mag_b * np.exp(1j * phase_b)

    # Perform inverse FFT for each channel
    ifft_r = np.fft.ifft2(np.fft.ifftshift(fft_r))
    ifft_g = np.fft.ifft2(np.fft.ifftshift(fft_g))
    ifft_b = np.fft.ifft2(np.fft.ifftshift(fft_b))

    # Get real values (discard imaginary part)
    r_channel = np.abs(ifft_r)
    g_channel = np.abs(ifft_g)
    b_channel = np.abs(ifft_b)

    # Normalize each channel to [0, 255]
    r_channel = (r_channel / np.max(r_channel) * 255).astype(np.uint8)
    g_channel = (g_channel / np.max(g_channel) * 255).astype(np.uint8)
    b_channel = (b_channel / np.max(b_channel) * 255).astype(np.uint8)

    # Merge the channels back into a color image
    reconstructed_image = cv2.merge([b_channel, g_channel, r_channel])

    # Save the reconstructed color image
    cv2.imwrite(output_image, reconstructed_image)

    print(f"Reconstructed color image saved as: {output_image}")

def visualize_fft(img_path, output_name="viz.png", resize_shape=None):
    # Magnitude are ints, so it has to be in range [0, 255] 
    magnitude_spectrum, phase_spectrum, orig_magnitude_spectrum = generate_fft(img_path, resize_shape)
    # Phases are floats, so it has to be in range [0, 1] 
    phase_for_display = (phase_spectrum - phase_spectrum.min()) / (phase_spectrum.max() - phase_spectrum.min()) 
    plt.figure(figsize=(12, 4))
    
    # Original Image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper visualization
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    # FFT Magnitude Spectrum
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Magnitude Spectrum (Log Scale)")
    plt.axis("off")

    # Phase Spectrum
    plt.subplot(1, 3, 3)
    plt.imshow(phase_for_display, cmap='gray')
    plt.title("Phase Spectrum")
    plt.axis("off")

    # plt.tight_layout()
    plt.savefig(output_name)


"""
if __name__ == "__main__":
    generate_fft_and_save_to_npz("dataset/images_rain/801_1.jpg", "801_1.jpg", "801_1.npz")
    reconstruct_from_fft_npz("801_1.npz", "801_1_recon.jpeg")
"""



"""
# Generate FFT (magnitude and phase)
magnitude, phase, orig_mag = generate_fft(image_path)

# Reconstruct image from FFT
reconstructed = reconstruct_from_fft_return(
    orig_mag[:, :, 2],  # R
    orig_mag[:, :, 1],  # G
    orig_mag[:, :, 0],  # B
    phase[:, :, 2],     # R
    phase[:, :, 1],     # G
    phase[:, :, 0],     # B
)

# Load and convert original image for comparison
original = cv2.imread(image_path)
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
reconstructed_rgb = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB)

# Display
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_rgb)
plt.title("Reconstructed from FFT")
plt.axis("off")

plt.tight_layout()
plt.show()

"""

image_path = "/Users/gennadumont/Downloads/MSPFN-deraining/model/test/test_data/R100H/inputcrop/1.png"

img = cv2.imread(image_path)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(ycrcb)

# FFT on just Y
f_Y = np.fft.fft2(Y)
fshift_Y = np.fft.fftshift(f_Y)
magnitude_Y = np.abs(fshift_Y)
phase_Y = np.angle(fshift_Y)


# Apply inverse FFT to filtered_Y (after masking)
"""
f_ishift = np.fft.ifftshift(filtered_fshift_Y)
Y_filtered = np.abs(np.fft.ifft2(f_ishift))
Y_filtered = np.clip(Y_filtered, 0, 255).astype(np.uint8)

filtered_ycrcb = cv2.merge((Y_filtered, Cr, Cb))
final_image = cv2.cvtColor(filtered_ycrcb, cv2.COLOR_YCrCb2BGR)
"""