import numpy as np
import cv2
from PIL import Image

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
    mag_r_spectrum = (mag_r_spectrum / np.max(mag_r_spectrum) * 255).astype(np.uint8)
    mag_g_spectrum = (mag_g_spectrum / np.max(mag_g_spectrum) * 255).astype(np.uint8)
    mag_b_spectrum = (mag_b_spectrum / np.max(mag_b_spectrum) * 255).astype(np.uint8)
    # NOTE: OpenCV uses BGR convention
    magnitude_spectrum = cv2.merge([mag_b_spectrum, mag_g_spectrum, mag_r_spectrum])
    phase_spectrum = cv2.merge([phase_r, phase_g, phase_b])
    orig_magnitude_spectrum = cv2.merge([mag_b, mag_g, mag_r])
    return magnitude_spectrum, phase_spectrum, orig_magnitude_spectrum

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

def reconstruct_from_rain_and_diff(rain_img_path, diff_npz):
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

if __name__ == "__main__":
    generate_fft_and_save_to_npz("dataset/images_rain/801_1.jpg", "801_1.jpg", "801_1.npz")
    reconstruct_from_fft_npz("801_1.npz", "801_1_recon.jpeg")