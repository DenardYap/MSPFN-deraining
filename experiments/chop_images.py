from PIL import Image
import os

type_ = "medium"
# Directory containing images
input_dir = f"rain_images/{type_}"
output_dir = f"rain_images/{type_}"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        
        # Get image dimensions
        width, height = img.size
        mid = width // 2  # Middle of the image

        # Crop left and right halves
        left_half = img.crop((0, 0, mid, height))
        right_half = img.crop((mid, 0, width, height))

        # Save cropped images
        left_half.save(os.path.join(output_dir, f"{filename}_rain.png"))
        right_half.save(os.path.join(output_dir, f"{filename}_derain.png"))

        print(f"Processed {filename}")

print("All images processed successfully!")
