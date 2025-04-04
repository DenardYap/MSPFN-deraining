import cv2
import os


current_directory = os.getcwd()

input_folder = f"{current_directory}/model/test/test_data/R100H/inputcrop"  # input folder
output_folder = os.path.join(current_directory, 'output_images') # output folder

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Loop through each image file
for image_file in image_files:
    # Read the image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply filter (replace this with your filter logic)
    filtered_image = cv2.GaussianBlur(image, (5, 5), 0)  # Example filter

    # Define the output path
    output_path = os.path.join(output_folder, image_file)
    
    # Save the filtered image
    cv2.imwrite(output_path, filtered_image)

    print(f"Processed and saved: {image_file}")
