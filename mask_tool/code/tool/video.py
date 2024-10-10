import cv2
import os

# Specify the directory containing the images
image_folder = 'pic'

# Check if the folder exists
if not os.path.exists(image_folder):
    print(f"The specified folder does not exist: {image_folder}")
else:
    # Gather all image file names and sort them
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    images.sort()
    
    # Debug print: Check the list of images found
    print(f"Found {len(images)} images: {images}")

    # Check if images were found
    if len(images) == 0:
        print("No images found in the specified folder.")
    else:
        # Read the first image to get the size
        first_image_path = os.path.join(image_folder, images[0])
        first_image = cv2.imread(first_image_path)
        
        if first_image is None:
            print(f"Error loading the first image: {first_image_path}")
        else:
            height, width, layers = first_image.shape

            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
            video = cv2.VideoWriter('K-Eng2.mp4', fourcc, 0.5, (width, height))

            # Loop through all images and write them to the video file
            for image in images:
                img_path = os.path.join(image_folder, image)
                img = cv2.imread(img_path)
                if img is not None:
                    video.write(img)

            # Release the VideoWriter object
            video.release()

            print("Video created successfully.")
