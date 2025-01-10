import time
from rembg import remove
from PIL import Image
import io

# Load the input image
input_path = 'box.avif'  # Replace with your input image path
output_path = 'box-1.png'  # Replace with your desired output path

# Start timer
start_time = time.time()

# Open the image and process it
with open(input_path, 'rb') as input_file:
    input_data = input_file.read()
    output_data = remove(input_data)

# Save the output image
with open(output_path, 'wb') as output_file:
    output_file.write(output_data)

# End timer and print time taken
end_time = time.time()
print(f'Background removed and saved as {output_path}')
print(f'Time taken: {end_time - start_time:.2f} seconds')
