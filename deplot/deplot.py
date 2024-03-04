import os
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image

processor = Pix2StructProcessor.from_pretrained('google/deplot')
model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

folder_path = '../seed_datasets'

# Initialize counter for number of PNGs found
num_png_files = 0

# Recursively iterate through all files and directories in folder 
for root, dirs, files in os.walk(folder_path):
    # Iterate through all files in the current directory
    for file_name in files:
        # Check if the file has a PNG extension
        if file_name.lower().endswith('.png'):
            num_png_files += 1
            # Construct the full path to the PNG file
            file_path = os.path.join(root, file_name)
            # DePlot the PNG
            image = Image.open(file_path)
            inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
            predictions = model.generate(**inputs, max_new_tokens=512)
            decoded_predictions = processor.decode(predictions[0], skip_special_tokens=True)
            print(f"Data for {file_path}:\n{decoded_predictions}\n")

# Print total number of PNGs found
print(f"Number of PNG files found: {num_png_files}")