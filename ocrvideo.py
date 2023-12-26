from PIL import Image, ImageEnhance
import os
from pytesseract import pytesseract

# Defining paths to tesseract.exe
# and the folder containing the images
path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
image_folder_path = r"detected_objects/car"  # Assuming all images are in this folder

# Providing the tesseract executable
# location to pytesseract library
pytesseract.tesseract_cmd = path_to_tesseract

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

# Process each image
for image_file in image_files:
    # Check if the file is an image (you can add more image extensions if needed)
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        # Construct the full path to the image
        image_path = os.path.join(image_folder_path, image_file)

        # Opening the image & storing it in an image object
        img = Image.open(image_path)

        # Enhance the image to improve text detection
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)  # Adjust the enhancement factor as needed

        # Passing the enhanced image object to image_to_string() function
        # This function will extract the text from the image
        text = pytesseract.image_to_string(img)

        # Check if text was detected in the image
        if text.strip():  # If text is not empty
            # Displaying the extracted text along with the image file name
            print(f"File: {image_file}")
            print(text[:-1])
            print("########")