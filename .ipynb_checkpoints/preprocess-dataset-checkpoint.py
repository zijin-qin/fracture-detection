import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

base_path = './Fracatlas'
subdirs = ['train', 'test', 'validation']
processed_images = []

def load_and_preprocess_images():
    for subdir in subdirs:
        image_path = os.path.join(base_path, subdir, 'images')
        
        for filename in os.listdir(image_path):
            # Constructed the full file path
            file_path = os.path.join(image_path, filename)
            
            # Loaded the image
            image = cv2.imread(file_path)
            if image is None:
                print(f"Could not load image {filename} in {subdir}")
                continue
            
            # Converted to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Normalized pixel brightness (0-1 range)
            normalized_image = grayscale_image / 255.0
            
            # Standardized (zero mean, unit variance)
            mean, std = np.mean(normalized_image), np.std(normalized_image)
            standardized_image = (normalized_image - mean) / std
            
            # Appended the processed image to the list
            processed_images.append((subdir, filename, standardized_image))
            
            # Displays a sample processed image
            plt.imshow(standardized_image, cmap='gray')
            plt.title(f'Processed Image: {filename} ({subdir})')
            plt.show()

    print("All images processed.")
    return processed_images

processed_images = load_and_preprocess_images()