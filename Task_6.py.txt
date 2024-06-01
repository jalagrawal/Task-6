import cv2
import numpy as np
import os
import requests
from matplotlib import pyplot as plt

def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError("Error downloading the image. Check the URL.")
    
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Couldn't load image from the URL.")
    
    return image

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    edges = cv2.Canny(morphed, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_length = 100
    filtered_contours = [contour for contour in contours if cv2.arcLength(contour, False) > min_contour_length]
    
    cv2.drawContours(image, filtered_contours, -1, (0, 0, 255), 2)
    
    return image

def detect_cracks_in_folder(folder_path):
    output_folder = os.path.join(folder_path, "labeled_images")
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)
            if image is None:
                print(f"Skipping {filename}: unable to load image.")
                continue
            
            labeled_image = process_image(image)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, labeled_image)
            print(f"Labeled image saved as {output_path}")

# Example usage
folder_path = "path/to/your/folder"  # Replace with your folder path
detect_cracks_in_folder(folder_path)
