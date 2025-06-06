from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms

def preprocess_image(image):    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) 
    ])

    input_tensor = transform(image).unsqueeze(0) # Adding a batch dimension

    return input_tensor

def extract_bounding_box(heatmap: np.ndarray, threshold: float = 0.5):
    """
    Extract bounding box from heatmap by thresholding.
    """
    heatmap = np.uint8(255 * heatmap)
    _, binary_map = cv2.threshold(heatmap, int(255 * threshold), 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    return (x, y, x + w, y + h)
