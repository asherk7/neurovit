from PIL import Image
import numpy as np
import cv2
import base64
import torchvision.transforms as transforms
import torch

def preprocess_image(image):    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) 
    ])

    input_mask = transform(image).permute(1, 2, 0).numpy() # Changing dimensions and converting to NP array
    input_tensor = transform(image).unsqueeze(0) # Adding a batch dimension

    return input_tensor, input_mask

def gen_cam(image, mask):
    # Create a heatmap from the Grad-CAM mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Superimpose the heatmap on the original image
    cam = (1 - 0.5) * heatmap + 0.5 * image
    cam = cam / np.max(cam)  # Normalize the result
    cam = np.uint8(255 * cam) # Convert to 8-bit image
    cam_base64 = cam_to_base64(cam) # Base 64
    return cam_base64

def cam_to_base64(cam_img):
    # Encode image to PNG format in memory
    success, buffer = cv2.imencode('.png', cam_img)
    if not success:
        raise ValueError("Failed to encode CAM image")

    # Convert to base64
    cam_base64 = base64.b64encode(buffer).decode('utf-8')
    return cam_base64
