import numpy as np
import cv2
import base64
import torchvision.transforms as transforms

def preprocess_image(image):
    """
    Preprocesses the input image for model inference and Grad-CAM visualization.

    This function resizes the image to 224x224, normalizes it with standard ImageNet mean and 
    standard deviation values, and returns both the processed tensor and a mask (for visualization).

    Args:
        image (PIL.Image): The input image to preprocess.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The preprocessed image tensor (with batch dimension).
            - np.ndarray: The normalized image array for visualization.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Normalize with ImageNet statistics
    ])

    # Apply the transformation and convert to numpy for visualization
    input_mask = transform(image).permute(1, 2, 0).numpy()
    input_tensor = transform(image).unsqueeze(0) 

    return input_tensor, input_mask

def gen_cam(image, mask):
    """
    Generates a Grad-CAM heatmap and superimposes it on the original image.

    Args:
        image (np.ndarray): The original image as a NumPy array (H, W, C).
        mask (np.ndarray): The Grad-CAM mask (heatmap) to overlay on the image.

    Returns:
        str: The resulting image in base64-encoded PNG format.
    """
    # Create a heatmap from the Grad-CAM mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET) # Apply the jet colormap
    heatmap = np.float32(heatmap) / 255 # Normalize 

    # Superimpose the heatmap on the original image
    cam = (1 - 0.5) * heatmap + 0.5 * image # Blend the heatmap and original image
    cam = cam / np.max(cam) # Normalize
    cam = np.uint8(255 * cam) # Convert the result to an 8-bit image

    # Convert the final image to base64
    cam_base64 = cam_to_base64(cam)
    return cam_base64

def cam_to_base64(cam_img):
    """
    Converts an image to a base64-encoded PNG format for embedding or transmission.

    Args:
        cam_img (np.ndarray): The image to convert, in uint8 format.

    Returns:
        str: The base64-encoded image as a string.
    """
    # Encode image to PNG format in memory
    success, buffer = cv2.imencode('.png', cam_img)
    if not success:
        raise ValueError("Failed to encode CAM image")

    # Convert the encoded image to base64
    cam_base64 = base64.b64encode(buffer).decode('utf-8')
    return cam_base64
