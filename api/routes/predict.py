from fastapi import APIRouter, UploadFile, File
from PIL import Image
import torch
import base64
from io import BytesIO
import numpy as np

from api.core.model import load_model, load_onnx_model
from api.core.image import preprocess_image, gen_cam
from api.core.gradcam import GradCam

router = APIRouter()

# Load trained ViT model
model = load_model("vit/model/vit.pth")
onnx_model = load_onnx_model("vit/model/vit.onnx")
model.eval()

# Class names must match training order
class_names = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for tumor classification and Grad-CAM visualization.

    Accepts an image file, runs it through the Vision Transformer model to predict 
    the tumor type, and generates a Grad-CAM heatmap highlighting important regions.

    Args:
        file (UploadFile): Uploaded MRI brain scan image in PNG/JPEG format.

    Returns:
        dict: A JSON response containing:
            - prediction (str): Predicted tumor class.
            - original_image (str): Base64-encoded resized original image.
            - box_image (str): Base64-encoded Grad-CAM heatmap image.
    """
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")  # Ensure image is in RGB format

    # Preprocess image for model input
    img_tensor, inputs = preprocess_image(img)
    img_np = img_tensor.cpu().numpy()
    
    # ONNX inference
    ort_inputs = {onnx_model.get_inputs()[0].name: img_np}
    ort_outputs = onnx_model.run(None, ort_inputs)
    y_pred_onnx = ort_outputs[0]
    
    predicted = np.argmax(y_pred_onnx, axis=1)[0]
    label = class_names[predicted]

    # Resize the original image to 224x224 for visualization
    img_resized_pil = img.resize((224, 224))
    buffered = BytesIO()
    img_resized_pil.save(buffered, format="PNG")
    original_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Use final attention norm layer for Grad-CAM
    target_layer = model.encoder[-1].self_attention.ln
    grad_cam = GradCam(model, target_layer)
    mask = grad_cam(img_tensor)  # Generate Grad-CAM heatmap
    heatmap_b64 = gen_cam(inputs, mask)  # Superimpose and encode as base64

    return {
        "prediction": label,
        "original_image": original_b64,
        "box_image": heatmap_b64
    }
