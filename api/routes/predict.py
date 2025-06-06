from fastapi import APIRouter, UploadFile, File
from PIL import Image
import torch
import numpy as np
import cv2
import base64
from io import BytesIO

from api.core.model import load_model
from api.core.image import preprocess_image, gen_cam, prepare_input
from api.core.gradcam import GradCam

router = APIRouter()

model = load_model("vit/model/vit.pth")
model.eval()
class_names = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"] # Make sure to match the class order in predictions

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    # Prediction
    img_tensor = preprocess_image(img)

    with torch.no_grad():
        y_pred = model(img_tensor)
        _, predicted = torch.max(y_pred, 1)
        label = class_names[predicted.item()]
    
    # Resize original image to match heatmap size
    img_resized_pil = img.resize((224, 224))
    buffered = BytesIO()
    img_resized_pil.save(buffered, format="PNG")
    original_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Heatmap
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (224, 224))
    img_normalized = np.float32(img_resized) / 255.0
    
    inputs = prepare_input(img_normalized)
    target_layer = model.mlp_head.norm

    grad_cam = GradCam(model, target_layer)
    mask = grad_cam(inputs)
    heatmap_b64 = gen_cam(img_normalized, mask)

    return {"prediction": label, "original_image": original_b64,"box_image": heatmap_b64}
