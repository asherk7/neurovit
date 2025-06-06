from fastapi import APIRouter, UploadFile, File
from PIL import Image
import torch
import numpy as np
import cv2
import base64
from io import BytesIO

from api.core.model import load_model
from api.core.image import preprocess_image

router = APIRouter()

model = load_model("vit/model/vit.pth")
model.eval()
class_names = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"] # Make sure to match the class order in predictions

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    img_tensor = preprocess_image(img)

    with torch.no_grad():
        y_pred = model(img_tensor)
        _, predicted = torch.max(y_pred, 1)
        label = class_names[predicted.item()]

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    original_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Simulated bounding box
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    cv2.rectangle(img_cv, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), (0, 255, 0), 2)
    cv2.putText(img_cv, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.png', img_cv)
    box_b64 = base64.b64encode(buffer).decode('utf-8')

    return {"prediction": label, "original_image": original_b64,"box_image": box_b64}
