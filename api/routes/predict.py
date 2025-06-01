# image upload + model inference

from fastapi import APIRouter, UploadFile, File
from PIL import Image
from io import BytesIO
from api.core.model import predict
from api.core.redis import cache_prediction, store_prediction

router = APIRouter()

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    cached = cache_prediction(contents)
    if cached:
        return {"cached": True, **cached}
    
    image = Image.open(BytesIO(contents)).convert("RGB")
    pred_class, probs = predict(image)

    result = {
        "predicted_class": pred_class,
        "probabilities": probs
    }
    store_prediction(contents, result)
    return {"cached": False, **result}
