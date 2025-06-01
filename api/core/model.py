#load model and predict

import torch
from PIL import Image

from api.utils.image import preprocess

model = torch.load("path_to_your_trained_model.pt", map_location="cpu")
model.eval()

def predict(image: Image.Image):
    input_tensor = preprocess(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1)
    return pred.item(), probs.squeeze().tolist()
