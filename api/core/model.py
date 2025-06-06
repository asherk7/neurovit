import torch
from vit.transformer.vit import ViT  

def load_model(weights_path: str):
    model = ViT()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model
