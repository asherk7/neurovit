import torch
import onnxruntime as ort
from vit.transformer.vit import ViT

def load_model(weights_path: str):
    """
    Loads a Vision Transformer (ViT) model with pretrained weights.

    This function initializes the ViT architecture and loads the specified weights
    into the model. The model is loaded on the GPU by default.

    Args:
        weights_path (str): Path to the saved model weights (.pt or .pth file).

    Returns:
        ViT: The Vision Transformer model with loaded weights.
    """
    model = ViT() 
    model.load_state_dict(torch.load(weights_path, map_location="cpu")) # Change to "cpu" if you want to load on CPU
    return model

def load_onnx_model(model_path: str):
    session = ort.InferenceSession(model_path)
    return session
