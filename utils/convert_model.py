import torch
from vit.transformer.vit import ViT
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

model = ViT()
model.load_state_dict(torch.load("vit/model/vit.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
path = "vit/model/vit.onnx"

torch.onnx.export(
    model,
    dummy_input,
    path,
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=True
)

print(f"Exported to {path}")
