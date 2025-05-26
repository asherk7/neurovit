import torch

def set_seeds(seed=42):
    torch.manual_seed(seed)
    #GPU
    torch.cuda.manual_seed(seed)