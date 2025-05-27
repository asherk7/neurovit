"""
creating a class token for a transformer model

according to the paper:
a class token is a learnable vector that is prepended to the sequence of image patches
if we have 196 patches of size 768, we add a class token to the beginning (of size 768), so we have 197 tokens in total
the class token is used to represent the entire image
"""

import torch
from torch import nn

class ClassToken(nn.Module):
    def __init__(self, embedded_dim, batch_size):
        super().__init__()
        self.embedded_dim = embedded_dim
        # Creating a learnable class token
        self.class_token = nn.Parameter(torch.randn(batch_size, 1, embedded_dim), requires_grad=True) 

    def forward(self, x):
        class_image = torch.cat((self.class_token, x), dim=1)  # Concatenate class token to the start of the image patches
        return class_image
