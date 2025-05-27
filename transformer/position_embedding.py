"""
creating position embeddings for transformer models

the position embedding is a learnable vector that is added to each patch embedding to provide positional information
transformers dont understand order, so we create a position embedding for each patch, and add it to the patch embedding (vector on vector addition)
"""

import torch
from torch import nn

class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embedded_dim):
        super().__init__()
        self.num_patches = num_patches
        self.embedded_dim = embedded_dim
        
        # Create learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches+1, embedded_dim), requires_grad=True) # +1 for the class token
        
    def forward(self, x):
        # x shape: (batch_size, num_patches, embed_dim)
        return x + self.position_embeddings