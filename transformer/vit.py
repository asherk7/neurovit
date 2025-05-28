"""
putting the whole model together
"""

import torch
from torch import nn
from transformer.embedded_patches import EmbeddedPatches
from transformer.encoder import Encoder
from transformer.mlp_head import MLP_Head

class ViT(nn.Module):
    def __init__(self, 
                 embedded_dim=768,
                 patch_size=16,
                 in_channels=3,
                 num_patches=196,
                 num_heads=12,
                 mlp_size=3072,
                 mlp_dropout=0.1,
                 attention_dropout=0,
                 num_classes=4,
                 ):
        super().__init__()

        self.embedded_patches = EmbeddedPatches(patch_size=patch_size, 
                                               in_channels=in_channels, 
                                               embedded_dim=embedded_dim, 
                                               num_patches=num_patches)
        
        self.encoder = nn.Sequential(*[Encoder(embedded_dim=embedded_dim,
                                                    num_heads=num_heads,
                                                    mlp_size=mlp_size,
                                                    mlp_dropout=mlp_dropout,
                                                    attention_dropout=attention_dropout) for _ in range(num_heads)])
        
        self.mlp_head = MLP_Head(embedded_dim=embedded_dim,
                                 num_classes=num_classes)

    def forward(self, x):
        x = self.embedded_patches(x)

        x = self.encoder(x)

        x = self.mlp_head(x)

        return x