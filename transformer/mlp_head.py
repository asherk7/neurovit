"""
the final mlp head

just consists of a normalization, then a linear layer for outputs
"""

from torch import nn

class MLP_Head(nn.Module):
    def __init__(self, embedded_dim=768, num_classes=4):
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape=embedded_dim)

        self.linear = nn.Linear(in_features=embedded_dim, out_features=num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        predictions = x[:, 0]
        return predictions