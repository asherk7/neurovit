"""
multilayer perceptron of the transformer

according to the paper:
we normalize the input, then we put it through two linear layers separated by a GELU activation
each linear layer is followed by a dropout layer
structure: normalize --> linear --> GELU --> dropout (0.1 in table 3) --> linear --> dropout
the mlp size is 3072 in table 1 of the paper
"""

from torch import nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, embedded_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape=embedded_dim)

        self.mlp = nn.Sequential(nn.Linear(in_features=embedded_dim, out_features=mlp_size),
                                 nn.GELU(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(in_features=mlp_size, out_features=embedded_dim),
                                 nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.norm(x)
        x = self.mlp(x)
        return x
