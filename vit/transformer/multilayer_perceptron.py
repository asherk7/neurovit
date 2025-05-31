"""
Multi-layer Perceptron (MLP) block used in the Transformer encoder.

According to the ViT paper:
- The input is first normalized using LayerNorm.
- Then passed through two Linear layers with a GELU activation in between.
- Dropout is applied after each Linear layer to prevent overfitting.

Architecture:
LayerNorm → Linear → GELU → Dropout → Linear → Dropout

MLP size is typically 3072, and default embedding size is 768,
as described in Table 1 and Table 3 of the ViT paper.
"""

from torch import nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, embedded_dim=768, mlp_size=3072, dropout=0.1):
        """
        Initializes the MLP block.

        Args:
            embedded_dim (int): Dimension of the input embedding.
            mlp_size (int): Hidden layer size of the MLP (usually 4x embedding dim).
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.ln = nn.LayerNorm(normalized_shape=embedded_dim)

        # MLP block: Linear → GELU → Dropout → Linear → Dropout
        self.linear = nn.Sequential(
            nn.Linear(in_features=embedded_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedded_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        """
        Forward pass through the MLP block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, embedded_dim).

        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        x = self.ln(x)
        x = self.linear(x)
        return x
