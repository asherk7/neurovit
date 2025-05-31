"""
Final MLP head used for classification in the Vision Transformer (ViT).

According to the paper:
- A LayerNorm is applied before the final linear classification layer.
- Only the output corresponding to the [CLS] token (classification token) is used for prediction.

Architecture:
LayerNorm → Linear → Output (batch_size, num_classes)
"""

from torch import nn

class MLP_Head(nn.Module):
    def __init__(self, embedded_dim=768, num_classes=4):
        """
        Initializes the MLP head.

        Args:
            embedded_dim (int): Dimensionality of the transformer output embeddings.
            num_classes (int): Number of output classes for classification.
        """
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape=embedded_dim)

        # This linear layer maps the embeddings to the number of classes (class logits)
        self.linear = nn.Linear(in_features=embedded_dim, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass through the MLP head.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, embedded_dim).

        Returns:
            Tensor: Class prediction logits of shape (batch_size,)
        """
        x = self.norm(x)
        x = self.linear(x)
        predictions = x[:, 0] # Takes the CLS token, which is of shape (batch_size, num_classes), containing the predictions for each class, for each batch sent through the model
        return predictions
