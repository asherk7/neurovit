from torch import nn
from vit.transformer.embedded_patches import EmbeddedPatches
from vit.transformer.encoder import Encoder
from vit.transformer.mlp_head import MLP_Head

class ViT(nn.Module):
    def __init__(self, 
                 embedded_dim=768,
                 patch_size=16,
                 in_channels=3,
                 num_patches=196,
                 num_heads=12,
                 mlp_size=3072,
                 mlp_dropout=0.1,
                 attention_dropout=0.1,
                 num_classes=4):
        """
        Vision Transformer (ViT) model.

        Combines patch embedding, a stack of Transformer encoder blocks, and an MLP head for classification.

        Args:
            embedded_dim (int): Dimensionality of patch embeddings.
            patch_size (int): Size of image patches (e.g., 16 for 16x16 patches).
            in_channels (int): Number of input image channels (e.g., 3 for RGB).
            num_patches (int): Total number of patches per image (e.g., 196 for 14x14 patches).
            num_heads (int): Number of attention heads in multi-head self-attention.
            mlp_size (int): Hidden layer size in the feedforward MLP within each encoder block.
            mlp_dropout (float): Dropout rate in the MLP layers.
            attention_dropout (float): Dropout rate in the attention layers.
            num_classes (int): Number of output classes for classification.
        """
        super().__init__()

        # Embed input images into patch tokens + class token + positional encoding
        self.embeddings = EmbeddedPatches(
            patch_size=patch_size, 
            in_channels=in_channels, 
            embedded_dim=embedded_dim, 
            num_patches=num_patches
        )
        
        # Stack of Transformer encoder blocks
        self.encoder = nn.Sequential(
            *[Encoder(
                embedded_dim=embedded_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout,
                attention_dropout=attention_dropout
            ) for _ in range(num_heads)]  # using num_heads as the number of encoder layers
        )
        
        # MLP head for final classification
        self.mlp_head = MLP_Head(
            embedded_dim=embedded_dim, 
            num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass through the Vision Transformer.

        Shape: (batch_size, channels, height, width) -> (batch_size, num_patches + 1, embedded_dim) -> (batch_size, num_patches + 1, embedded_dim) -> (batch_size, num_classes)

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Logits of shape (B, num_classes)
        """
        x = self.embeddings(x) 
        x = self.encoder(x) 
        x = self.mlp_head(x) 
        return x
