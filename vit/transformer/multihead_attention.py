"""
Multihead Self-Attention (MSA) module used in the Vision Transformer (ViT).

According to the ViT paper:
- The input is first normalized using LayerNorm (Pre-LN architecture).
- Then passed through a MultiheadAttention layer.
- No dropout is applied after the QKV projections in this specific implementation.
- The number of attention heads is 12, as stated in the base ViT configuration.

Architecture:
LayerNorm → MultiheadAttention
"""

from torch import nn

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embedded_dim=768, num_heads=12, dropout=0):
        """
        Initializes the multihead self-attention module.

        Args:
            embedded_dim (int): Dimension of input embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability within the attention mechanism.
        """
        super().__init__()

        self.ln = nn.LayerNorm(normalized_shape=embedded_dim)

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedded_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True # Ensures input shape is (batch_size, sequence_length, embedded_dim)
        )

    def forward(self, x):
        """
        Forward pass through the self-attention layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, embedded_dim).

        Returns:
            Tensor: Output tensor after applying self-attention, same shape as input.
        """
        x = self.ln(x) 
        x, _ = self.multihead_attention(
            query=x,
            key=x,
            value=x,
            need_weights=False  # We don’t need attention weights
        )
        return x
