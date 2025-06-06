"""
transformer encoder block

in the paper, it has the normalization and MSA block with a residual connection, then the normalization and MLP block with a residual connection
this is all encapsulated in the transformer encoder block, which will be this file

we know there are 12 layers of the encoder, as table 1 shows layers L = 12, and equations 1-4 show that the MSA and MLP blocks range from l=1...L for z_l and z_l-1 respectively
"""
"""
Transformer Encoder Block from Vision Transformer (ViT)

According to the ViT paper:
- Input goes through LayerNorm → Multi-Head Self-Attention (MSA) → Residual Connection (output = MSA(input) + input)
- Output goes through LayerNorm → Multi-Layer Perceptron (MLP) → Residual Connection (output = MLP(input) + input)
- Each encoder block processes its input and adds residual connections after MSA and MLP layers. (residual connections add the input before transformation to the output of the transformation)
- Number of encoder layers L = 12 (from Table 1)
- MSA and MLP layers are applied L times (see equations 1–4)

Architecture:
LayerNorm → MSA → Residual → LayerNorm → MLP → Residual
"""

from torch import nn
from vit.transformer.multihead_attention import MultiheadSelfAttention
from vit.transformer.multilayer_perceptron import MultiLayerPerceptron
#nn.TransformerEncoderLayer is the pytorch implementation, which is the better choice if not reimplementing a paper

class Encoder(nn.Module):
    def __init__(self, 
                 embedded_dim=768, 
                 num_heads=12, 
                 mlp_size=3072, 
                 mlp_dropout=0.1, 
                 attention_dropout=0):
        """
        Initializes the encoder block with a self-attention module and a feedforward MLP.

        Args:
            embedded_dim (int): Size of each input embedding vector.
            num_heads (int): Number of attention heads in the MSA module.
            mlp_size (int): Hidden layer size of the MLP.
            mlp_dropout (float): Dropout probability for the MLP.
            attention_dropout (float): Dropout probability for the attention mechanism.
        """
        super().__init__()
        
        self.self_attention = MultiheadSelfAttention(
            embedded_dim=embedded_dim, 
            num_heads=num_heads, 
            dropout=attention_dropout
        )
        
        self.mlp = MultiLayerPerceptron(
            embedded_dim=embedded_dim, 
            mlp_size=mlp_size, 
            dropout=mlp_dropout
        )
    
    def forward(self, x):
        """
        Forward pass of the encoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, embedded_dim)

        Returns:
            Tensor: Output tensor after applying MSA and MLP blocks with residual connections.
        """
        # Residual connection around MSA
        x = self.self_attention(x) + x

        # Residual connection around MLP
        x = self.mlp(x) + x
        
        return x
