"""
transformer encoder block

in the paper, it has the normalization and MSA block with a residual connection, then the normalization and MLP block with a residual connection
this is all encapsulated in the transformer encoder block, which will be this file

we know there are 12 layers of the encoder, as table 1 shows layers L = 12, and equations 1-4 show that the MSA and MLP blocks range from l=1...L for z_l and z_l-1 respectively
"""

from torch import nn
from transformer.multihead_attention import MultiheadSelfAttention
from transformer.multilayer_perceptron import MultiLayerPerceptron
#nn.TransformerEncoderLayer can be used when not recreating a paper

class Encoder(nn.Module):
    def __init__(self, 
                 embedded_dim=768, 
                 num_heads=12, 
                 mlp_size=3072, 
                 mlp_dropout=0.1, 
                 attention_dropout=0):
        super().__init__()
        
        self.self_attention = MultiheadSelfAttention(embedded_dim=embedded_dim, 
                                          num_heads=num_heads, 
                                          dropout=attention_dropout)
        
        self.mlp = MultiLayerPerceptron(embedded_dim=embedded_dim, 
                                        mlp_size=mlp_size, 
                                        dropout=mlp_dropout)
    
    def forward(self, x):
        #use residual connections
        x = self.self_attention(x) + x
        x = self.mlp(x) + x
        return x