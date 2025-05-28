"""
multihead attention class

takes the input x, applies normalization, then sends it through a multihead attention layer

according to the paper, heads=12, and dropout isn't used after the qkv-projections (the transformer block)
"""
from torch import nn

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embedded_dim=768, num_heads=12, dropout=0):
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape=embedded_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedded_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         batch_first=True) #our batch size comes first
    
    def forward(self, x):
        x = self.norm(x)
        x, _ = self.multihead_attention(query=x,
                                        key=x,
                                        value=x,
                                        need_weights=False) # we don't need the result weights
        return x
