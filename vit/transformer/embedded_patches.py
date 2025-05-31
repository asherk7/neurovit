"""
Embedded Patches Block of the Vision Transformer (ViT)

Explanation:
This module takes in the input image, divides it into patches, adds a vector to represent the entire image, then adds a positional vector to every image patch to retain positional information.

Patch Embedding:
Turns the input image into a sequence of image patches (16x16), then flattens the patches into a vector, resulting in a sequence of vectors for each patch of the image.
Input shape: (batch_size, channels, height, width) (e.g. 32, 3, 224, 224)
Output shape: (batch_size, num_patches, embedded_dim) (e.g. 32, 196, 768)
    num_patches = (height * width) / (patch_size^2) (according to the paper)
    embedded_dim = patch_size^2 * channels (according to the paper)
    patch_size is typically 16, as shown in Table 5 of the ViT paper, which is the best performing size.

Class Token:
Takes the sequence of patch embeddings and prepends a learnable class token to the sequence.
This class token is a learnable parameter that represents the entire image, and is the same shape as the patch embeddings (embedded_dim).

Positional Encoding:
Adds a learnable positional embedding to each patch embedding to retain positional information.
Transformers don't understand order/position, so the position embedding is created and added to each patch embedding (vector on vector addition).

Architecture:
Input Image → Patch Embedding → Class Token Addition → Positional Encoding
"""

import torch
from torch import nn

class EmbeddedPatches(nn.Module):
    def __init__(self, 
                 patch_size=16, 
                 in_channels=3, 
                 embedded_dim=768, 
                 num_patches=196):
        """
        Initializes the EmbeddedPatches module.

        Args:
            patch_size (int): Height and width of each square image patch.
            in_channels (int): Number of input channels in the image (e.g., 3 for RGB).
            embedded_dim (int): Output dimension of each patch embedding.
            num_patches (int): Total number of patches per image (e.g., 14x14 = 196 for 224x224 input).
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedded_dim = embedded_dim
        self.num_patches = num_patches

        # Patch embedding layer uses Conv2d to convert the input image into non-overlapping patches of size patch_size x patch_size, and embeds each patch into a vector of dimension embedded_dim.
        # We set kernel_size and stride to patch_size, so the convolution slides in non-overlapping steps, extracting patches without overlap. (A fixed-size frame that divides the image into patches).
        # Each 16x16 patch (with 3 channels) is linearly projected to a vector of dimension `embedded_dim` (e.g., 768).
        # The Conv2D treats each patch like a mini-image and outputs an embedded representation for it.
        self.patch_embeddings = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.embedded_dim, 
                                 kernel_size=self.patch_size, 
                                 stride=self.patch_size)
        
        # Flatten the output of the Conv2d layer to create a sequence of patch embeddings (batch_size, embedded_dim, height, width) -> (batch_size, num_patches, embedded_dim)
        self.flatten = nn.Flatten(start_dim=2)

        # Class token: a learnable parameter that will be prepended to the sequence of patch embeddings (captures the global information of the image)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedded_dim), requires_grad=True)

        # Positional embeddings: learnable parameters that will be added to each patch embedding to retain positional information
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches+1, self.embedded_dim), requires_grad=True) # Num_patches+1 for the class token

    def forward(self, x):
        """
        Applies patch embedding, adds class token and position embeddings.

        (batch_size, channels, height, width) -> (batch_size, channels, height', width') -> (batch_size, embedded_dim, num_patches+1) -> (batch_size, num_patches+1, embedded_dim)

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Embedded sequence of shape (B, N+1, D)
        """
        x = self.patch_embedding(x)
        x = self.class_token(x)
        x = self.position_embedding(x)
        return x

    def patch_embedding(self, x):
        """
        Converts image to patch embeddings.

        Args:
            x (Tensor): Input image tensor (B, C, H, W)

        Returns:
            Tensor: Patch embeddings (B, N, D)
        """
        x = self.patch_embeddings(x)
        x = self.flatten(x)
        x = x.transpose(1, 2)
        return x

    def class_token(self, x):
        """
        Prepends the class token to the patch sequence.

        Args:
            x (Tensor): Patch embeddings (B, N, D)

        Returns:
            Tensor: Sequence with class token (B, N+1, D)
        """
        batch_size = x.shape[0]
        token = self.cls_token.expand(batch_size, -1, -1) # Expand the class token to match the batch size (so it can be concatenated properly)
        x = torch.cat((token, x), dim=1) # Concatenate class token to the start of the image patches
        return x
    
    def position_embedding(self, x):
        """
        Adds positional encoding to each token in the sequence.

        Args:
            x (Tensor): Token sequence (B, N+1, D)

        Returns:
            Tensor: Positionally encoded sequence (B, N+1, D)
        """
        x = x + self.position_embeddings
        return x
