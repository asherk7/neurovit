"""
Embedded patches block of the ViT paper

this block consists of taking the input image, turning it into a sequence of image patches, adding a vector for representing the entire image, then adding a positional vector to every image patch

the first section: patch embedding
taking the image and turning it into patches of images (16x16), then flattening the patches into a vector, so we have a sequence of vectors for each patch
input shape: (channels, height, width) (e.g. 3, 224, 224)
output shape: (num_patches, patch_size^2 * channels) (e.g. 196, 768)
    where num_patches = height*width / patch_size^2
on table 5, the paper shows patch size = 16 is the best performing so we will use that

second section: class token
a class token is a learnable vector that is prepended to the sequence of image patches
if we have 196 patches of size 768, we add a class token to the beginning (of size 768), so we have 197 tokens in total
the class token is used to represent the entire image

third section: positional embedding
the position embedding is a learnable vector that is added to each patch embedding to provide positional information
transformers dont understand order, so we create a position embedding for each patch, and add it to the patch embedding (vector on vector addition)
"""
import torch
from torch import nn

class EmbeddedPatches(nn.Module):
    #nn.Module runs .forward() method automatically when called, so we can use it like a function
    """
    Patch Embedding Layer for Vision Transformers.
    This layer divides an input image into patches and embeds each patch into a vector.
    Args:
        patch_size (int): Size of each square patch.
        in_channels (int): Number of input channels
        embedded_dim (int): Dimension of the embedded patches (usually patch_size^2 * in_channels).
        batch_size (int): Batch size for the input images.
        num_of_patches (int): Number of patches in the input image.
    """

    def __init__(self, patch_size=16, in_channels=3, embedded_dim=768, num_patches=196):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedded_dim = embedded_dim
        self.num_patches = num_patches

        self.patcher = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.embedded_dim, 
                                 kernel_size=self.patch_size, 
                                 stride=self.patch_size)
        self.flatten = nn.Flatten(start_dim=2)
        self.token = nn.Parameter(torch.randn(1, 1, self.embedded_dim), requires_grad=True)
        self.positions = nn.Parameter(torch.randn(1, self.num_patches+1, self.embedded_dim), requires_grad=True) # +1 for the class token

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.class_token(x)
        x = self.position_embedding(x)
        return x

    def patch_embedding(self, x):
        x = self.patcher(x)
        x = self.flatten(x)
        x = x.transpose(1, 2)
        return x

    def class_token(self, x):
        batch_size = x.shape[0]
        token = self.token.expand(batch_size, -1, -1)
        x = torch.cat((token, x), dim=1) # Concatenate class token to the start of the image patches
        return x
    
    def position_embedding(self, x):
        x = x + self.positions #add the vectors
        return x
