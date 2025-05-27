"""
turn the input image into a sequence of image patches, adding a position number to specify order of patches (for the attention block)


according to the paper:
input shape is (height, width, channels) (224, 224, 3)
output shape is (num_patches, patch_size^2 * channels) (196, 768)
    where num_patches = height*width / patch_size^2

on table 5, the paper shows patch size = 16 is the best performing 

summarized: we take an image, divide it into patches of size 16x16 (imagine cutting an image into smaller squares), and flatten each patch into a vector.
"""

from torch import nn

class PatchEmbedding(nn.Module):
    #nn.Module runs .forward() method automatically when called, so we can use it like a function
    """
    Patch Embedding Layer for Vision Transformers.
    This layer divides an input image into patches and embeds each patch into a vector.
    Args:
        image_dim (tuple): Dimensions of the input image (height, width).
        patch_size (int): Size of each square patch.
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
    """
    def __init__(self, image_dim, patch_size=16, in_channels=3):
        super().__init__()
        self.image_size = image_dim[0]*image_dim[1]
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = patch_size * patch_size * in_channels  # Flattened size of each patch

        # Create a convolutional layer to extract patches
        self.patch_creator = nn.Conv2d(in_channels=in_channels,
                                    out_channels=self.embed_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size)
        
        self.flatten = nn.Flatten(start_dim=2)  # Flatten the spatial dimensions

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width)
        x = self.patch_creator(x)  # Apply convolution to extract patches
        x = self.flatten(x)  # Flatten the spatial dimensions
        x = x.transpose(1, 2)  # Rearrange to (batch_size, num_patches, embed_dim)
        return x