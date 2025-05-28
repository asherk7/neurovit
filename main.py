import os
import torch

from utils.data_setup import create_dataloaders, transform_images
from utils.eda.visualizations import plot_image
from transformer.embedded_patches import EmbeddedPatches

NUM_EPOCHS = 5
BATCH_SIZE = 32
PATCH_SIZE = 16 

def visualize():
    pass
    #do all image visualizations/saving images here, refactor the entire codebase to make sense (move eda folder into utils?)

def main():
    train_dir = "data/Training"
    test_dir = "data/Testing"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transform_images()

    train_dataloader, test_dataloader, classes = create_dataloaders(train_dir, test_dir, transform, BATCH_SIZE)

    #get image
    image = next(iter(train_dataloader))[0][0]  # Get the first image from the first batch
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    print(image.shape)

    embedded_dim = PATCH_SIZE * PATCH_SIZE * 3 
    num_patches = (image.shape[2] * image.shape[3]) // (PATCH_SIZE * PATCH_SIZE)

    #embedded patches section
    embedded_patches = EmbeddedPatches(patch_size=PATCH_SIZE, 
                                       in_channels=image.shape[1],
                                       embedded_dim=embedded_dim,
                                       num_patches=num_patches).to(device)
    image = embedded_patches(image)
    print(f"image shape: {image.shape}")


    #transformer encoder section

    #normalization
    #multi-head attention
    #residual block

    #normalization
    #multi-head attention
    #residual block


    #mlp head section


    #output



if __name__ == '__main__':
    main()