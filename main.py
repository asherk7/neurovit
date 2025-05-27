import os
import torch

from utils.data_setup import create_dataloaders, transform_images
from utils.eda.visualizations import plot_image
from transformer.input_embedding import PatchEmbedding

NUM_EPOCHS = 5
BATCH_SIZE = 32

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
    image_1 = next(iter(train_dataloader))[0][0]  # Get the first image from the first batch
    image_1 = image_1.unsqueeze(0).to(device)  # Add batch dimension

    #patch embedding layer
    patch_embedding = PatchEmbedding(image_dim=(224, 224), patch_size=16, in_channels=3).to(device)
    image_patches = patch_embedding(image_1)  # Get the patches
    print(f"Patch shape: {image_patches.shape}")

    #
    
if __name__ == '__main__':
    main()