import os
import torch

from utils.data_setup import create_dataloaders, transform_images
from utils.eda.visualizations import plot_image
from transformer.patch_embedding import PatchEmbedding
from transformer.class_token import ClassToken
from transformer.position_embedding import PositionEmbedding

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

    embedded_dim = PATCH_SIZE * PATCH_SIZE * 3 
    num_of_patches = (image.shape[2] * image.shape[3]) // (PATCH_SIZE * PATCH_SIZE)

    #patch embedding layer
    patch_embedding = PatchEmbedding(patch_size=PATCH_SIZE, in_channels=image.shape[1], embedded_dim=embedded_dim).to(device)  
    image = patch_embedding(image) 
    print(f"Patch shape: {image.shape}")

    #class token
    class_token = ClassToken(embedded_dim=embedded_dim, batch_size=image.shape[0]).to(device)
    image = class_token(image)
    print(f"Class token shape: {image.shape}")

    #position embedding
    position_embedding = PositionEmbedding(num_patches=num_of_patches, embedded_dim=embedded_dim).to(device)
    image = position_embedding(image)
    print(f"Position embedding shape: {image.shape}")

    

if __name__ == '__main__':
    main()