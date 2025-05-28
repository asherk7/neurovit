import os
import torch

from utils.data_setup import create_dataloaders, transform_images
from utils.eda.visualizations import plot_image
from transformer.embedded_patches import EmbeddedPatches
from transformer.multihead_attention import MultiheadSelfAttention
from transformer.multilayer_perceptron import MultiLayerPerceptron

NUM_EPOCHS = 5
BATCH_SIZE = 32
PATCH_SIZE = 16 
NUM_HEADS = 12
MLP_SIZE = 3072
DROPOUT=0.1

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


    #transformer encoder section
    #multihead attention
    msa = MultiheadSelfAttention(embedded_dim=embedded_dim, num_heads=NUM_HEADS, dropout=0).to(device)
    image = msa(image)

    #mlp head section
    mlp = MultiLayerPerceptron(embedded_dim=embedded_dim, mlp_size=MLP_SIZE, dropout=DROPOUT).to(device)
    image = mlp(image)
    print(f"image shape: {image.shape}")

    #output



if __name__ == '__main__':
    main()