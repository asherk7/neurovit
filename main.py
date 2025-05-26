import os
import torch

from utils.data_setup import create_dataloaders, transform_images
from utils.eda.visualizations import plot_image

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
    
if __name__ == '__main__':
    main()