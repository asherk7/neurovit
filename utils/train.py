import torch

from utils.data_setup import create_dataloaders, transform_images
from transformer.vit import ViT

#Model Parameters
PATCH_SIZE = 16 
NUM_HEADS = 12
MLP_SIZE = 3072
MLP_DROPOUT=0.1
ATTENTION_DROPOUT = 0
IN_CHANNELS = 3
EMBEDDED_DIM = PATCH_SIZE*PATCH_SIZE*IN_CHANNELS
NUM_PATCHES = (224*224)//(PATCH_SIZE*PATCH_SIZE)

#Hyperparmameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

train_dir = "data/Training"
test_dir = "data/Testing"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_transformer = transform_images()
    train_dataloader, test_dataloader, classes = create_dataloaders(train_dir, test_dir, data_transformer, BATCH_SIZE)

    vit = ViT(embedded_dim=EMBEDDED_DIM,
              patch_size=PATCH_SIZE,
              in_channels=IN_CHANNELS,
              num_patches=NUM_PATCHES,
              num_heads=NUM_HEADS,
              mlp_size=MLP_SIZE,
              mlp_dropout=MLP_DROPOUT,
              attention_dropout=ATTENTION_DROPOUT,
              num_classes=classes)

if __name__ == '__main__':
    main()