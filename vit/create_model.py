import torch
import sys
import os

# Add project root to sys.path to allow importing local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vit.data_setup import create_dataloaders, transform_images
from utils import set_seeds, load_pretrained_weights, model_summary, get_metrics
from eda.visualizations import visualize
from pipeline.train import train
from pipeline.test import test
from transformer.vit import ViT
from transformers import get_cosine_schedule_with_warmup

# Vision Transformer Parameters

# Model parameters from the original ViT paper
PATCH_SIZE = 16
NUM_HEADS = 12
MLP_SIZE = 3072
MLP_DROPOUT = 0.1
ATTENTION_DROPOUT = 0.1  # Paper used 0.0 for large datasets, it's increased here to prevent overfitting on this smaller dataset

IN_CHANNELS = 3  # RGB images
EMBEDDED_DIM = PATCH_SIZE * PATCH_SIZE * IN_CHANNELS
NUM_PATCHES = (224 * 224) // (PATCH_SIZE * PATCH_SIZE)  # Paper works best with 224x224 image size

# Training hyperparameters from the paper, adjusted due to limitations of the dataset and training environment
NUM_EPOCHS = 25
BATCH_SIZE = 32 # The paper used 4096 for the large ImageNet dataset
LEARNING_RATE = 0.0001 # The paper used 3e-3 (0.003)
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01 # The paper used 0.1

# Dataset directories
train_dir = "data/Training"
test_dir = "data/Testing"

# Pretrained weights from PyTorch ViT model trained on ImageNet1k
pretrained_weights = 'vit/model/vit_b_16-c867db91.pth'

def main():
    # Set device to MPS (Apple Silicon) if available, otherwise use CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Set random seeds for reproducibility
    set_seeds()

    # Create DataLoaders
    train_transformer = transform_images(train=True)
    test_transformer = transform_images(train=False)
    train_dataloader, val_dataloader, test_dataloader, classes = create_dataloaders(
        train_dir=train_dir, 
        test_dir=test_dir, 
        train_transform=train_transformer, 
        test_transform=test_transformer,
        batch_size=BATCH_SIZE
    )
    
    num_classes = len(classes)

    # Initialize the Vision Transformer model
    model = ViT(
        embedded_dim=EMBEDDED_DIM,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        num_patches=NUM_PATCHES,
        num_heads=NUM_HEADS,
        mlp_size=MLP_SIZE,
        mlp_dropout=MLP_DROPOUT,
        attention_dropout=ATTENTION_DROPOUT,
        num_classes=num_classes
    ).to(device)
    
    # Load pretrained weights from HuggingFace ViT model
    model = load_pretrained_weights(model, pretrained_weights=pretrained_weights)

    # Print model summary
    #model_summary(model)
    
    # The loss function used in the paper is CrossEntropyLoss
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define Adam Optimizer as in the paper
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY
    )

    # Total and warmup steps for learning rate scheduler
    total_steps = len(train_dataloader) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    # Cosine LR scheduler with warmup (helps avoid bad local minima early in training)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Train the model
    results = train(model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    epochs=NUM_EPOCHS,
                    device=device)
    
    # Test the model
    y_pred, y_true = test(model=model, test_dataloader=test_dataloader, device=device)
    
    # Visualize results and calculate metrics
    visualize(results, y_pred, y_true, classes)
    get_metrics(y_pred, y_true)

    # Save the trained model state dictionary (weights)
    torch.save(obj=model.state_dict(), f='vit/model/vit.pth')

if __name__ == '__main__':
    main()