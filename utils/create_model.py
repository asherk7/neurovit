import torch
import sys
import os

# Add the project root directory to sys.path so I can import from the transformer folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_setup import create_dataloaders, transform_images
from utils import set_seeds
from visualizations import visualize
from pipeline.train import train
from pipeline.test import test
from transformer.vit import ViT
from transformers import get_cosine_schedule_with_warmup

#Model Parameters found from the paper
PATCH_SIZE = 16 
NUM_HEADS = 12
MLP_SIZE = 3072
MLP_DROPOUT=0.1
ATTENTION_DROPOUT = 0
IN_CHANNELS = 3
EMBEDDED_DIM = PATCH_SIZE*PATCH_SIZE*IN_CHANNELS
NUM_PATCHES = (224*224)//(PATCH_SIZE*PATCH_SIZE)

#Hyperparmameters found from the paper
#Batch size has been reduced from 4096 due to limitations, weight decay and learning rate were taken from the small ImageNet parameters due to the small size of our dataset
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.003
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.3

train_dir = "data/Training"
test_dir = "data/Testing"

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    data_transformer = transform_images()
    train_dataloader, val_dataloader, test_dataloader, classes = create_dataloaders(train_dir=train_dir, 
                                                                    test_dir=test_dir, 
                                                                    transform=data_transformer, 
                                                                    batch_size=BATCH_SIZE)
    
    num_classes = len(classes)

    model = ViT(embedded_dim=EMBEDDED_DIM,
              patch_size=PATCH_SIZE,
              in_channels=IN_CHANNELS,
              num_patches=NUM_PATCHES,
              num_heads=NUM_HEADS,
              mlp_size=MLP_SIZE,
              mlp_dropout=MLP_DROPOUT,
              attention_dropout=ATTENTION_DROPOUT,
              num_classes=num_classes).to(device)
    
    # The paper uses the Adam optimizer 
    loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(params=model.parameters(), 
                                lr=LEARNING_RATE,
                                betas=BETAS,
                                weight_decay=WEIGHT_DECAY)
    

    total_steps = len(train_dataloader) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)

    # scheduler used in the paper (explain warmup for preventing optimizer trapped in local min, using annealing for adam)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    set_seeds()

    results = train(model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    epochs=NUM_EPOCHS,
                    device=device)
    
    visualize(results)

    test(model=model, test_dataloader=test_dataloader, device=device)
    
    torch.save(obj=model.state_dict(), f='transformer/model/vit.pth')

if __name__ == '__main__':
    main()