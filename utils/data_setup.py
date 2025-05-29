import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

NUM_WORKERS = os.cpu_count() or 0  # Use all available CPU cores, default to 0 if none are available

def create_dataloaders(train_dir, test_dir, transform: transforms.Compose, batch_size, num_workers=NUM_WORKERS):
    """
    Creates training and testing DataLoaders using the directory path, turning them into PyTorch Datasets, then into PyTorch DataLoaders

    Args:
        train_dir: Path to the training directory
        test_dir: Path to the testing directory
        transform: torchvision transforms to apply to the training and testing data
        batch_size: Number of samples per batch in each of the DataLoaders
        num_workers: integer for number of workers per DataLoader
    """

    # create datasets using torchvision
    train_data = datasets.ImageFolder(train_dir, transform)
    test_data = datasets.ImageFolder(test_dir, transform)

    classes = train_data.classes

    # get a train/val split
    train_size = int(0.8* len(train_data))
    val_size = len(train_data) - train_size
    
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

    # turn the image datasets into data loaders
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size, 
                                  shuffle=True, 
                                  num_workers=num_workers, 
                                  pin_memory=True)
    
    val_dataloader = DataLoader(val_dataset,
                                batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)
    
    test_dataloader = DataLoader(test_data, 
                                 batch_size, 
                                 shuffle=False, 
                                 num_workers=num_workers, 
                                 pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, classes

def transform_images():
    IMG_SIZE = 224 #according to table 3 in the paper

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 2 tuples for mean/std, 3 values for RBG
    ])

    return transform

