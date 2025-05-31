import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from utils import get_class_distribution

# Use half the CPU cores or 0 if none are available
NUM_WORKERS = (os.cpu_count() // 2) or 0  

def create_dataloaders(train_dir, test_dir, 
                       train_transform: transforms.Compose, 
                       test_transform: transforms.Compose, 
                       batch_size: int, 
                       num_workers: int = NUM_WORKERS):
    """
    Creates DataLoaders for training, validation, and testing datasets.

    Args:
        train_dir (str): Path to the training directory.
        test_dir (str): Path to the testing directory.
        train_transform (transforms.Compose): Transformations for training images.
        test_transform (transforms.Compose): Transformations for validation/testing images.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader, class_names)
    """

    # Create datasets
    train_data = datasets.ImageFolder(train_dir, train_transform)
    test_data = datasets.ImageFolder(test_dir, test_transform)
    classes = train_data.classes

    # Split the training data into training and validation sets
    train_size = int(0.8* len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

    # Print class distributions for debugging purposes
    #print("Train distribution:", get_class_distribution(train_dataset, train_data))
    #print("Val distribution:", get_class_distribution(val_dataset, train_data))

    # Turn the image datasets into DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, classes

def transform_images(train=True):
    """
    Applies image transformations for training or testing datasets.

    Args:
        train (bool): Whether to return the training transforms.

    Returns:
        torchvision.transforms.Compose: Transform pipeline.
    """
    IMG_SIZE = 224 # Standard size for ViT models (From table 3 of the paper)

    if train:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # Matches values required for the pytorch pretrained weights
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # Values for each channel
        ])

    return transform

if __name__ == "__main__":
    raise NotImplementedError("Set up paths and transforms before calling create_dataloaders().")
