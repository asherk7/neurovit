import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from utils import get_class_distribution

NUM_WORKERS = (os.cpu_count() // 2) or 0  # Use all available CPU cores, default to 0 if none are available

def create_dataloaders(train_dir, test_dir, train_transform: transforms.Compose, test_transform: transforms.Compose, batch_size, num_workers=NUM_WORKERS):
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
    train_data = datasets.ImageFolder(train_dir, train_transform)
    test_data = datasets.ImageFolder(test_dir, test_transform)

    classes = train_data.classes

    # get a train/val split
    train_size = int(0.8* len(train_data))
    val_size = len(train_data) - train_size
    
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

    #print("Train distribution:", get_class_distribution(train_dataset, train_data))
    #print("Val distribution:", get_class_distribution(val_dataset, train_data))

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

def transform_images(train=True):
    IMG_SIZE = 224 # Standard size for ViT models, according to table 3 in the paper

    if train:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # matching values to the pytorch pretrained weights
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # 3 values for RBG
        ])

    return transform

if __name__ == "__main__":
    create_dataloaders()