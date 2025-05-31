"""
Visualization utilities for ViT model training and evaluation.

Includes:
- Sample image display with label
- Training curves (loss and accuracy)
- Confusion matrix plot
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_image(dataloader, classes):
    """
    Displays the first image in a batch with its label.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing image batches.
        classes (list): List of class names.
    """
    image_batch, label_batch = next(iter(dataloader))
    image, label = image_batch[0], label_batch[0]

    plt.imshow(image.permute(1, 2, 0)) # Rearrange image dimensions to suit matplotlib, (rgb, height, width) -> (height, width, rgb)
    plt.title(classes[label])
    plt.axis(False)
    plt.show() 

def plot_training_curves(results):
    """
    Plots training and validation accuracy and loss curves.

    Args:
        results (dict): Dictionary containing training and validation metrics:
                        keys = ["train_acc", "val_acc", "train_loss", "val_loss"]
    """
    train_acc = results["train_acc"]
    val_acc = results["val_acc"]
    train_loss = results["train_loss"]
    val_loss = results["val_loss"]
    epochs = range(len(train_acc))

    plt.figure(figsize=(15, 7))

    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train_accuracy")
    plt.plot(epochs, val_acc, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig('vit/eda/images/training_graph.png')

def get_matrix(y_pred, y_true, classes):
    """
    Plots a confusion matrix heatmap.

    Args:
        y_pred (list or array): Model predictions.
        y_true (list or array): True labels.
        classes (list): List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes, yticklabels=classes,
                cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig('vit/eda/images/confusion_matrix.png')

def visualize(results, y_pred, y_true, classes):
    """
    Combines training curve and confusion matrix visualizations.

    Args:
        results (dict): Training/validation metrics.
        y_pred (list or array): Model predictions.
        y_true (list or array): Ground truth labels.
        classes (list): Class names.
    """
    plot_training_curves(results)
    get_matrix(y_pred, y_true, classes)
    