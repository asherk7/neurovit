import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_image(dataloader, classes):
    image_batch, label_batch = next(iter(dataloader))
    image, label = image_batch[0], label_batch[0]

    plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
    plt.title(classes[label])
    plt.axis(False)
    plt.show() 

def plot_training_curves(results):
    train_acc = results["train_acc"]
    val_acc = results["val_acc"]
    train_loss = results["train_loss"]
    val_loss = results["val_loss"]
    epochs = range(len(train_acc))

    plt.figure(figsize=(15, 7))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train_accuracy")
    plt.plot(epochs, val_acc, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig('utils/images/training_graph.png')

def get_matrix(y_pred, y_true, classes):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes, yticklabels=classes,
                cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig('utils/images/confusion_matrix.png')

def visualize(results, y_pred, y_true, classes):
    plot_training_curves(results)
    get_matrix(y_pred, y_true, classes)
    