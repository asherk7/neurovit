import matplotlib.pyplot as plt

def plot_image(dataloader, classes):
    image_batch, label_batch = next(iter(dataloader))
    image, label = image_batch[0], label_batch[0]

    plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
    plt.title(classes[label])
    plt.axis(False)
    plt.show() 

def accuracy_graph(results):
    train_acc = results["train_acc"]
    val_acc = results["val_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train_accuracy")
    plt.plot(epochs, val_acc, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def loss_graph(results):
    train_loss = results["train_loss"]
    val_loss = results["val_loss"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15,7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

def epoch_history():
    pass

def get_matrix():
    pass

def visualize(results):
    accuracy_graph(results)
    loss_graph(results)
    epoch_history()
    get_matrix()