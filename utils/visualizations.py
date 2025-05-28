import matplotlib.pyplot as plt

def plot_image(dataloader, classes):
    image_batch, label_batch = next(iter(dataloader))
    image, label = image_batch[0], label_batch[0]

    plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
    plt.title(classes[label])
    plt.axis(False)
    plt.show() 

def accuracy_graph(results):
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def loss_graph(results):
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15,7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
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