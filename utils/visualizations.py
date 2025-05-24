import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import get_dataframes

def brain_plot(root_dir):
    df = get_dataframes(root_dir)
    train_df = df['train']

    plt.figure(figsize=(20, 20))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        index = i*375
        img_path = train_df['filepath'].iloc[index]
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(train_df['label'].iloc[index])
        plt.axis('off')
    plt.savefig('utils/images/tumor.jpg')

def accuracy_graph():
    pass

def loss_graph():
    pass

def epoch_history():
    pass

def get_matrix():
    pass

def main():
    brain_plot('data')
    accuracy_graph()
    loss_graph()
    epoch_history()
    get_matrix()

if __name__ == '__main__':
    main()
