'''
MRI Brain Tumor Classification using ViT Transformers

This script performs exploratory data analysis (EDA) on a dataset of brain MRI scans.
The dataset is used for tumor classification into four categories:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

Dataset Key Features:
- Image: MRI scan of the brain
- Label: Class of tumor (glioma, meningioma, pituitary tumor, no tumor)
- Size: 7023 images (~5700 images for training, ~1300 images for testing)
- Format: JPEG
- Dimensions: 512x512 pixels
- Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
'''

import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import get_dataframes

def graph_distribution(df, split):
    """
    Plot class distribution of MRI images.

    Args:
        df (pd.DataFrame): DataFrame containing 'label' column.
        split (str): Data split name (e.g., 'Training', 'Testing') for the title.
    """
    plt.figure(figsize=(10, 5))

    sns.countplot(data=df, x='label')

    plt.title(f'{split} Set Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()

def sample_images(df):
    """
    Display 16 sample MRI images from the training dataset for visual inspection.
    Saves the figure to utils/images/tumor.jpg.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'filepath' and 'label' columns.
    """
    plt.figure(figsize=(20, 20))

    for i in range(16):
        plt.subplot(4, 4, i + 1)

        index = i*375 # Very rough way to get a balanced set of all tumor classifications

        img_path = df['filepath'].iloc[index]
        img = plt.imread(img_path)

        plt.imshow(img)
        plt.title(df['label'].iloc[index])
        plt.axis('off')

    plt.savefig('vit/eda/images//tumor.jpg')
    plt.show()

def main():
    """
    Main function to run the EDA pipeline:
    - Load the data
    - Plot class distributions
    - Show sample MRI images
    """
    train_df, test_df = get_dataframes('data')
    graph_distribution(train_df, 'Training')
    graph_distribution(test_df, 'Testing')
    sample_images(train_df)

if __name__ == "__main__":
    main()