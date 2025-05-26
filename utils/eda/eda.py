'''
MRI Brain Tumor Classification using ViT Transformers

As technology advances, the use of MRI scans for diagnosing brain tumors has become increasingly common. 
Due to the advancements in both imaging technology and machine learning, we can now analyze MRI scans with greater accuracy and speed than ever before for diagnosing medical conditions. 

This file will focus on exploring a dataset of MRI scans that contain images of brain tumors. 
These images are split into four categories: glioma, meningioma, pituitary tumor, and no tumor. 
We will analyze data distributions, class imbalances, visualizing our data, then cleaning the data. 

Dataset Overview

Key Features:
- Image: MRI scan of the brain
- Label: Class of tumor (glioma, meningioma, pituitary tumor, no tumor)
- Size: 7023 images (~5700 images for training, ~1300 images for testing)
- Format: JPEG
- Dimensions: 512x512 pixels
- Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_loader import get_dataframes

def graph_distribution(df, split):
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='label')
    plt.title(f'{split} Set Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()

def sample_images(df):
    plt.figure(figsize=(20, 20))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        index = i*375 # very rough way to get a balanced set of all tumor classifications
        img_path = df['filepath'].iloc[index]
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(df['label'].iloc[index])
        plt.axis('off')

    plt.savefig('utils/images/tumor.jpg')
    plt.show()

def main():
    train_df, test_df = get_dataframes('data')
    graph_distribution(train_df, 'Training')
    graph_distribution(test_df, 'Testing')
    sample_images(train_df)

main()