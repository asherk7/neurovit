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

def create_image_dataframe(root_dir):
    data = []
    for split in ['Testing', 'Training']:
        split_path = os.path.join(root_dir, split)
        for label in os.listdir(split_path):
            label_path = os.path.join(split_path, label)
            if os.path.isdir(label_path):
                for fname in os.listdir(label_path):
                    if fname.lower().endswith(('.jpg', '.jpeg')):
                        full_path = os.path.join(label_path, fname)
                        data.append({
                            'filepath': full_path,
                            'label': label,
                            'split': split
                        })
    return pd.DataFrame(data)

def create_dataframes():
    df = create_image_dataframe('data')

    train_df = df[df['split'] == 'Training'].copy()
    train_df.drop(columns=['split'], inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    print(train_df.head())

    test_df = df[df['split'] == 'Testing'].copy()
    test_df.drop(columns=['split'], inplace=True)
    print(test_df.head())

    return (train_df, test_df)

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
        index = i*375
        img_path = df['filepath'].iloc[index]
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(df['label'].iloc[index])
        plt.axis('off')
        
    plt.show()

def main():
    train_df, test_df = create_dataframes()
    graph_distribution(train_df, 'Training')
    graph_distribution(test_df, 'Testing')
    sample_images(train_df)

main()