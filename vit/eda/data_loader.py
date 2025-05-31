"""
Dataframe creation utility for image dataset.

Expected directory structure:
root_dir/
    ├── Training/
    │   ├── Class1/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── Class2/
    └── Testing/
        ├── Class1/
        └── Class2/
"""
import pandas as pd
import os

def create_image_dataframe(root_dir):
    """
    Creates a dataframe of all image file paths and their corresponding labels and split type.

    Args:
        root_dir (str): Path to the root directory containing 'Training' and 'Testing' subfolders.

    Returns:
        pd.DataFrame: DataFrame with columns ['filepath', 'label', 'split']
    """
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

def get_dataframes(root_dir):
    """
    Separates the complete dataframe into training and testing sets.

    Args:
        root_dir (str): Path to the dataset root directory.

    Returns:
        tuple: (train_df, test_df)
            - train_df: DataFrame with columns ['filepath', 'label']
            - test_df: DataFrame with columns ['filepath', 'label']
    """
    df = create_image_dataframe(root_dir)

    if df.empty:
        raise ValueError("No images found in the specified directory structure.")
    if 'split' not in df.columns:
        raise ValueError("DataFrame does not contain 'split' column. Ensure the directory structure is correct.")
    
    # Splitting the dataframe into training and testing sets, since it's combined right now
    train_df = df[df['split'] == 'Training'].copy()
    train_df.drop(columns=['split'], inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    print(train_df.head())

    test_df = df[df['split'] == 'Testing'].copy()
    test_df.drop(columns=['split'], inplace=True)
    print(test_df.head())

    return (train_df, test_df)
