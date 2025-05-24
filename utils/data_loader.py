import pandas as pd
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

def get_dataframes(root_dir):
    df = create_image_dataframe(root_dir)

    train_df = df[df['split'] == 'Training'].copy()
    train_df.drop(columns=['split'], inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    test_df = df[df['split'] == 'Testing'].copy()
    test_df.drop(columns=['split'], inplace=True)

    dataframes = {
        'train': train_df,
        'test': test_df
    }
    
    return dataframes
