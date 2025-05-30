import torch
from torchinfo import summary
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

def set_seeds(seed=42):
    torch.manual_seed(seed)
    #GPU
    torch.cuda.manual_seed(seed)

def get_metrics(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'report': classification_report(y_true, y_pred), #contains f1-score, precision, and recall
        'matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics

def model_summary(vit):
    summary(model=vit,
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    return summary

def get_class_distribution(dataset, data):
    """
    Get the class distribution of the dataset.
    Args:
        dataset: The dataset from which to get the class distribution.
        data: The data object containing targets.
    Returns:
        Counter: A counter object with class labels as keys and their counts as values.
    """
    labels = []
    for idx in dataset.indices: 
        label = data.targets[idx]
        labels.append(label)
    return Counter(labels)

def load_pretrained_weights(model, pretrained_weights):
    """
    Load pretrained weights into the model.
    
    Args:
        model: The model to load weights into.        
    Returns:
        model: The model with loaded weights.
    """
    state_dict = torch.load(pretrained_weights, map_location='mps')
    converted_state_dict = rename_keys(state_dict)
    model.load_state_dict(converted_state_dict, strict=False)

    return model

def rename_keys(pretrained_dict):
    new_state_dict = {}
    for key, value in pretrained_dict.items():
        new_key = key

        # Rename rules
        new_key = new_key.replace("class_token", "embeddings.cls_token")
        new_key = new_key.replace("conv_proj.weight", "embeddings.patch_embeddings.weight")
        new_key = new_key.replace("conv_proj.bias", "embeddings.patch_embeddings.bias")
        new_key = new_key.replace("encoder.pos_embedding", "embeddings.position_embeddings")

        for i in range(12):  # number of encoder layers
            new_key = new_key.replace(f"encoder.layers.encoder_layer_{i}.ln_1.weight", f"encoder.{i}.self_attention.ln.weight")
            new_key = new_key.replace(f"encoder.layers.encoder_layer_{i}.ln_1.bias", f"encoder.{i}.self_attention.ln.bias")
            new_key = new_key.replace(f"encoder.layers.encoder_layer_{i}.self_attention.in_proj_weight", f"encoder.{i}.self_attention.multihead_attention.in_proj_weight")
            new_key = new_key.replace(f"encoder.layers.encoder_layer_{i}.self_attention.in_proj_bias", f"encoder.{i}.self_attention.multihead_attention.in_proj_bias")
            new_key = new_key.replace(f"encoder.layers.encoder_layer_{i}.self_attention.out_proj.weight", f"encoder.{i}.self_attention.multihead_attention.out_proj.weight")
            new_key = new_key.replace(f"encoder.layers.encoder_layer_{i}.self_attention.out_proj.bias", f"encoder.{i}.self_attention.multihead_attention.out_proj.bias")
            new_key = new_key.replace(f"encoder.layers.encoder_layer_{i}.ln_2.weight", f"encoder.{i}.mlp.ln.weight")
            new_key = new_key.replace(f"encoder.layers.encoder_layer_{i}.ln_2.bias", f"encoder.{i}.mlp.ln.bias")
            new_key = new_key.replace(f"encoder.layers.encoder_layer_{i}.mlp.linear_1.weight", f"encoder.{i}.mlp.linear.0.weight")
            new_key = new_key.replace(f"encoder.layers.encoder_layer_{i}.mlp.linear_1.bias", f"encoder.{i}.mlp.linear.0.bias")
            new_key = new_key.replace(f"encoder.layers.encoder_layer_{i}.mlp.linear_2.weight", f"encoder.{i}.mlp.linear.3.weight")
            new_key = new_key.replace(f"encoder.layers.encoder_layer_{i}.mlp.linear_2.bias", f"encoder.{i}.mlp.linear.3.bias")

        new_state_dict[new_key] = value
    return new_state_dict
