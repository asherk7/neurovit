import torch
from torchinfo import summary
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

def set_seeds(seed=42):
    """
    Set seed for reproducibility across CPU and GPU.
    """
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)

def get_metrics(y_true, y_pred):
    """
    Compute and print classification metrics.

    Args:
        y_true (list or array): Ground truth labels.
        y_pred (list or array): Predicted labels.

    Returns:
        dict: Contains accuracy and full classification report.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'report': classification_report(y_true, y_pred), # Contains f1-score, precision, and recall
    }

    #print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(metrics['report'])

    return metrics

def model_summary(vit):
    """
    Print a summary of the model using torchinfo.

    Args:
        model (nn.Module): The model to summarize.

    Returns:
        summary object: Torchinfo summary.
    """
    summary(model=vit,
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    return summary

def get_class_distribution(dataset, data):
    """
    Get the distribution of classes in a subset of a dataset.

    Args:
        dataset: Subset (like from random_split).
        data: Full dataset (e.g., ImageFolder) to access `.targets`.

    Returns:
        Counter: Class label counts.
    """
    labels = [data.targets[idx] for idx in dataset.indices]
    return Counter(labels)

def load_pretrained_weights(model, pretrained_weights):
    """
    Load pre-trained weights into the model with key remapping.

    Args:
        model (nn.Module): Model to load weights into.
        pretrained_weights (str): Path to the .pt or .pth weight file.

    Returns:
        nn.Module: Model with loaded weights.
    """
    state_dict = torch.load(pretrained_weights, map_location='mps', weights_only=True)
    converted_state_dict = rename_keys(state_dict)

    # Changing the pretrained head weights to match the custom model's head, since the custom model has a different head structure
    """keys_to_update = [k for k in converted_state_dict.keys() if 'mlp_head.linear' in k]
    for k in keys_to_update:
        converted_state_dict[k] = model.state_dict()[k]"""

    model.load_state_dict(converted_state_dict, strict=True)

    return model

def rename_keys(pretrained_dict):
    """
    Rename keys in pretrained state dict to match custom model's keys.

    Args:
        pretrained_dict (dict): Original pretrained state dict.

    Returns:
        dict: Converted state dict with updated key names.
    """
    new_state_dict = {}
    for key, value in pretrained_dict.items():
        new_key = key

        # Embedded / Head layers
        new_key = new_key.replace("class_token", "embeddings.cls_token")
        new_key = new_key.replace("conv_proj.weight", "embeddings.patch_embeddings.weight")
        new_key = new_key.replace("conv_proj.bias", "embeddings.patch_embeddings.bias")
        new_key = new_key.replace("encoder.pos_embedding", "embeddings.position_embeddings")
        new_key = new_key.replace("encoder.ln.weight", "mlp_head.norm.weight")
        new_key = new_key.replace("encoder.ln.bias", "mlp_head.norm.bias")
        new_key = new_key.replace("heads.head.weight", "mlp_head.linear.weight")
        new_key = new_key.replace("heads.head.bias", "mlp_head.linear.bias")

        # Encoder layers
        for i in range(12):
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
