import torch
from torchinfo import summary
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

#put test function here