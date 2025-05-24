from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_metrics(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'report': classification_report(y_true, y_pred), #contains f1-score, precision, and recall
        'matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics
