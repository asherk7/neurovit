import torch

def test(model, test_dataloader, device):
    """
    Evaluates the trained model on the test dataset and returns predictions and labels.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        test_dataloader (DataLoader): DataLoader containing the test dataset.
        device (torch.device): Device to run inference on (CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - y_pred_total (list): List of predicted labels for the entire test dataset.
            - y_true_total (list): List of true labels for the entire test dataset.
    """
    model.eval() # Set the model to evaluation mode (disables dropout, makes model consistent, etc.)

    correct = 0 
    y_pred_total = []
    y_true_total = [] 

    with torch.no_grad(): # Disable gradient calculations to speed up inference (saves memory)
        for batch, (X, y_true) in enumerate(test_dataloader):
            X, y_true = X.to(device), y_true.to(device)

            # Make predictions, get logits, and count correct predictions
            y_pred = model(X)
            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y_true).sum().item()

            # Move predictions and true labels to CPU (python can only handle data from CPU) and convert to lists
            y_pred_total.extend(predicted.cpu().tolist())
            y_true_total.extend(y_true.cpu().tolist())

        # Calculate overall test accuracy
        test_accuracy = correct / len(test_dataloader.dataset)

    print(f"Test Accuracy: {100 * test_accuracy:.2f}%")

    return y_pred_total, y_true_total
