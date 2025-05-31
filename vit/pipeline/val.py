import torch

def validate_step(model, val_dataloader, loss_fn, device):
    """
    Performs a validation step for the given model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_dataloader (DataLoader): DataLoader for the validation data.
        loss_fn (torch.nn.Module): Loss function to evaluate the predictions.
        device (torch.device): Device to perform computations on (CPU or GPU, MPS for apple).

    Returns:
        tuple: A tuple containing:
            - val_loss (float): Average validation loss across all batches.
            - val_acc (float): Validation accuracy across all batches.
    """
    model.eval() # Set the model to evaluation mode (disables dropout, makes model consistent, etc.)

    running_loss = 0 # Sum of losses across all batches for the epoch
    correct = 0 

    with torch.no_grad(): # Disable gradient calculation for faster validation, helps save memory
        for batch, (X, y_true) in enumerate(val_dataloader):
            X, y_true = X.to(device), y_true.to(device)

            # Getting predictions, computing loss, and accumulating results
            y_pred = model(X)

            loss = loss_fn(y_pred, y_true)
            running_loss += loss.item() * X.size(0)

            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y_true).sum().item()

    # Calculate average loss and accuracy
    val_loss = running_loss / len(val_dataloader.dataset)
    val_acc = correct / len(val_dataloader.dataset)

    return val_loss, val_acc
