import torch
from pipeline.val import validate_step

def train_step(model, train_dataloader, optimizer, loss_fn, device):
    """
    Perform one training epoch.

    Args:
        model (torch.nn.Module): The model being trained.
        train_dataloader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): Optimizer for updating model weights.
        loss_fn (Loss): Loss function.
        device (torch.device): Device to run computations on.

    Returns:
        tuple: Average training loss and accuracy for the epoch.
    """

    model.train() # Set model to training mode (enables dropout, batch normalization, etc.)

    running_loss = 0 # Sum of losses across all batches for the epoch
    correct = 0

    for batch, (X, y_true) in enumerate(train_dataloader):
        X, y_true = X.to(device), y_true.to(device)

        optimizer.zero_grad() # Reset gradients

        # Getting predictions, computing loss, backpropagating, and updating weights
        y_pred = model(X)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

        # Accumulating results
        running_loss += loss.item() * X.size(0)
        _, predicted = torch.max(y_pred, 1)
        correct += (predicted==y_true).sum().item()

    train_loss = running_loss / len(train_dataloader.dataset)
    train_acc = correct / len(train_dataloader.dataset)

    return train_loss, train_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):
    """
    Train the model over multiple epochs with validation and early stopping.

    Args:
        model (torch.nn.Module): Model to be trained.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): Optimizer for training.
        scheduler (LambdaLR): Learning rate scheduler.
        loss_fn (Loss): Loss function.
        epochs (int): Number of epochs to train.
        device (torch.device): Device to run computations on (CPU/GPU/MPS).

    Returns:
        dict: Training and validation loss/accuracy per epoch.
    """
    
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []}
    
    # Initialize variables for early stopping
    best_val_loss = float('inf') 
    failed_epochs = 0
    early_stop = 3
    
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model, 
                                           train_dataloader=train_dataloader,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn,
                                           device=device)
        val_loss, val_acc = validate_step(model=model,
                                          val_dataloader=val_dataloader,
                                          loss_fn=loss_fn,
                                          device=device)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Validation loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)

        # If model isn't improving, trigger early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            failed_epochs = 0
        else:
            failed_epochs += 1
        
        if failed_epochs >= early_stop:
            print("Early stopping triggered due to complacent model learning")
            break

        # Step the scheduler to adjust learning rate
        scheduler.step()

        # Clear cache to free up memory
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return results