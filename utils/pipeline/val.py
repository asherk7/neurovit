import torch

def validate_step(model, val_dataloader, loss_fn, device):
    model.eval()

    running_loss, last_loss = 0, 0

    with torch.no_grad():
        for batch, (X, y_true) in enumerate(val_dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y_true)

            running_loss += loss.item() * X.size(0)
            _, preds = torch.max(y_pred, 1)
            correct += (preds==y_true).sum().item()

        val_loss = running_loss / len(val_dataloader)
        val_acc = correct / len(val_dataloader)
        return val_loss, val_acc
