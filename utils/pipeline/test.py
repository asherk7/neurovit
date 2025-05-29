import torch

def test(model, test_dataloader, device):
    model.eval()

    correct = 0
    y_pred_total = []
    y_true_total = []

    with torch.no_grad():
        for batch, (X, y_true) in enumerate(test_dataloader):
            X, y_true = X.to(device), y_true.to(device)

            y_pred = model(X)

            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y_true).sum().item()

            y_pred_total.extend(predicted.cpu().tolist())
            y_true_total.extend(y_true.cpu().tolist())

        test_accuracy = correct / len(test_dataloader.dataset)
            
    print(f"Test Accuracy: {100 * test_accuracy:.2f}%")

    return y_pred_total, y_true_total
