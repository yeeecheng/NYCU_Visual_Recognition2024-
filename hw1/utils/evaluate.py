import torch

def evaluate(model, val_loader, criterion, device):

    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    for batch in val_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        out = model(images)
        loss = criterion(out, labels)
        total_loss += loss.item()
        _, predicted = torch.max(out, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    acc = 100 * correct / total
    
    return avg_loss, acc