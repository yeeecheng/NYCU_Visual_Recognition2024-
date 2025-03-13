import torch

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(out, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    acc = 100 * correct / total
    
    return avg_loss, acc