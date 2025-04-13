import torch

def evaluate(model, val_loader, device):

    model.train()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            total_loss += losses.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss