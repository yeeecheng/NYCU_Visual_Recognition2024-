from tqdm import tqdm

def train_one_epoch(model, optimizer, train_loader, device, pbar, rank):

    model.train()
    total_loss = 0.0

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if len(images) == 0:
            continue
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if rank == 0:
            pbar.set_postfix(loss=losses.item())

    avg_loss = total_loss / len(train_loader)
    return avg_loss