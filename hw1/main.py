import os
import json
import time
import torch
import argparse
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.model import Resnet152, count_parameters
from utils.evaluate import evaluate
from utils.train import train_one_epoch
from utils.dataloader import ClassificationDataset


import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms.functional as F

def save_transformed_images(dataset, save_dir="./src/transformed_samples", num_images=10):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_images, len(dataset))):
        img, label = dataset[i]  
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean 
        img = torch.clamp(img, 0, 1)
        vutils.save_image(img, os.path.join(save_dir, f"sample_{i}_label_{label}.png"))

def main(args):

    dist.init_process_group(backend=args.backend, init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("logs", f"run_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        config_path = os.path.join(log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)

    train_transform = transforms.Compose([
        transforms.Resize((560, 560)),
        transforms.CenterCrop(size=((512, 512))),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomRotation(degrees=20, center=(0, 0)),
        transforms.ToTensor(),          
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),  
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    train_dataset = ClassificationDataset(root_dir=args.data_path, mode="train", transform=train_transform)
    val_dataset = ClassificationDataset(root_dir=args.data_path, mode="val", transform=val_transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers)

    if rank == 0:
        print("Saving transformed images for inspection...")
        save_transformed_images(train_dataset, save_dir="transformed_samples", num_images=10)

    num_classes = len(train_dataset.classes)
    model = Resnet152(num_classes=num_classes)
    model.to(device)
    count_parameters(model)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    best_val_acc = 0.0
    strat_epoch = 0
    pbar = tqdm(range(strat_epoch, args.epochs))
    for epoch in pbar:
        train_sampler.set_epoch(epoch)
        train_avg_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_avg_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_loss_tensor = torch.tensor(train_avg_loss, device=device)
        train_acc_tensor = torch.tensor(train_acc, device=device)
        val_loss_tensor = torch.tensor(val_avg_loss, device=device)
        val_acc_tensor = torch.tensor(val_acc, device=device)
        
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
        
        train_avg_loss = train_loss_tensor.item() / world_size
        train_acc = train_acc_tensor.item() / world_size
        val_avg_loss = val_loss_tensor.item() / world_size
        val_acc = val_acc_tensor.item() / world_size
        scheduler.step()
        
        if rank == 0:
            pbar.set_description(f"Epoch [{epoch+1}/{args.epochs}], train_loss: {train_avg_loss:.4f}, train_acc: {train_acc:.2f}%, val_loss: {val_avg_loss:.4f}, val_acc: {val_acc:.2f}%")
            writer.add_scalar('Loss/train', train_avg_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/val', val_avg_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc
                }
                torch.save(checkpoint, os.path.join("./weights", f'best_model{epoch}.pth'))
                pbar.set_description(f"New best model saved! Epoch [{epoch+1}/{args.epochs}], train_loss: {train_avg_loss:.4f}, train_acc: {train_acc:.2f}%, val_loss: {val_avg_loss:.4f}, val_acc: {val_acc:.2f}%")
    
    if rank == 0:
        writer.close()
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend: nccl / gloo / mpi")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    main(args)