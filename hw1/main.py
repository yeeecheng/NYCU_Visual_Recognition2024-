import os
import torch
import argparse
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import resnet152, ResNet152_Weights
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.train import train_one_epoch
from utils.evaluate import evaluate
from utils.dataloader import ClassificationDataset

def main(args):

    dist.init_process_group(backend=args.backend, init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        writer = SummaryWriter(log_dir="./logs")

    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.CenterCrop(size=((256, 256))),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20, center=(0, 0)),
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    train_dataset = ClassificationDataset(root_dir=args.data_path, mode="train", transform=train_transform)
    val_dataset = ClassificationDataset(root_dir=args.data_path, mode="val", transform=val_transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers)

    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    best_val_acc = 0.0
    strat_epoch = 0
    pbar = tqdm(range(strat_epoch, args.epochs))
    for epoch in pbar:
        train_sampler.set_epoch(epoch)
        train_avg_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_avg_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend: nccl / gloo / mpi")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    main(args)