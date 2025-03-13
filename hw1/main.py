import os
import glob
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.distributed as dist

from torch.utils.data import DataLoader
from torchvision.models import resnet152, ResNet152_Weights
from tqdm import tqdm
from train import train_one_epoch
from evaluate import evaluate
from dataloader import ClassificationDataset

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(size=((200, 200))),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, center=(0, 0)),
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((200, 200)),  
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])    
    ])

    train_dataset = ClassificationDataset(root_dir=args.data_path, mode="train", transform=train_transform)
    val_dataset = ClassificationDataset(root_dir=args.data_path, mode="val", transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    
    strat_epoch = 0
    pbar = tqdm(range(strat_epoch, args.epochs))
    for epoch in pbar:
        train_avg_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_avg_loss, val_acc = evaluate(model, val_loader, criterion, device)
        pbar.set_description(f"Epoch [{epoch+1}/{args.epochs}], train_loss: {train_avg_loss:.4f}, train_acc: {train_acc:.2f}%,val_loss: {val_avg_loss:.4f}, val_acc: {val_acc:.2f}%")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend: nccl / gloo / mpi")
    # parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Total number of GPUs")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    main(args)