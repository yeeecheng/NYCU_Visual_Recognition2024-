import os
import json
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from task1_predict import predict
from utils.evaluate import evaluate
from utils.utils import evaluate_mAP
from utils.train import train_one_epoch
from utils.model import FasterRCNN, count_parameters
from utils.dataloader import DigitDetectionDataset, collate_fn
# import torchvision.utils as vutils

# def save_transformed_images(dataset, save_dir="./src/transformed_samples", num_images=10):
#     os.makedirs(save_dir, exist_ok=True)
#     for i in range(min(num_images, len(dataset))):
#         img, label = dataset[i]  
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#         img = img * std + mean 
#         img = torch.clamp(img, 0, 1)
#         vutils.save_image(img, os.path.join(save_dir, f"sample_{i}_label_{label}.png"))

def get_train_transform():
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

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
        
        config_path = os.path.join(log_dir, f"config_{timestamp}.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)

        output_dir = os.path.join(log_dir, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

    train_dataset = DigitDetectionDataset(data_root_path=args.data_dir, mode="train", transforms=get_train_transform())
    val_dataset = DigitDetectionDataset(data_root_path=args.data_dir, mode="valid", transforms=get_val_transform())
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=args.num_workers, collate_fn=collate_fn)

    eval_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # if rank == 0:
    #     save_transformed_images(train_dataset, save_dir="transformed_samples", num_images=10)

    # ten digit + 1 backgoround
    num_classes = 11 
    model = FasterRCNN(num_classes=num_classes)
    model.to(device)

    if rank == 0:
        count_parameters(model)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_mAP = 0.0
    strat_epoch = 0
    pbar = tqdm(range(strat_epoch, args.epochs))
    for epoch in pbar:
        train_sampler.set_epoch(epoch)

        train_avg_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_avg_loss = evaluate(model, val_loader, device)

        train_loss_tensor = torch.tensor(train_avg_loss, device=device)
        val_loss_tensor = torch.tensor(val_avg_loss, device=device)
        
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        
        train_avg_loss = train_loss_tensor.item() / world_size
        val_avg_loss = val_loss_tensor.item() / world_size
        scheduler.step()

        if rank == 0:
            writer.add_scalar('Loss/train', train_avg_loss, epoch)
            writer.add_scalar('Loss/val', val_avg_loss, epoch)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
            # writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            
            detections = predict(model, eval_loader, device)
            train_pred_json = os.path.join(log_dir, "./train_pred.json")
            with open(train_pred_json, "w") as f:
                json.dump(detections, f)
            mAP_score = evaluate_mAP("./nycu-hw2-data/valid.json", train_pred_json)
            writer.add_scalar('mAP', mAP_score, epoch)

            pbar.set_description(f"Epoch [{epoch+1}/{args.epochs}], train_loss: {train_avg_loss:.4f}, val_loss: {val_avg_loss:.4f}, val mAP: {best_mAP}")
            if mAP_score > best_mAP:
                best_mAP = mAP_score
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mAP': best_mAP
                }
                torch.save(checkpoint, os.path.join(output_dir, f'best_model{epoch}.pth'))
                pbar.set_description(f"New best model saved! Epoch [{epoch + 1}/{args.epochs}], train_loss: {train_avg_loss:.4f}, val_loss: {val_avg_loss:.4f}, val mAP: {best_mAP:.4f}")
            
    if rank == 0:
        writer.close()
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="DDP Faster R-CNN for Digit Detection")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./weights")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--backend", type=str, default="nccl")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
