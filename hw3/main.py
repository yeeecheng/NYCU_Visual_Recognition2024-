import os
import time
import torch
import argparse
import torchvision
import albumentations as A
import torch.distributed as dist
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from dataset import MedicalInstanceDataset, collate_fn
from dataset_tile import MedicalInstanceDatasetTile, collate_fn_tile
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.model import MaskRCNN
from utils.model_resnet101 import MaskRCNN_ResNeXt101
from utils.train import train_one_epoch


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, p=0.5, border_mode=0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

def train(args):
    dist.init_process_group(backend=args.backend, init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=output_dir)

    train_dataset = MedicalInstanceDataset(root_dir=args.data_path, transforms=get_train_transform())
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)

    model = MaskRCNN(num_classes= (4 + 1)).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f}M")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                                momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = args.epochs

    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}]") if rank == 0 else train_loader
        train_avg_loss = train_one_epoch(model, optimizer, train_loader, device, pbar, rank)

        train_loss_tensor = torch.tensor(train_avg_loss, device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_avg_loss = train_loss_tensor.item() / world_size

        if rank == 0:
            pbar.set_postfix(loss=train_avg_loss)
        torch.cuda.empty_cache()
        # lr_scheduler.step()

        if rank == 0:
            print(f"[Epoch {epoch+1}] Total Loss: {train_avg_loss:.4f}")
            writer.add_scalar("Loss/train", train_avg_loss, epoch + 1)
            writer.flush()

            if train_avg_loss < best_loss:
                best_loss = train_avg_loss
                ckpt = {
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss,
                }
                torch.save(ckpt, os.path.join(output_dir, f"maskrcnn_medical_epoch{epoch+1}.pth"))
                print(f"[Epoch {epoch+1}] New best model saved with loss {best_loss:.4f}")

    if rank == 0:
        writer.close()
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Instance Segmentation Training")
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size per GPU')
    parser.add_argument('--num-classes', type=int, default=5, help='number of classes including background')
    parser.add_argument('--data-path', type=str, default='./hw3-data-release/train', help='path to training data root')
    parser.add_argument('--output-dir', type=str, default='./logs', help='directory to save checkpoints')
    parser.add_argument("--backend", type=str, default="nccl")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)

