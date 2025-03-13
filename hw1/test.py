import os
import torch
import argparse
import pandas as pd
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet152, ResNet152_Weights
from PIL import Image
from utils.dataloader import ClassificationDataset

def predict(model, test_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for img_name, pred in zip(img_names, predicted.cpu().numpy()):
                predictions.append((os.path.basename(img_name), pred))

    return predictions

def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ClassificationDataset(root_dir=args.data_path, mode="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = resnet152(weights=None)
    num_classes = 100
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    predictions = predict(model, test_loader, device)

    df = pd.DataFrame(predictions, columns=["image_name", "pred_label"])
    df.to_csv(args.output_csv, index=False)
    print(f"Successfully save to {args.output_csv}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default= "/mnt/HDD7/yicheng/visual_recognition/hw1/weights/best_model98.pth")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--output_csv", type=str, default="./prediction.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test(args)
    
    