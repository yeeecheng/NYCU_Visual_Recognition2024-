import os
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from utils.model import Resnet152, count_parameters
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from utils.dataloader import ClassificationDataset
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score

def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')  # normalize per row
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def predict_for_val(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

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
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mode = "val"

    test_dataset = ClassificationDataset(root_dir=args.data_path, mode=mode, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    num_classes = 100
    model = Resnet152(num_classes=num_classes, mode="test")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    count_parameters(model)
    if mode == "test":
        
        predictions = predict(model, test_loader, device)

        df = pd.DataFrame(predictions, columns=["image_name", "pred_label"])
        df.to_csv(args.output_csv, index=False)
        print(f"Successfully save to {args.output_csv}")

    else:
        all_labels, all_preds = predict_for_val(model, test_loader, device)
        class_names = [str(i) for i in range(100)]
      
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score (macro): {f1:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"Precision (macro): {precision:.4f}")
        
        print("\nPer-class metrics:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        # 繪製 confusion matrix
        plot_confusion_matrix(all_labels, all_preds, class_names, save_path="confusion_matrix.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default= "*.pth")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--output_csv", type=str, default="./prediction.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test(args)
    
    