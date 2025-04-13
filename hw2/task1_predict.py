import os
import torch
import json
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.model import FasterRCNN
from utils.dataloader import DigitDetectionDataset, collate_fn

def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

@torch.no_grad()
def predict(model, dataloader, device, score_thresh=0.8):
    model.eval()
    results = []

    for images, infos in tqdm(dataloader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, info in zip(outputs, infos):
            image_id = int(info["image_id"])  # ensure image_id is int rather than str

            boxes = output["boxes"].cpu()
            scores = output["scores"].cpu()
            labels = output["labels"].cpu()

            for box, score, label in zip(boxes, scores, labels):
                if score < score_thresh:
                    continue
                x1, y1, x2, y2 = box.tolist()
                bbox = [x1, y1, x2 - x1, y2 - y1]  # [x_min, y_min, width, height]

                results.append({
                    "image_id": image_id,
                    "bbox": bbox,
                    "score": round(score.item(), 4),
                    "category_id": label.item()
                })
    return results

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = DigitDetectionDataset(data_root_path=args.data_dir, mode="test", transforms=get_test_transform())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    num_classes = 11
    model = FasterRCNN(num_classes=num_classes, mode="test")
    checkpoint = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    detections = predict(model, test_loader, device, score_thresh=args.score_thresh)

    with open(args.output_json, "w") as f:
        json.dump(detections, f)
    print(f"Saved {len(detections)} detections to {args.output_json}")
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="pred.json")
    parser.add_argument("--score_thresh", type=float, default=0.8)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
