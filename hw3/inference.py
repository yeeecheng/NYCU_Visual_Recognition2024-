import os
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
# from utils.model import MaskRCNN
from utils.model import MaskRCNN
from utils.model_resnet101 import MaskRCNN_ResNeXt101
from pycocotools import mask as mask_utils
from torchvision.transforms import functional as TF
from torchvision import transforms

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_model(ckpt_path, device):
    model = MaskRCNN_ResNeXt101(num_classes=5)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model

def prepare_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = inference_transform(img)
    return tensor.unsqueeze(0)

def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def run_inference(model, test_dir, id_map_json, output_path, device, score_thresh=0.5, save_vis_dir="inf_vis"):
    results = []

    with open(id_map_json, "r") as f:
        raw = json.load(f)
        name_to_id = {item["file_name"]: item["id"] for item in raw}

    if save_vis_dir:
        os.makedirs(save_vis_dir, exist_ok=True)

    test_files = sorted(os.listdir(test_dir))
    for fname in tqdm(test_files):
        img_path = os.path.join(test_dir, fname)
        original_img = cv2.imread(img_path)
        image = prepare_image(img_path).to(device)

        with torch.no_grad():
            output = model(image)[0]

        image_id = name_to_id[fname]
        for i in range(len(output["scores"])):
            score = output["scores"][i].item()
            if score < score_thresh:
                continue

            label = int(output["labels"][i].item())
            mask = output["masks"][i, 0].cpu().numpy() > 0.5
            encoded_mask = encode_mask(mask)

            results.append({
                "image_id": image_id,
                "category_id": label,
                "segmentation": encoded_mask,
                "score": score
            })

            if save_vis_dir:
                vis_img = original_img.copy()
                vis_img[mask] = [0, 255, 0]  # 把預測區域塗綠
                cv2.imwrite(os.path.join(save_vis_dir, f"{fname[:-4]}_mask_{i}_score_{score:.2f}.png"), vis_img)

    with open(output_path, "w") as f:
        json.dump(results, f)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Medical Instance Segmentation")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--test-dir", type=str, default="./hw3-data-release/test_release", help="Path to test image directory")
    parser.add_argument("--id-map", type=str, default="./hw3-data-release/test_image_name_to_ids.json", help="Path to test_image_name_to_ids.json")
    parser.add_argument("--output", type=str, default="./test-results.json", help="Output json file name")
    parser.add_argument("--score-thresh", type=float, default=0.5, help="Score threshold for predictions")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    run_inference(model, args.test_dir, args.id_map, args.output, device, args.score_thresh)