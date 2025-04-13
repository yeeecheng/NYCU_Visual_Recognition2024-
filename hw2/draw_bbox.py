import json
import pandas as pd
import argparse
from collections import defaultdict
import os
import cv2

def load_predictions(pred_json_path):
    with open(pred_json_path, 'r') as f:
        predictions = json.load(f)
    return predictions

def get_all_image_ids(predictions, image_dir):
    ids_from_preds = {int(pred["image_id"]) for pred in predictions}

    if image_dir:
        image_ids_from_dir = set()
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image_id = int(os.path.splitext(fname)[0])
                    image_ids_from_dir.add(image_id)
                except:
                    continue
        return image_ids_from_dir.union(ids_from_preds)
    else:
        return ids_from_preds

def draw_predictions_on_image(image_path, predictions, save_path):
    image = cv2.imread(image_path)
    for pred in predictions:
        bbox = pred['bbox']
        category = pred['category_id'] - 1  # 0-based
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, str(category), (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imwrite(save_path, image)

def process_predictions(predictions, all_image_ids, image_dir=None, vis_dir=None):
    image_results = defaultdict(list)

    for pred in predictions:
        image_id = int(pred["image_id"])
        bbox = pred["bbox"]
        x_min = bbox[0]
        category_id = pred["category_id"]
        image_results[image_id].append((x_min, category_id, bbox))

    final_results = []
    for image_id in sorted(all_image_ids):
        preds = image_results.get(image_id, [])

        if not preds:
            final_results.append({"image_id": image_id, "pred_label": -1})
        else:
            preds_sorted = sorted(preds, key=lambda x: x[0])
            digit_str = ''.join(str(c - 1) for _, c, _ in preds_sorted)
            final_results.append({"image_id": image_id, "pred_label": digit_str})

        # Draw predictions if vis_dir is provided
        if vis_dir and image_dir:
            image_name = f"{image_id}.jpg"
            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):  # try .png
                image_path = os.path.join(image_dir, f"{image_id}.png")
            if os.path.exists(image_path):
                vis_preds = [{'bbox': bbox, 'category_id': c} for _, c, bbox in preds]
                save_path = os.path.join(vis_dir, os.path.basename(image_path))
                draw_predictions_on_image(image_path, vis_preds, save_path)

    return final_results

def save_to_csv(results, output_csv_path):
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Task 2 CSV saved to {output_csv_path}")

def main(args):
    os.makedirs(args.vis_dir, exist_ok=True) if args.vis_dir else None
    predictions = load_predictions(args.pred_json)
    all_image_ids = get_all_image_ids(predictions, args.image_dir)
    results = process_predictions(predictions, all_image_ids, args.image_dir, args.vis_dir)
    save_to_csv(results, args.output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_json", type=str, default="./pred.json", help="Path to Task 1 prediction JSON file")
    parser.add_argument("--image_dir", type=str, default= "./nycu-hw2-data/test", help="Path to test images folder")
    parser.add_argument("--output_csv", type=str, default="pred.csv", help="Path to save Task 2 CSV output")
    parser.add_argument("--vis_dir", type=str, default="./bbox", help="Directory to save visualized images (optional)")
    args = parser.parse_args()
    main(args)
