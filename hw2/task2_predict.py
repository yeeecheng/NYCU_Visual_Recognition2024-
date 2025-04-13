import json
import pandas as pd
import argparse
from collections import defaultdict
import os

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

def process_predictions(predictions, all_image_ids):
    image_results = defaultdict(list)

    for pred in predictions:
        image_id = int(pred["image_id"])
        bbox = pred["bbox"]
        x_min = bbox[0]
        category_id = pred["category_id"]
        image_results[image_id].append((x_min, category_id))

    final_results = []
    for image_id in sorted(all_image_ids):
        preds = image_results.get(image_id, [])
        if not preds:
            final_results.append({"image_id": image_id, "pred_label": -1})
        else:
            preds_sorted = sorted(preds, key=lambda x: x[0])
            digit_str = ''.join(str(c - 1) for _, c in preds_sorted)
            final_results.append({"image_id": image_id, "pred_label": digit_str})
    return final_results

def save_to_csv(results, output_csv_path):
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Task 2 CSV saved to {output_csv_path}")

def main(args):
    predictions = load_predictions(args.pred_json)
    all_image_ids = get_all_image_ids(predictions, args.image_dir)
    results = process_predictions(predictions, all_image_ids)
    save_to_csv(results, args.output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_json", type=str, required=True, help="Path to Task 1 prediction JSON file")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to test images folder (to recover missing image_ids)")
    parser.add_argument("--output_csv", type=str, default="pred.csv", help="Path to save Task 2 CSV output")
    args = parser.parse_args()
    main(args)
