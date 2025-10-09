import fiftyone as fo
import fiftyone.utils.coco as fouc
from pycocotools.coco import COCO
import os
import json
import argparse
import tempfile


def wrap_predictions_to_coco(predictions, coco_gt):
    wrapped = {
        "images": [],
        "annotations": [],
        "categories": coco_gt.dataset.get("categories", [])
    }

    image_ids = set()
    for pred in predictions:
        image_id = pred["image_id"]
        if image_id not in image_ids:
            image_ids.add(image_id)
            img_meta = coco_gt.imgs[image_id]
            wrapped["images"].append({
                "id": image_id,
                "file_name": img_meta["file_name"],
                "width": img_meta["width"],
                "height": img_meta["height"]
            })

        wrapped["annotations"].append({
            "id": len(wrapped["annotations"]),
            "image_id": image_id,
            "category_id": pred["category_id"],
            "bbox": pred["bbox"],
            "score": pred.get("score", 1.0),
            "area": pred["bbox"][2] * pred["bbox"][3],
            "iscrowd": 0
        })

    return wrapped


def main():
    parser = argparse.ArgumentParser(description="Visualize COCO ground truth and predictions with FiftyOne")
    parser.add_argument("--coco_json", type=str, required=True, help="Path to COCO ground truth JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--predictions_json", type=str, help="Path to predictions JSON file")
    args = parser.parse_args()

    # Load ground truth
    coco_gt = COCO(args.coco_json)

    # Create FiftyOne dataset
    dataset = fo.Dataset(name="efficientdet_voxel51_view")
    fouc.add_coco_labels(dataset, args.coco_json, label_field="ground_truth", images_dir=args.image_dir)

    # Load and wrap predictions
    if args.predictions_json and os.path.exists(args.predictions_json):
        with open(args.predictions_json, "r") as f:
            raw_preds = json.load(f)

        # Handle nested format (e.g., list of epochs)
        if isinstance(raw_preds, list) and isinstance(raw_preds[0], dict) and "predictions" in raw_preds[0]:
            raw_preds = raw_preds[-1]["predictions"]

        wrapped_preds = wrap_predictions_to_coco(raw_preds, coco_gt)

        # Save wrapped predictions to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(wrapped_preds, tmp)
            pred_path = tmp.name

        fouc.add_coco_labels(dataset, pred_path, label_field="predictions", images_dir=args.image_dir)

    # Launch FiftyOne app
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()