import fiftyone as fo
import fiftyone.utils.coco as fouc
from pycocotools.coco import COCO
import os
import json
import argparse
import tempfile


# ... (wrap_annotations_to_coco and wrap_predictions_to_coco functions remain the same) ...
def wrap_annotations_to_coco(raw_data, image_dir):
    image_map = {}
    categories = {}
    annotations = []

    for i, ann in enumerate(raw_data):
        raw_image_key = ann["image"]
        
        # COCO 'file_name' must contain the full relative path (e.g., 'samples/CAM_FRONT/...')
        file_name = raw_image_key 
        
        width = ann.get("width", 1600)
        height = ann.get("height", 900)
        category_id = ann["category_id"]

        if raw_image_key not in image_map:
            image_id = len(image_map) + 1 
            image_map[raw_image_key] = {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height
            }
        else:
            image_id = image_map[raw_image_key]["id"]

        if category_id not in categories:
            categories[category_id] = {
                "id": category_id,
                "name": f"class_{category_id}",
                "supercategory": "object"
            }

        annotations.append({
            "id": len(annotations) + 1,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": ann["bbox"],
            "area": ann["bbox"][2] * ann["bbox"][3],
            "iscrowd": 0
        })

    return {
        "images": list(image_map.values()),
        "annotations": annotations,
        "categories": list(categories.values())
    }


def wrap_predictions_to_coco(predictions, coco_gt):
    wrapped = {
        "images": [],
        "annotations": [],
        "categories": coco_gt.dataset.get("categories", [])
    }

    image_ids = set()
    for i, pred in enumerate(predictions):
        image_id = pred["image_id"] 
        
        if image_id not in image_ids:
            image_ids.add(image_id)
            img_meta = coco_gt.imgs[image_id]
            
            file_name = img_meta["file_name"]

            wrapped["images"].append({
                "id": image_id,
                "file_name": file_name,
                "width": img_meta["width"],
                "height": img_meta["height"]
            })

        wrapped["annotations"].append({
            "id": i + 1, 
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
    
    gt_path = args.coco_json
    pred_path = None
    temp_gt_file = None
    
    # 💥 FINAL PATH FIX: Normalize and remove trailing slash.
    # This prevents the eta library from trying to load the directory as a JSON file.
    image_dir_abs_norm = os.path.abspath(args.image_dir)
    image_dir_abs_norm = os.path.normpath(image_dir_abs_norm).replace('\\', '/')
    
    # 1. Load ground truth and handle list-based format
    with open(args.coco_json, "r") as f:
        raw_gt = json.load(f)

    if isinstance(raw_gt, list):
        print("Detected list-based annotation file. Wrapping to COCO format...")
        wrapped_gt = wrap_annotations_to_coco(raw_gt, args.image_dir)
        
        temp_gt_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(wrapped_gt, temp_gt_file)
        temp_gt_file.close()
        gt_path = temp_gt_file.name

    try:
        # 2. Load COCO object
        coco_gt = COCO(gt_path)

        # --- DIAGNOSTIC CHECK ---
        first_image_filename = coco_gt.imgs[1]["file_name"] if 1 in coco_gt.imgs else "N/A"
        expected_path = os.path.join(os.path.abspath(args.image_dir), first_image_filename)

        print("-" * 50)
        print("Path Diagnostic")
        print(f"Image Directory (Final Normalized): {image_dir_abs_norm}")
        print(f"COCO file_name (1st): {first_image_filename}")
        print(f"Image exists? {os.path.exists(expected_path)}")
        print("-" * 50)
        
        if not os.path.exists(expected_path):
             raise FileNotFoundError(f"Image file not found at expected location. Expected file: {expected_path}")
        # ------------------------

        # 3. Create FiftyOne dataset
        dataset_name = "efficientdet_voxel51_view"
        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)
        dataset = fo.Dataset(name=dataset_name)

        # 4. Add Ground Truth labels
        # Use the strictly normalized path
        fouc.add_coco_labels(
            dataset, 
            gt_path, 
            image_dir_abs_norm, 
            "ground_truth"
        )

        # 5. Load and wrap predictions
        if args.predictions_json and os.path.exists(args.predictions_json):
            with open(args.predictions_json, "r") as f:
                raw_preds = json.load(f)

            if isinstance(raw_preds, list) and isinstance(raw_preds[0], dict) and "predictions" in raw_preds[0]:
                raw_preds = raw_preds[-1]["predictions"]

            wrapped_preds = wrap_predictions_to_coco(raw_preds, coco_gt)

            temp_pred_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(wrapped_preds, temp_pred_file)
            temp_pred_file.close()
            pred_path = temp_pred_file.name

            # 6. Add Prediction labels
            # Use the strictly normalized path again
            fouc.add_coco_labels(
                dataset, 
                pred_path, 
                image_dir_abs_norm, 
                "predictions"
            )

        # 7. Launch FiftyOne app
        print("Launching FiftyOne App...")
        session = fo.launch_app(dataset)
        session.wait()

    finally:
        # Clean up temporary files
        if temp_gt_file is not None and os.path.exists(gt_path) and gt_path != args.coco_json:
            os.remove(gt_path)
        if pred_path is not None and os.path.exists(pred_path):
            os.remove(pred_path)


if __name__ == "__main__":
    main()