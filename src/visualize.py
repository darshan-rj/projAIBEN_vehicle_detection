import fiftyone as fo
import fiftyone.utils.coco as fouc
from fiftyone import types
from pycocotools.coco import COCO
import os
import json
import argparse
import tempfile


def wrap_annotations_to_coco(raw_data):
    """
    Wraps a list of raw annotations (e.g., from a custom format) into a COCO
    ground truth dictionary structure.
    """
    image_map = {}
    categories = {}
    annotations = []

    for i, ann in enumerate(raw_data):
        raw_image_key = ann["image"]
        file_name = raw_image_key
        # Use default width/height if not provided
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
                # In a real scenario, you'd look up the class name here
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
    """
    Wraps a list of raw predictions into a COCO predictions dictionary structure,
    matching image IDs to the ground truth (coco_gt) structure.
    
    Corrects the '__CAMM_FRONT__' typo found in the prediction file 
    and matches against the bare filename from the COCO GT structure.
    """
    wrapped = {
        "images": [],
        "annotations": [],
        "categories": coco_gt.dataset.get("categories", []) 
    }
    
    # Create a map of bare COCO filename -> COCO image metadata for efficient lookup
    coco_image_map = {}
    for img_id, img_meta in coco_gt.imgs.items():
        # Use the bare filename for matching, as predictions lack the path prefix
        base_name = os.path.basename(img_meta["file_name"])
        coco_image_map[base_name] = img_meta 

    image_ids = set()
    unresolved_count = 0
    total_predictions = len(predictions)
    
    for i, pred in enumerate(predictions):
        image_id = pred.get("image_id") 
        raw_file_name = pred.get("image")
        
        if image_id is None and raw_file_name:
            # 1. Correct the known typo: __CAMM_FRONT__ -> __CAM_FRONT__
            corrected_file_name = raw_file_name.replace("__CAMM_FRONT__", "__CAM_FRONT__")
            
            # 2. Look up the image ID using the corrected filename
            if corrected_file_name in coco_image_map:
                img_meta = coco_image_map[corrected_file_name]
                image_id = img_meta["id"]

        if image_id is None:
            unresolved_count += 1
            continue

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
            "id": len(wrapped["annotations"]) + 1,
            "image_id": image_id,
            "category_id": pred["category_id"],
            "bbox": pred["bbox"],
            "score": pred.get("score", 1.0),
            "area": pred["bbox"][2] * pred["bbox"][3],
            "iscrowd": 0
        })
    
    # Print data mismatch warnings
    if unresolved_count > 0:
        print(f"⚠️ Warning: Skipped {unresolved_count} predictions out of {total_predictions} due to unresolved image references.")
        if unresolved_count == total_predictions:
             print("🛑 Critical: All predictions were skipped. Check if prediction image names match ground truth image names.")

    return wrapped


def validate_json_file(path, label):
    """Validates that a file path points to a non-empty, valid JSON file."""
    if os.path.isdir(path):
        raise ValueError(f"❌ {label} path is a directory, not a JSON file: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} file not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"{label} file is empty: {path}")
    with open(path, "r") as f:
        try:
            json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"{label} file is not valid JSON: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize COCO ground truth and predictions with FiftyOne")
    parser.add_argument("--coco_json", type=str, required=True, help="Path to COCO ground truth JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--predictions_json", type=str, help="Path to predictions JSON file")
    parser.add_argument("--dataset_name", type=str, default="efficientdet_voxel51_view", help="Name of FiftyOne dataset")
    args = parser.parse_args()

    gt_path = args.coco_json
    pred_path = None
    temp_gt_file = None

    # Normalize and absolutize the image directory path
    image_dir_abs_norm = os.path.abspath(args.image_dir).replace("\\", "/")

    # 1. Handle custom (non-COCO) ground truth format
    with open(gt_path, "r") as f:
        raw_gt = json.load(f)

    if isinstance(raw_gt, list):
        print("Wrapping ground truth annotations to COCO format...")
        wrapped_gt = wrap_annotations_to_coco(raw_gt)
        temp_gt_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(wrapped_gt, temp_gt_file)
        temp_gt_file.close()
        gt_path = temp_gt_file.name

    validate_json_file(gt_path, "Ground truth")

    # Load pycocotools object to help with image metadata and prediction wrapping
    coco_gt = COCO(gt_path)

    # Path Diagnostic 
    first_image_filename = coco_gt.imgs[1]["file_name"] if 1 in coco_gt.imgs else "N/A"
    expected_path = os.path.join(image_dir_abs_norm, first_image_filename).replace("\\", "/")
    print("-" * 50)
    print("Path Diagnostic")
    print(f"Image Directory (Final Normalized): {image_dir_abs_norm}")
    print(f"COCO file_name (1st): {first_image_filename}")
    print(f"Image exists? {os.path.exists(expected_path)}")
    print("-" * 50)

    if not os.path.exists(expected_path):
        raise FileNotFoundError(f"Image file not found at expected location: {expected_path}")

    # 2. Dataset Loading (Fix for initial directory error)
    if args.dataset_name in fo.list_datasets():
        fo.delete_dataset(args.dataset_name)

    print(f"✅ Final ground truth path: {gt_path}")
    print("Loading samples and ground truth using fo.Dataset.from_dir...")

    dataset = fo.Dataset.from_dir(
        dataset_dir=image_dir_abs_norm,
        dataset_type=fo.types.COCODetectionDataset,
        name=args.dataset_name,
        labels_path=gt_path,
        data_path=image_dir_abs_norm, 
        label_field="ground_truth",
    )

    # 3. Predictions Loading
    if args.predictions_json and os.path.exists(args.predictions_json):
        validate_json_file(args.predictions_json, "Predictions")

        with open(args.predictions_json, "r") as f:
            raw_preds = json.load(f)

        if isinstance(raw_preds, list) and isinstance(raw_preds[0], dict) and "predictions" in raw_preds[0]:
            raw_preds = raw_preds[-1]["predictions"]

        wrapped_preds = wrap_predictions_to_coco(raw_preds, coco_gt)

        # FIX: Check if predictions were successfully wrapped before trying to load them
        if not wrapped_preds["annotations"]:
            print("⚠️ Warning: Wrapped predictions list is empty after attempting to match to images. "
                  "Skipping addition of 'predictions' field.")
        else:
            # Only create the temporary file if there are annotations to write
            temp_pred_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(wrapped_preds, temp_pred_file)
            temp_pred_file.close()
            pred_path = temp_pred_file.name
    
            validate_json_file(pred_path, "Predictions")
            print(f"Adding predictions from temporary file: {pred_path}")
    
            # FIX: The required positional arguments are (dataset, label_field, labels_path, categories)
            categories = coco_gt.dataset.get("categories", []) 
    
            fouc.add_coco_labels(
                dataset, 
                "predictions", # 2nd positional: label_field
                pred_path,     # 3rd positional: labels_path
                categories     # 4th positional: categories
            )
    
    # 4. Launch and Cleanup
    print("Launching FiftyOne App...")
    session = fo.launch_app(dataset)
    session.wait()

    # Clean up temporary files
    if temp_gt_file and os.path.exists(gt_path) and gt_path != args.coco_json:
        os.remove(gt_path)
    if pred_path and os.path.exists(pred_path) and pred_path != args.predictions_json:
        os.remove(pred_path)


if __name__ == "__main__":
    main()