# src/evaluate.py

import torch
import json
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from effdet import EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet
from model_utils import wrap_model
from dataloader import get_dataloaders
from evaluation_logger import EvaluationLogger


def load_model(checkpoint_path, num_classes, input_size, device):
    config = get_efficientdet_config('tf_efficientdet_d0')
    config.num_classes = num_classes
    config.image_size = (input_size, input_size)
    model = EfficientDet(config, pretrained_backbone=False)
    model.class_net = HeadNet(config, num_outputs=num_classes)
    model = wrap_model(model, config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def evaluate_model(model, val_loader, device, debug=False):
    model.eval()

    ann_path = val_loader.dataset.annFile
    with open(ann_path, 'r+') as f:
        ann_data = json.load(f)
        if 'info' not in ann_data:
            ann_data['info'] = {
                "description": "Auto-added info field for COCO eval",
                "version": "1.0",
                "year": 2025
            }
            f.seek(0)
            json.dump(ann_data, f)
            f.truncate()

    coco_gt = COCO(ann_path)
    predictions = []
    image_map = {}

    if debug:
        print(f"\n🧪 Dataset contains {len(val_loader.dataset)} validation samples")
        print(f"Annotation file: {ann_path}")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            input_size = model.config.image_size[0]

            for i, bbox in enumerate(targets['bbox']):
                if bbox.numel() > 0:
                    targets['bbox'][i][:, [0, 2]] /= input_size
                    targets['bbox'][i][:, [1, 3]] /= input_size

            outputs = model(torch.stack(images), targets)
            if isinstance(outputs, dict):
                outputs = [outputs]

            for i, output in enumerate(outputs):
                if not isinstance(output, dict) or not all(k in output for k in ['bbox', 'scores', 'cls']):
                    continue

                boxes = output['bbox'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['cls'].cpu().numpy()
                img_id = i

                image_map[img_id] = {
                    'bbox': targets['bbox'][i],
                    'cls': targets['cls'][i],
                    'img_size': targets['img_size'][i],
                    'img_scale': targets['img_scale'][i]
                }

                if debug:
                    print(f"\n🔍 Debug Info for Image {img_id}")
                    print(f"  Boxes: {boxes.shape}")
                    print(f"  Scores: {scores}")
                    print(f"  Labels: {labels}")
                    print(f"  Filtered Predictions: {sum(score > 0.01 for score in scores)}")

                for box, score, label in zip(boxes, scores, labels):
                    if score < 0.01:
                        continue
                    x1, y1, x2, y2 = box
                    predictions.append({
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score)
                    })

    if not predictions:
        print("⚠️ No predictions were made by the model. Skipping COCO evaluation.")
        return 0.0, [], image_map

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_score = coco_eval.stats[0]
    return map_score, predictions, image_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EfficientDet model")