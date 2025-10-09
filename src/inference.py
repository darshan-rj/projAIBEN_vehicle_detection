# src/inference.py

import os
import torch
import yaml
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet
from effdet.bench import DetBenchTrain


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(num_classes, input_size, ckpt_path):
    config = get_efficientdet_config('tf_efficientdet_d0')
    config.num_classes = num_classes
    config.image_size = (input_size, input_size)
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=num_classes)
    model = DetBenchTrain(net, config)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    return model


def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, image


def create_dummy_targets(batch_size, input_size):
    return {
        'bbox': [torch.zeros((1, 4)) for _ in range(batch_size)],
        'cls': [torch.zeros((1,), dtype=torch.int64) for _ in range(batch_size)],
        'img_size': torch.tensor([[input_size, input_size]] * batch_size),
        'img_scale': torch.tensor([1.0] * batch_size)
    }


def run_inference(image_path):
    cfg = load_config()
    input_size = cfg['model']['input_size']
    model = create_model(
        cfg['model']['num_classes'],
        input_size,
        os.path.join(cfg['model']['checkpoint_dir'], "best_model.pth")
    )

    input_tensor, original_image = preprocess_image(image_path, input_size)
    dummy_targets = create_dummy_targets(input_tensor.shape[0], input_size)

    with torch.no_grad():
        outputs = model(input_tensor, dummy_targets)

    detections = outputs['detections'].cpu().numpy()[0]  # shape: [max_det, 6]

    image_np = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2, score, label = det
        if score < cfg['inference']['conf_threshold']:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = int(label)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, f"Class {label} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Inference", image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    run_inference(args.image_path)