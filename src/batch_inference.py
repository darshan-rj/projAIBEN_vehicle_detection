import os
import torch
import yaml
import cv2
import json
from PIL import Image
from tqdm import tqdm
from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet
from effdet.bench import DetBenchTrain
from torchvision import transforms


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
    return transform(image).unsqueeze(0)


def create_dummy_targets(batch_size, input_size):
    return {
        'bbox': [torch.zeros((1, 4)) for _ in range(batch_size)],
        'cls': [torch.zeros((1,), dtype=torch.int64) for _ in range(batch_size)],
        'img_size': torch.tensor([[input_size, input_size]] * batch_size),
        'img_scale': torch.tensor([1.0] * batch_size)
    }


def run_batch_inference(image_dir, output_json):
    cfg = load_config()
    model = create_model(cfg['model']['num_classes'], cfg['model']['input_size'],
                         os.path.join(cfg['model']['checkpoint_dir'], "best_model.pth"))

    results = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith(cfg['dataset']['image_ext'])]

    for fname in tqdm(image_files, desc="Running batch inference"):
        path = os.path.join(image_dir, fname)
        input_tensor = preprocess_image(path, cfg['model']['input_size'])
        dummy_targets = create_dummy_targets(input_tensor.shape[0], cfg['model']['input_size'])

        with torch.no_grad():
            outputs = model(input_tensor, dummy_targets)

        detections = outputs['detections'].cpu().numpy()[0]  # shape: [max_det, 6]

        for det in detections:
            x1, y1, x2, y2, score, label = det
            if score < cfg['inference']['conf_threshold']:
                continue
            results.append({
                "image": fname,
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "score": float(score),
                "category_id": int(label)
            })

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Inference results saved to {output_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run batch inference")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of input images")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save results JSON")
    args = parser.parse_args()
    run_batch_inference(args.image_dir, args.output_json)