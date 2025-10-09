# src/train.py

import os
import torch
from effdet.efficientdet import HeadNet
from effdet import get_efficientdet_config, EfficientDet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from training_logger import TrainingLogger
from utils import deep_update
from dataloader import get_dataloaders, load_config
from model_utils import wrap_model


def create_model(num_classes, input_size):
    config = get_efficientdet_config('tf_efficientdet_d0')
    config.num_classes = num_classes
    config.image_size = (input_size, input_size)
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=num_classes)
    return net, config


def train(cfg_override=None):
    cfg = load_config()
    if cfg_override:
        cfg = deep_update(cfg, cfg_override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_model, config = create_model(cfg['model']['num_classes'], cfg['model']['input_size'])
    model = wrap_model(base_model, config).to(device)

    train_loader, _ = get_dataloaders()
    input_size = cfg['model']['input_size']

    if cfg.get("benchmark"):
        train_loader.dataset.ids = train_loader.dataset.ids[:50]

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'])
    num_epochs = cfg['training']['epochs']
    os.makedirs(cfg['model']['checkpoint_dir'], exist_ok=True)

    logger = TrainingLogger(
        log_path=os.path.join(cfg['model']['checkpoint_dir'], "training_log.txt"),
        csv_path=os.path.join(cfg['model']['checkpoint_dir'], "metrics.csv")
    )
    writer = SummaryWriter(log_dir=os.path.join(cfg['model']['checkpoint_dir'], "tensorboard"))
    best_model_path = os.path.join(cfg['model']['checkpoint_dir'], "best_model.pth")

    model.train()
    all_metrics = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images_gpu = [img.to(device) for img in images]

            try:
                for i, bbox in enumerate(targets['bbox']):
                    if bbox.numel() > 0:
                        targets['bbox'][i][:, [0, 2]] /= input_size
                        targets['bbox'][i][:, [1, 3]] /= input_size
            except Exception as e:
                print(f"[NORMALIZATION ERROR] Batch {batch_idx}: {e}")
                continue

            try:
                loss_dict = model(torch.stack(images_gpu), targets)
                loss = loss_dict['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            except Exception as e:
                print(f"[TRAINING ERROR] Batch {batch_idx}: {e}")
                continue

        all_metrics.append({'epoch': epoch+1, 'loss': epoch_loss})
        logger.log_epoch(epoch+1, epoch_loss)
        writer.add_scalar("Loss/train", epoch_loss, epoch+1)

        torch.save(model.state_dict(), f"{cfg['model']['checkpoint_dir']}/epoch_{epoch+1}.pth")
        if logger.is_best(epoch_loss):
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model updated at epoch {epoch+1}")

    import csv
    results_dir = cfg['model'].get('results_dir', cfg['model']['checkpoint_dir'])
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'metrics.csv'), 'w', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=['epoch', 'loss'])
        writer_csv.writeheader()
        writer_csv.writerows(all_metrics)

    writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train EfficientDet")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    args = parser.parse_args()

    if args.benchmark:
        print("⚡ Running benchmark mode: EfficientDet-D0, 2 epochs, batch size 2, limited data samples.")
        train(cfg_override={
            "model": {
                "input_size": 512,
                "num_classes": 5,
                "checkpoint_dir": "models/efficientdet_d0_benchmark"
            },
            "training": {
                "epochs": 2,
                "batch_size": 2,
                "lr": 0.001
            },
            "benchmark": True
        })
    else:
        print("🚀 Running full training mode with default configuration.")
        train()