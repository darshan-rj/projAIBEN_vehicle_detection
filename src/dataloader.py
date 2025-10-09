import os
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import yaml


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    batched_targets = {
        'bbox': [t['bbox'] for t in targets],
        'cls': [t['cls'] for t in targets],
        'img_scale': torch.stack([t['img_scale'] for t in targets]),
        'img_size': torch.stack([t['img_size'] for t in targets])
    }
    return images, batched_targets


class CocoVehicleDataset(Dataset):
    def __init__(self, root, annFile, input_size, transform=None):
        self.root = root
        self.annFile = annFile
        self.coco = COCO(annFile)
        self.ids = [img_id for img_id in sorted(self.coco.imgs.keys())
                    if self._has_valid_annotation(img_id)]
        self.input_size = input_size
        self.transform = transform
        self.cat_id_map = self._build_cat_id_map()
        if len(self.ids) == 0:
            raise RuntimeError(f"No valid images with annotations found in {annFile}")

    def _has_valid_annotation(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        return any(ann['bbox'][2] > 0 and ann['bbox'][3] > 0 for ann in anns)

    def _build_cat_id_map(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        return {cat['id']: idx for idx, cat in enumerate(sorted(cats, key=lambda c: c['id']))}

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_id_map[ann['category_id']])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {
            'bbox': boxes,
            'cls': labels,
            'img_scale': torch.tensor([1.0], dtype=torch.float32).squeeze(),
            'img_size': torch.tensor([img_info['height'], img_info['width']], dtype=torch.float32)
        }

        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.ids)


def get_transforms(train=True, input_size=640):
    base = [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
    if train:
        aug = [transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomHorizontalFlip(0.5)]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_dataloaders(config_path="config.yaml"):
    cfg = load_config(config_path)
    root = cfg['dataset']['root']
    batch_size = cfg['training']['batch_size']
    input_size = cfg['model']['input_size']
    train_dataset = CocoVehicleDataset(root, cfg['dataset']['splits']['train'], input_size, get_transforms(True, input_size))
    val_dataset = CocoVehicleDataset(root, cfg['dataset']['splits']['val'], input_size, get_transforms(False, input_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader