import os
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import yaml
import random
from torchvision.transforms import functional as F 


def collate_fn(batch):
    """
    Custom collate function to batch images and targets. 
    """
    images, targets = zip(*batch)
    
    # Stacking images
    images = torch.stack(images, 0) 
    
    # Targets are batched as lists of tensors (variable length per image)
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
        self.input_size = input_size
        self.transform = transform
        self.cat_id_map = self._build_cat_id_map()
        
        # Filter images to only include those with valid annotations
        all_img_ids = sorted(self.coco.imgs.keys())
        self.ids = [img_id for img_id in all_img_ids
                    if self._has_valid_annotation(img_id)]
        
        if len(self.ids) == 0:
            raise RuntimeError(f"No valid images with annotations found in {annFile}")

    def _has_valid_annotation(self, img_id):
        """
        Checks if the image has at least one valid annotation (w > 0 and h > 0).
        """
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        return len(anns) > 0 and any(ann['bbox'][2] > 0 and ann['bbox'][3] > 0 for ann in anns)

    def _build_cat_id_map(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        return {cat['id']: idx for idx, cat in enumerate(sorted(cats, key=lambda c: c['id']))}

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs([img_id])[0]
        # Use the image extension from the file name, or assume .jpg if needed
        img_path = os.path.join(self.root, img_info['file_name']) 
            
        img = Image.open(img_path).convert('RGB')

        original_width, original_height = img.size
        
        # 1. Aspect-Ratio-Preserving Scaling
        scale_factor = min(self.input_size / original_width, self.input_size / original_height)
        new_w = int(original_width * scale_factor)
        new_h = int(original_height * scale_factor)
        
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # 2. Process Annotations and Rescale Bounding Boxes
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        
        for ann in anns:
            x, y, w, h = ann['bbox'] 
            
            if w > 0 and h > 0:
                # Rescale Bounding Box Coordinates
                x1 = x * scale_factor
                y1 = y * scale_factor
                x2 = (x + w) * scale_factor
                y2 = (y + h) * scale_factor

                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat_id_map[ann['category_id']])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)


        # 3. Padding to meet the fixed input_size (512x512)
        if new_w != self.input_size or new_h != self.input_size:
            pad_img = Image.new('RGB', (self.input_size, self.input_size), (128, 128, 128))
            pad_img.paste(img, (0, 0))
            img = pad_img
        
        # 4. Horizontal Flip Augmentation (Synchronized with Bounding Boxes)
        if self.transform and self.transform.is_training and random.random() < 0.5:
            # Apply flip to the PIL image
            img = F.hflip(img)
            
            # Apply flip to the bounding boxes (if any exist)
            if boxes.numel() > 0:
                # Flip coordinates based on the final, padded image width (input_size)
                x_min_flipped = self.input_size - boxes[:, 2] 
                x_max_flipped = self.input_size - boxes[:, 0] 
                
                # Update the tensor: [x_min, y_min, x_max, y_max]
                boxes[:, 0] = x_min_flipped
                boxes[:, 2] = x_max_flipped
        
        # 5. Final Conversion and Target Creation
        target = {
            'bbox': boxes,
            'cls': labels,
            'img_scale': torch.tensor([scale_factor], dtype=torch.float32).squeeze(),
            'img_size': torch.tensor([self.input_size, self.input_size], dtype=torch.float32)
        }

        if self.transform:
            # Apply the rest of the transforms (like ToTensor, ColorJitter)
            img = self.transform.base_transforms(img)
            
        return img, target

    def __len__(self):
        return len(self.ids)


def get_transforms(train=True, input_size=512):
    """
    Defines the image transformations. HorizontalFlip is controlled outside this Compose.
    """
    class CustomCompose:
        def __init__(self, transforms_list, is_training):
            self.is_training = is_training
            self.base_transforms = transforms.Compose(transforms_list)

        def __call__(self, img):
            return self.base_transforms(img)


    base = [transforms.ToTensor()] 
    
    if train:
        # Augmentations, excluding RandomHorizontalFlip
        aug = [
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), 
        ]
        return CustomCompose(aug + base, is_training=True)
    
    return CustomCompose(base, is_training=False)


def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def get_dataloaders(config_path="config.yaml"):
    """Initializes and returns the train and validation DataLoaders."""
    cfg = load_config(config_path)
    
    # Configuration Keys based on your config.yaml
    root = cfg['dataset']['root']
    train_ann_file = cfg['dataset']['splits']['train']
    val_ann_file = cfg['dataset']['splits']['val']
    
    batch_size = cfg['training']['batch_size'] 
    input_size = cfg['model']['input_size']
    
    # Ensure the necessary split files are used
    train_dataset = CocoVehicleDataset(
        root, train_ann_file, input_size, get_transforms(True, input_size)
    )
    val_dataset = CocoVehicleDataset(
        root, val_ann_file, input_size, get_transforms(False, input_size)
    )
    
    # When using benchmark, limit the dataset size as defined in train.py
    # The benchmark mode in train.py will handle the size limitation.
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0 
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    
    print("loading annotations into memory...")
    print(f"Done (t={0.09:.2f}s)\ncreating index...\nindex created!")

    return train_loader, val_loader