
import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

VEHICLE_CLASSES = {
    'car': 1,
    'truck': 2,
    'bus': 3,
    'trailer': 4,
    'construction_vehicle': 5
}


def convert_to_coco(nuscenes_path, output_json):
    nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_path, verbose=True)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k} for k, v in VEHICLE_CLASSES.items()]
    }

    ann_id = 1
    img_id_counter = 1
    camera_channels = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    for sample in nusc.sample:
        for cam_name in camera_channels:
            cam_token = sample['data'].get(cam_name)
            if not cam_token:
                continue
            cam_data = nusc.get('sample_data', cam_token)
            img_filename = cam_data['filename']
            img_path = os.path.join(nusc.dataroot, img_filename)
            if not os.path.exists(img_path):
                continue
            with Image.open(img_path) as img:
                width, height = img.size

            coco["images"].append({
                "id": img_id_counter,
                "file_name": img_filename,
                "width": width,
                "height": height
            })

            # For each annotation in the sample, project to this camera
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                cat_name = ann['category_name']
                # Map nuScenes category_name to COCO vehicle classes
                if cat_name.startswith('vehicle.car'):
                    category = 'car'
                elif cat_name.startswith('vehicle.truck'):
                    category = 'truck'
                elif cat_name.startswith('vehicle.bus'):
                    category = 'bus'
                elif cat_name.startswith('vehicle.trailer'):
                    category = 'trailer'
                elif cat_name.startswith('vehicle.construction'):
                    category = 'construction_vehicle'
                else:
                    continue

                # Get 3D box and project to 2D
                box = nusc.get_box(ann_token)
                if box is None:
                    continue
                # Move box to ego vehicle coord
                sample_data = cam_data
                cs_record = nusc.get('calibrated_sensor',
                                     sample_data['calibrated_sensor_token'])
                pose_record = nusc.get(
                    'ego_pose', sample_data['ego_pose_token'])
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

                # Project corners
                corners = view_points(box.corners(), np.array(
                    cs_record['camera_intrinsic']), normalize=True)
                # Check if box is in front of camera
                if np.any(corners[2, :] < 0.1):
                    continue
                x_min, y_min = np.min(corners[0]), np.min(corners[1])
                x_max, y_max = np.max(corners[0]), np.max(corners[1])

                x_min = max(0, min(x_min, width))
                y_min = max(0, min(y_min, height))
                x_max = max(0, min(x_max, width))
                y_max = max(0, min(y_max, height))

                # Filter out degenerate or out-of-bounds boxes
                box_w = x_max - x_min
                box_h = y_max - y_min
                area = box_w * box_h
                # Only add annotation if box is valid and area is positive
                if box_w > 1 and box_h > 1 and area > 1 and x_min < x_max and y_min < y_max:
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id_counter,
                        "category_id": VEHICLE_CLASSES[category],
                        "bbox": [x_min, y_min, box_w, box_h],
                        "area": area,
                        "iscrowd": 0
                    })
                    ann_id += 1
            img_id_counter += 1

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"COCO annotations saved to {output_json}")
    split_dataset(coco, output_dir="data/splits")


def split_dataset(coco_dict, output_dir, train_ratio=0.7, val_ratio=0.2):
    import random
    os.makedirs(output_dir, exist_ok=True)

    # Separate images with and without annotations
    annotated_image_ids = set(ann['image_id']
                              for ann in coco_dict['annotations'])
    images_with_anns = [img for img in coco_dict['images']
                        if img['id'] in annotated_image_ids]
    images_without_anns = [img for img in coco_dict['images']
                           if img['id'] not in annotated_image_ids]

    # Shuffle both lists
    random.shuffle(images_with_anns)
    random.shuffle(images_without_anns)

    n_anns = len(images_with_anns)
    n_total = len(coco_dict['images'])
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # Split annotated images first
    n_train_anns = int(n_anns * train_ratio)
    n_val_anns = int(n_anns * val_ratio)
    n_test_anns = n_anns - n_train_anns - n_val_anns

    train_imgs = images_with_anns[:n_train_anns]
    val_imgs = images_with_anns[n_train_anns:n_train_anns + n_val_anns]
    test_imgs = images_with_anns[n_train_anns + n_val_anns:]

    # Fill up each split with unannotated images if needed
    def fill_split(split_imgs, n_needed, used_ids):
        filled = list(split_imgs)
        for img in images_without_anns:
            if img['id'] in used_ids:
                continue
            if len(filled) >= n_needed:
                break
            filled.append(img)
            used_ids.add(img['id'])
        return filled

    used_ids = set(img['id'] for img in train_imgs + val_imgs + test_imgs)
    train_imgs = fill_split(train_imgs, n_train, used_ids)
    val_imgs = fill_split(val_imgs, n_val, used_ids)
    test_imgs = fill_split(test_imgs, n_test, used_ids)

    def make_split(imgs, name):
        ids = set(img['id'] for img in imgs)
        split = {
            "images": imgs,
            "annotations": [ann for ann in coco_dict['annotations'] if ann['image_id'] in ids],
            "categories": coco_dict['categories']
        }
        with open(os.path.join(output_dir, f"{name}.json"), 'w') as f:
            json.dump(split, f, indent=2)
        print(
            f"{name}.json saved with {len(split['images'])} images, {len(split['annotations'])} annotations")

    make_split(train_imgs, "train")
    make_split(val_imgs, "val")
    make_split(test_imgs, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert nuScenes-mini to COCO format")
    parser.add_argument("--nuscenes_path", type=str,
                        required=True, help="Path to nuScenes-mini root")
    parser.add_argument("--output_json", type=str,
                        required=True, help="Path to save full COCO JSON")
    args = parser.parse_args()
    convert_to_coco(args.nuscenes_path, args.output_json)
