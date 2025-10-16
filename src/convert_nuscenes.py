import os
import json
import argparse
import random # Added for shuffling in split_dataset
from tqdm import tqdm
from PIL import Image
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

# Dictionary to map vehicle names to COCO category IDs
VEHICLE_CLASSES = {
    'car': 1,
    'truck': 2,
    'bus': 3,
    'trailer': 4,
    'construction_vehicle': 5
}


def convert_to_coco(nuscenes_path, output_json):
    """
    Converts nuScenes-mini dataset annotations into COCO format for vehicle detection.
    3D bounding boxes are projected onto 2D camera images.
    """
    # Use 'v1.0-mini' as per the original code
    nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_path, verbose=True)
    
    # Initialize the COCO output structure
    output_data = {
        "categories": [{"id": v, "name": k} for k, v in VEHICLE_CLASSES.items()],
        "images": [],
        "annotations": [] # CORRECTED: Added top-level annotations list
    }

    ann_id = 1
    img_id_counter = 1
    camera_channels = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    print(f"Starting conversion for {len(nusc.sample)} samples...")
    # Use tqdm for progress tracking
    for sample in tqdm(nusc.sample, desc="Processing nuScenes samples"):
        for cam_name in camera_channels:
            cam_token = sample['data'].get(cam_name)
            if not cam_token:
                continue
                
            cam_data = nusc.get('sample_data', cam_token)
            img_filename = cam_data['filename']
            img_path = os.path.join(nusc.dataroot, img_filename)
            
            if not os.path.exists(img_path):
                # print(f"Warning: Image not found at {img_path}")
                continue
                
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                # print(f"Error opening image {img_path}: {e}")
                continue


            image_entry = {
                "id": img_id_counter,
                "file_name": img_filename,
                "width": width,
                "height": height,
            }
            
            # Temporary list to hold annotations for the current image
            image_annotations = [] 

            # Get the sensor and ego-vehicle calibration data
            sample_data = cam_data
            cs_record = nusc.get('calibrated_sensor',
                                 sample_data['calibrated_sensor_token'])
            pose_record = nusc.get(
                'ego_pose', sample_data['ego_pose_token'])

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
                    continue # Skip non-vehicle or unlisted vehicle classes

                # Get 3D box and transform it from global coordinates to the 
                # camera's coordinate system for 2D projection
                box = nusc.get_box(ann_token)
                if box is None:
                    continue
                    
                # 1. World to Ego-vehicle frame at sample timestamp
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)
                # 2. Ego-vehicle frame to Sensor (Camera) frame
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

                # Project the 3D box corners to 2D image coordinates
                corners = view_points(box.corners(), np.array(
                    cs_record['camera_intrinsic']), normalize=True)
                
                # Check if the object is in front of the camera (z-coordinate check)
                # The minimum z-coordinate of the 8 box corners must be positive
                # A small epsilon (0.1) is used for a safety margin
                if np.any(corners[2, :] < 0.1): 
                    continue
                    
                # Calculate the 2D bounding box (AABB of projected 3D corners)
                x_min, y_min = np.min(corners[0]), np.min(corners[1])
                x_max, y_max = np.max(corners[0]), np.max(corners[1])

                # Clamp coordinates to image boundaries
                x_min = max(0, min(x_min, width))
                y_min = max(0, min(y_min, height))
                x_max = max(0, min(x_max, width))
                y_max = max(0, min(y_max, height))

                # Filter out degenerate or invalid boxes
                box_w = x_max - x_min
                box_h = y_max - y_min
                area = box_w * box_h
                
                # Only add annotation if box is valid and area is positive
                if box_w > 1 and box_h > 1 and area > 1 and x_min < x_max and y_min < y_max:
                    image_annotations.append({
                        "id": ann_id,
                        "image_id": img_id_counter, # COCO format requires image_id in annotations
                        "category_id": VEHICLE_CLASSES[category],
                        # COCO bbox format is [x_min, y_min, width, height]
                        "bbox": [x_min, y_min, box_w, box_h], 
                        "area": area,
                        "iscrowd": 0
                    })
                    ann_id += 1
                    
            # Add image and its annotations to the final structure
            output_data["images"].append(image_entry)
            output_data["annotations"].extend(image_annotations) # CORRECTED: Extend the top-level annotations
            img_id_counter += 1

    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"COCO annotations saved to {output_json} with {len(output_data['images'])} images and {len(output_data['annotations'])} annotations.")
    
    # Pass the full output_data for splitting
    split_dataset(output_data, output_dir="data/splits")


def split_dataset(data_dict, output_dir, train_ratio=0.7, val_ratio=0.2):
    """
    Splits the COCO-formatted data into training, validation, and test sets.
    The split is based on images, trying to maintain the ratio of annotated images.
    """
    
    # The fix: The original code was using a variable 'coco_dict' which was never defined.
    # It should use the input argument 'data_dict'.
    
    os.makedirs(output_dir, exist_ok=True)

    # Group annotations by image_id for easy look-up
    img_to_anns = {}
    for ann in data_dict['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # Separate images with and without annotations
    images_with_anns = []
    images_without_anns = []

    for img in data_dict['images']:
        # Get annotations for the image. If none, it's an unannotated image.
        img_id = img['id']
        annotations = img_to_anns.get(img_id, [])
        
        # Store annotations within the image dict for easy split creation later
        img['annotations'] = annotations 
        
        if len(annotations) > 0:
            images_with_anns.append(img)
        else:
            images_without_anns.append(img)

    # Shuffle both lists to ensure random distribution
    random.shuffle(images_with_anns)
    random.shuffle(images_without_anns)

    n_total = len(data_dict['images'])
    n_anns = len(images_with_anns)
    
    # Calculate target split sizes
    n_train_target = int(n_total * train_ratio)
    n_val_target = int(n_total * val_ratio)
    # The remaining images go to the test set
    n_test_target = n_total - n_train_target - n_val_target 

    # 1. Split annotated images first to ensure representation in all sets
    # Distribute annotated images proportionally
    n_train_anns = int(n_anns * train_ratio)
    n_val_anns = int(n_anns * val_ratio)
    n_test_anns = n_anns - n_train_anns - n_val_anns 

    train_imgs = images_with_anns[:n_train_anns]
    val_imgs = images_with_anns[n_train_anns:n_train_anns + n_val_anns]
    test_imgs = images_with_anns[n_train_anns + n_val_anns:]
    
    # Keep track of unannotated images added to avoid duplication
    unannotated_pool = list(images_without_anns)
    
    # 2. Fill up each split with unannotated images if needed
    def fill_split(split_imgs, n_needed, pool):
        """Fills a split with images from the unannotated pool up to n_needed."""
        filled = list(split_imgs)
        while len(filled) < n_needed and pool:
            filled.append(pool.pop(0)) # Move image from pool to split
        return filled

    # Fill train split
    train_imgs = fill_split(train_imgs, n_train_target, unannotated_pool)
    # Fill validation split
    val_imgs = fill_split(val_imgs, n_val_target, unannotated_pool)
    # Fill test split with the rest of the available images
    test_imgs.extend(unannotated_pool) # Add all remaining unannotated images to test

    # Adjust test size if it exceeds the target (can happen if float precision
    # makes the sum of targets < n_total)
    test_imgs = test_imgs[:n_test_target] 


    def make_split(imgs, name):
        """Creates and saves a split file."""
        # The fix: The original make_split function used `split['annotations']`
        # but the list of annotations is already nested inside each image entry.
        # For a standard COCO format *file*, we need to extract all annotations
        # into a top-level list.
        all_anns = []
        for img in imgs:
            # Check if 'annotations' key exists and extract
            if 'annotations' in img:
                all_anns.extend(img.pop('annotations')) # Extract and remove from image dict

        split = {
            "images": imgs,
            "annotations": all_anns, # CORRECTED: Include top-level annotations
            "categories": data_dict['categories']
        }
        
        filepath = os.path.join(output_dir, f"{name}.json")
        with open(filepath, 'w') as f:
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