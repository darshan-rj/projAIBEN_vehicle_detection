import json
from collections import Counter

# Path to your split file (change as needed)
split_path = "data/splits/train.json"

with open(split_path, "r") as f:
    data = json.load(f)

image_ids = set(img["id"] for img in data["images"])
ann_image_ids = [ann["image_id"] for ann in data["annotations"]]

# Count how many annotations per image
ann_count = Counter(ann_image_ids)

print(f"Total images: {len(image_ids)}")
print(f"Total annotations: {len(data['annotations'])}")

# How many images have at least one annotation?
images_with_anns = [img_id for img_id in image_ids if ann_count[img_id] > 0]
print(f"Images with at least one annotation: {len(images_with_anns)}")

# Print a few images with no annotations
images_without_anns = [
    img_id for img_id in image_ids if ann_count[img_id] == 0]
print(f"Images with NO annotations: {len(images_without_anns)}")
if images_without_anns:
    print("Sample image IDs with no annotations:", images_without_anns[:10])

# Print a few annotation image_ids that are not in images (should be empty)
ann_ids_not_in_images = [
    img_id for img_id in ann_count if img_id not in image_ids]
if ann_ids_not_in_images:
    print("Annotations referencing missing images:", ann_ids_not_in_images)
else:
    print("All annotation image_ids are present in images.")
