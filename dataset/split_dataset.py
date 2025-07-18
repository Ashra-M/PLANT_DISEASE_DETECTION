import os
import shutil
import random
from tqdm import tqdm

# Set paths
source_dir = "PlantVillage"
output_dir = "dataset/data"
train_split = 0.8  # 80% training, 20% validation

# Target folders
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")

# Create target directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Loop through each class folder
for class_name in tqdm(os.listdir(source_dir), desc="Splitting dataset"):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_idx = int(len(images) * train_split)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Create class subfolders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Move images
    for img in train_images:
        shutil.copy2(
            os.path.join(class_path, img), os.path.join(train_dir, class_name, img)
        )

    for img in val_images:
        shutil.copy2(
            os.path.join(class_path, img), os.path.join(val_dir, class_name, img)
        )

print("✅ Dataset successfully split into:")
print(f"   - Training: {train_dir}")
print(f"   - Validation: {val_dir}")
