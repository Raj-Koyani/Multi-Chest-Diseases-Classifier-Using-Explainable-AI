import os
import shutil
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

# ========================
# CONFIG
# ========================
DATA_DIR = "dataset_processed"  # Your preprocessed dataset
OUTPUT_DIR = "dataset_final"    # Folder where train/val/test folders will be created
IMG_SIZE = 224
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ========================
# DATA TRANSFORMS (placeholder)
# ========================
# Only for resizing before saving; can skip if already resized
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE))
])

# ========================
# LOAD FULL DATASET
# ========================
full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
num_samples = len(full_dataset)
indices = list(range(num_samples))
np.random.seed(RANDOM_SEED)
np.random.shuffle(indices)

# ========================
# SPLIT INDICES
# ========================
train_end = int(TRAIN_SPLIT * num_samples)
val_end = int((TRAIN_SPLIT + VAL_SPLIT) * num_samples)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

# ========================
# HELPER FUNCTION TO SAVE IMAGES
# ========================
def save_subset(subset_indices, subset_name):
    for idx in subset_indices:
        img_path, label = full_dataset.samples[idx]
        class_name = full_dataset.classes[label]

        out_dir = os.path.join(OUTPUT_DIR, subset_name, class_name)
        os.makedirs(out_dir, exist_ok=True)

        # Load image and save
        img = Image.open(img_path)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        base_name = os.path.basename(img_path)
        save_path = os.path.join(out_dir, base_name)
        img.save(save_path)

# ========================
# SAVE SPLITS
# ========================
save_subset(train_idx, "train")
save_subset(val_idx, "val")
save_subset(test_idx, "test")

# ========================
# PRINT SPLIT INFO
# ========================
def get_class_distribution(folder_path):
    counts = {}
    for cls in os.listdir(folder_path):
        cls_path = os.path.join(folder_path, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len(os.listdir(cls_path))
    return counts

print("Classes:", full_dataset.classes)
print("Training set distribution:", get_class_distribution(os.path.join(OUTPUT_DIR, "train")))
print("Validation set distribution:", get_class_distribution(os.path.join(OUTPUT_DIR, "val")))
print("Test set distribution:", get_class_distribution(os.path.join(OUTPUT_DIR, "test")))
print(f"Total samples: {num_samples}")