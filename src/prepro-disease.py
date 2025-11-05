import os
import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ========================
# CONFIG
# ========================
RAW_DIR = "D:\pjt\dataset"           # your merged dataset path
PROCESSED_DIR = "dataset_processed"
IMG_SIZE = (224, 224)             # target image size
AUG_PER_IMAGE = 2                 # how many augmentations per minority image

# ========================
# MAKE OUTPUT DIRS
# ========================
os.makedirs(PROCESSED_DIR, exist_ok=True)

# get all classes
classes = [cls for cls in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, cls))]

# get counts before preprocessing
class_counts = {cls: len(os.listdir(os.path.join(RAW_DIR, cls))) for cls in classes}
print("\n📊 Raw class counts (before preprocessing):")
for cls, count in class_counts.items():
    print(f"   {cls}: {count}")

# smallest class size
min_count = min(class_counts.values())
print(f"\nBalancing all classes to: {min_count} images each\n")

# ========================
# DATA AUGMENTATION SETUP
# ========================
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

# ========================
# PREPROCESSING FUNCTION
# ========================
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    # Resize
    img = cv2.resize(img, IMG_SIZE)
    # Convert grayscale to RGB (3 channels)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

# ========================
# PROCESS + BALANCE DATA
# ========================
for cls in classes:
    cls_dir = os.path.join(RAW_DIR, cls)
    out_dir = os.path.join(PROCESSED_DIR, cls)
    os.makedirs(out_dir, exist_ok=True)

    images = os.listdir(cls_dir)
    random.shuffle(images)

    # store processed images in memory for augmentation
    processed_imgs = []

    # DOWN-SAMPLING: if class bigger than min_count
    selected_imgs = images[:min_count]

    # preprocess & save selected images
    for img_name in tqdm(selected_imgs, desc=f"Processing {cls}"):
        img_path = os.path.join(cls_dir, img_name)
        img = preprocess_image(img_path)
        if img is None:
            continue
        processed_imgs.append(img)
        save_path = os.path.join(out_dir, img_name)
        cv2.imwrite(save_path, img)

    # AUGMENT if class < min_count
    current_count = len(os.listdir(out_dir))
    while current_count < min_count:
        img = random.choice(processed_imgs)
        img = np.expand_dims(img, 0)
        aug_iter = datagen.flow(img, batch_size=1)
        aug_img = next(aug_iter)[0].astype(np.uint8)
        save_name = f"aug_{current_count}.png"
        cv2.imwrite(os.path.join(out_dir, save_name), aug_img)
        current_count += 1

print("\n✅ Balanced preprocessed dataset saved at:", PROCESSED_DIR)

# ========================
# FINAL CHECK
# ========================
final_counts = {cls: len(os.listdir(os.path.join(PROCESSED_DIR, cls))) for cls in classes}
print("\n📊 Processed dataset class counts (after preprocessing):")
for cls, count in final_counts.items():
    print(f"   {cls}: {count}")