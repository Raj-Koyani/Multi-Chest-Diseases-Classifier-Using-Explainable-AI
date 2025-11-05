# Step 1 — split dataset into train/val/test

import splitfolders
import os

INPUT_FOLDER = "D:\pjt2\dataset"  # your dataset
OUTPUT_FOLDER = "data_split"           # will be created inside project folder

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Ratios: 70% train, 20% val, 10% test
splitfolders.ratio(INPUT_FOLDER, output=OUTPUT_FOLDER, seed=42, ratio=(0.7, 0.2, 0.1))

print("✅ Dataset split into train/val/test")
print("Check:", os.listdir(OUTPUT_FOLDER))