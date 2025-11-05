import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import os
from tqdm import tqdm  # ✅ for live progress bars

# ==========================
# CONFIG
# ==========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 1️⃣ Load datasets
train_ds = image_dataset_from_directory(
    "data_split/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = image_dataset_from_directory(
    "data_split/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
test_ds = image_dataset_from_directory(
    "data_split/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("✅ Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# 2️⃣ DenseNet feature extractor (no top layer)
base_model = DenseNet121(weights="imagenet", include_top=False, pooling="avg")

# 3️⃣ Function to extract features (with tqdm progress)
def extract_features(dataset, dataset_name=""):
    features = []
    labels = []
    print(f"\n🚀 Extracting features for {dataset_name} dataset...")
    total_batches = tf.data.experimental.cardinality(dataset).numpy()
    for images, labs in tqdm(dataset, total=total_batches, desc=f"Processing {dataset_name}"):
        preprocessed = preprocess_input(images)
        feats = base_model.predict(preprocessed, verbose=0)
        features.append(feats)
        labels.append(labs)
    return np.concatenate(features), np.concatenate(labels)

# 4️⃣ Run extraction with live progress
X_train, y_train = extract_features(train_ds, "Train")
X_val, y_val = extract_features(val_ds, "Validation")
X_test, y_test = extract_features(test_ds, "Test")

print("\n✅ Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# 5️⃣ Save extracted features for reuse
os.makedirs("features", exist_ok=True)
np.save("features/X_train.npy", X_train)
np.save("features/y_train.npy", y_train)
np.save("features/X_val.npy", X_val)
np.save("features/y_val.npy", y_val)
np.save("features/X_test.npy", X_test)
np.save("features/y_test.npy", y_test)

print("\n✅ DenseNet feature extraction completed successfully!")