import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm

# ========================
# CONFIGURABLE HYPERPARAMETERS
# ========================
EPOCHS = 20           # CPU-friendly
BATCH_SIZE = 16       # Reduce for RAM limits
FEATURES_DIR = "C:\\Users\\Admin\\OneDrive\\Desktop\\raj project\\features"
MODEL_SAVE_PATH = "best_model_cpu.h5"

# ========================
# LOAD PRE-EXTRACTED FEATURES
# ========================
X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
X_val   = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
y_val   = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))
X_test  = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

print("✅ Feature shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)

# ========================
# BUILD CPU-FRIENDLY MODEL
# ========================
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # smaller
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # binary classification: chest vs non-chest
])

model.compile(optimizer=Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ========================
# CALLBACKS
# ========================
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                             save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5,
                           restore_best_weights=True, verbose=1)

# ========================
# TRAIN WITH VERBOSE PROGRESS (CPU-friendly)
# ========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stop],
    verbose=1  # CPU-friendly progress
)

# ========================
# TEST EVALUATION
# ========================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Best CPU-friendly model saved at {MODEL_SAVE_PATH}")