import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.model import build_model

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 10

train_dir = "dataset/Training"
test_dir = "dataset/Testing"

# -----------------------------
# Load training dataset
# -----------------------------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# -----------------------------
# Load validation dataset
# -----------------------------
val_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# -----------------------------
# Optimize performance
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Build model
# -----------------------------
model = build_model(num_classes=4)

# Force model to build so summary shows trainable layers
model.build((None, 224, 224, 3))
model.summary()

# -----------------------------
# Callbacks
# -----------------------------
os.makedirs("saved_models", exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "saved_models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

# -----------------------------
# Save final model
# -----------------------------
model.save("saved_models/final_model.keras")

# -----------------------------
# Plot training history
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("saved_models/training_accuracy.png")
plt.show()

print("\nTraining completed successfully!")
print("Best model saved to: saved_models/best_model.keras")
print("Final model saved to: saved_models/final_model.keras")
print("Accuracy plot saved to: saved_models/training_accuracy.png")