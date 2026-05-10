import tensorflow as tf

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

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
# Load testing dataset
# -----------------------------
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# Display class names
# -----------------------------
class_names = train_dataset.class_names

print("\nClass Names:")
print(class_names)

print("\nNumber of Classes:", len(class_names))

# -----------------------------
# Optimize performance
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

print("\nDatasets created successfully!")