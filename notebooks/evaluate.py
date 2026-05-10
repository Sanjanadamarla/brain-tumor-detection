import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DIR = "dataset/Testing"

# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names
print("Class Names:", class_names)

# Prefetch for performance
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# Load best model
model = tf.keras.models.load_model("saved_models/best_model.keras")

# Evaluate accuracy
loss, accuracy = model.evaluate(test_dataset)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Get true labels
y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])

# Get predictions
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("saved_models/confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved to: saved_models/confusion_matrix.png")