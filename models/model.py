import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(num_classes=4):
    # Load pretrained MobileNetV2 without top classification layers
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = True

    # Freeze the first 100 layers, train the remaining layers
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Build custom classifier
    model = models.Sequential([
        layers.Rescaling(1.0 / 255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()