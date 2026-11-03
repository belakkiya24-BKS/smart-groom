
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# Dataset location
DATASET_DIR = Path("Dataset")

# Image settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 8

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

# Build AI model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(
    train_ds,
    epochs=10
)

# Save trained model
model.save("groom_model.keras")

print("AI Model Training Completed!")
print("Model saved as groom_model.keras")
