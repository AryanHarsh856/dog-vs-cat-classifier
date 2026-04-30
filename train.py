import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
dataset = tfds.load("cats_vs_dogs", as_supervised=False)
data = dataset["train"].take(2000).shuffle(2000)

train_size = 1600
val_size = 200

train_data = data.take(train_size)
remaining = data.skip(train_size)

val_data = remaining.take(val_size)
test_data = remaining.skip(val_size)

IMG_SIZE = 128

def preprocess(x):
    image = tf.image.resize(x["image"], (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    label = x["label"]
    return image, label

train_data = train_data.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
val_data = val_data.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = test_data.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Evaluate
model.evaluate(test_data)



# Create folder (IMPORTANT)
import os
os.makedirs("model", exist_ok=True)

# Save model
model.save("model/dog_cat_model.h5")

print("Model saved successfully!")