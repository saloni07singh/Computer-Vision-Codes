
# Put images into data/train/<class_name>/ and data/val/<class_name>/
import tensorflow as tf
from tensorflow.keras import layers, models

SIZE = (64, 64)

train = tf.keras.utils.image_dataset_from_directory(
    "data/train", image_size=SIZE, batch_size=32
)
val = tf.keras.utils.image_dataset_from_directory(
    "data/val", image_size=SIZE, batch_size=32
)

classes = train.class_names

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(len(classes), activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train, validation_data=val, epochs=5)
model.save("traffic_sign_model.keras")
open("classes.txt", "w").write("\\n".join(classes))
print("Model saved.")