import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Habana specific libraries
from habana_frameworks.tensorflow import load_habana_module
load_habana_module()

import pathlib


dataset_url = '...tgz'
data_dir = tf.keras.utils.get_file('attention_data', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.png')))
# print(image_count)

# Create dataset
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
# print(class_names)

# Create model
normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

num_classes = 3

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.summary()

epochs = 10


def train():
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    model.save_weights('/media/classifier.h5')


def eval(image):
    test_path = image
    img = keras.preprocessing.image.load_img(
        test_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    model.load_weights('/Users/joshuabelofsky/Desktop/AWS/classifier.h5')
    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])
    print("Classification: " + (class_names[np.argmax(score)]))
    attention = (class_names[np.argmax(score)])
    return attention
