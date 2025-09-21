import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Load Fashion MNIST dataset
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the images to [0, 1]
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0

# Split into training (50,000) and validation (10,000)
x_train = x_train_full[:50000]
y_train = y_train_full[:50000]
x_val = x_train_full[50000:]
y_val = y_train_full[50000:]

# Reshape to (batch_size, 28, 28, 1) for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

shuffle_val = len(x_train) // 1000
batch_size = 32

# Convert NumPy arrays to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset   = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_dataset  = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Shuffle, batch, and prefetch
train_dataset = train_dataset.shuffle(shuffle_val).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset   = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset  = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def AlexNet():
  inp = layers.Input((28, 28, 1))
  x = layers.Conv2D(64, 3, 1, activation = 'relu')(inp)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D(3, 2)(x)
  x = layers.Conv2D(128, 3, 1, activation = 'relu')(inp)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D(3, 2)(x)
  x = layers.Conv2D(128, 3, 1, activation='relu')(x)  
  x = layers.Conv2D(256, 3, 1, activation='relu')(x)
  x = layers.MaxPooling2D(3, 2)(x)  
  x = layers.Flatten()(x)
  x = layers.Dense(256, activation='relu')(x)  
  x = layers.Dropout(0.6)(x)  
  x = layers.Dense(256, activation='relu')(x)  
  x = layers.Dropout(0.6)(x)  
  x = layers.Dense(10, activation='softmax')(x)

  model = Model(inputs = inp, outputs = x)

  return model

model = AlexNet()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

es = EarlyStopping(patience=5, 
                   monitor='loss')
model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=20,
          callbacks=[es])


