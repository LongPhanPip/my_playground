import tensorflow as tf
from tensorflow import keras

from sklearn import model_selection
import load_mnist


X, y, X_test, y_test = load_mnist.load_data()
X = X.reshape(-1, 28, 28) / 255
X_test = X_test.reshape(-1, 28, 28) / 255

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.1, random_state=42)

input_ = keras.layers.Input(shape=[28, 28, 1])
conv1 = keras.layers.Conv2D(64, 5, padding='same', activation='relu')(input_)
max_pool1 = keras.layers.MaxPooling2D(2)(conv1)
conv2 = keras.layers.Conv2D(128, 5, strides=1, padding='same', activation='relu')(max_pool1)
conv3 = keras.layers.Conv2D(128, 5, strides=1, padding='same', activation='relu')(conv2)
max_pool2 = keras.layers.MaxPooling2D(2)(conv3)
conv4 = keras.layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(max_pool2)
conv5 = keras.layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(conv4)
max_pool3 = keras.layers.MaxPooling2D(2)(conv5)
flatten = keras.layers.Flatten()(max_pool3)
dense = keras.layers.Dense(128, activation='relu')(flatten)
dropout = keras.layers.Dropout(0.5)(dense)
output = keras.layers.Dense(10, activation='softmax')(dropout)

model = keras.Model(inputs=[input_], outputs=[output])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
model.evaluate(X_test, y_test)
