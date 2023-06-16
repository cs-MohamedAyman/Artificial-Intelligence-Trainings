## Ungraded Lab: Lambda Layer

This lab will show how you can define custom layers with the [Lambda](https://keras.io/api/layers/core_layers/lambda/) layer. You can either use [lambda functions](https://w3schools.com/python/python_lambda.asp) within the Lambda layer or define a custom function that the Lambda layer will call. Let's get started!

## Imports


```python
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
from tensorflow.keras import backend as K
```

## Prepare the Dataset


```python
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 0s 0us/step


## Build the Model

Here, we'll use a Lambda layer to define a custom layer in our network. We're using a lambda function to get the absolute value of the layer input.


```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128),
  tf.keras.layers.Lambda(lambda x: tf.abs(x)),
  tf.keras.layers.Dense(10, activation='softmax')
])
```


```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

    Train on 60000 samples
    Epoch 1/5
    60000/60000 [==============================] - 5s 83us/sample - loss: 0.2188 - accuracy: 0.9373
    Epoch 2/5
    60000/60000 [==============================] - 5s 78us/sample - loss: 0.0884 - accuracy: 0.9736
    Epoch 3/5
    60000/60000 [==============================] - 5s 78us/sample - loss: 0.0617 - accuracy: 0.9808
    Epoch 4/5
    60000/60000 [==============================] - 5s 79us/sample - loss: 0.0462 - accuracy: 0.9851
    Epoch 5/5
    60000/60000 [==============================] - 5s 79us/sample - loss: 0.0371 - accuracy: 0.9885
    10000/10000 [==============================] - 0s 38us/sample - loss: 0.0863 - accuracy: 0.9765





    [0.086345823792601, 0.9765]



Another way to use the Lambda layer is to pass in a function defined outside the model. The code below shows how a custom ReLU function is used as a custom layer in the model.


```python
def my_relu(x):
    return K.maximum(-0.1, x)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Lambda(my_relu),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

    Train on 60000 samples
    Epoch 1/5
    60000/60000 [==============================] - 5s 83us/sample - loss: 0.2602 - accuracy: 0.9263
    Epoch 2/5
    60000/60000 [==============================] - 5s 79us/sample - loss: 0.1148 - accuracy: 0.9656
    Epoch 3/5
    60000/60000 [==============================] - 5s 78us/sample - loss: 0.0809 - accuracy: 0.9752
    Epoch 4/5
    60000/60000 [==============================] - 5s 83us/sample - loss: 0.0592 - accuracy: 0.9816
    Epoch 5/5
    60000/60000 [==============================] - 5s 82us/sample - loss: 0.0455 - accuracy: 0.9858
    10000/10000 [==============================] - 0s 33us/sample - loss: 0.0736 - accuracy: 0.9766





    [0.07362337216013111, 0.9766]


