<a href="https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-3-public/blob/main/Course%201%20-%20Custom%20Models%2C%20Layers%20and%20Loss%20Functions/Week%205%20-%20Callbacks/C1_W5_Lab_1_exploring-callbacks.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Ungraded Lab: Introduction to Keras callbacks

In Keras, `Callback` is a Python class meant to be subclassed to provide specific functionality, with a set of methods called at various stages of training (including batch/epoch start and ends), testing, and predicting. Callbacks are useful to get a view on internal states and statistics of the model during training. The methods of the callbacks can  be called at different stages of training/evaluating/inference. Keras has available [callbacks](https://keras.io/api/callbacks/) and we'll show how you can use it in the following sections. Please click the **Open in Colab** badge above to complete this exercise in Colab. This will allow you to take advantage of the free GPU runtime (for faster training) and compatibility with all the packages needed in this notebook.

## Model methods that take callbacks
Users can supply a list of callbacks to the following `tf.keras.Model` methods:
* [`fit()`](https://tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#fit), [`fit_generator()`](https://tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#fit_generator)
Trains the model for a fixed number of epochs (iterations over a dataset, or data yielded batch-by-batch by a Python generator).
* [`evaluate()`](https://tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#evaluate), [`evaluate_generator()`](https://tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#evaluate_generator)
Evaluates the model for given data or data generator. Outputs the loss and metric values from the evaluation.
* [`predict()`](https://tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#predict), [`predict_generator()`](https://tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model#predict_generator)
Generates output predictions for the input data or data generator.

## Imports


```python
from __future__ import absolute_import, division, print_function, unicode_literals

try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
except Exception:
    pass

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import io
from PIL import Image

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
%load_ext tensorboard

import os
import matplotlib.pylab as plt
import numpy as np
import math
import datetime
import pandas as pd

print("Version: ", tf.__version__)
tf.get_logger().setLevel('INFO')
```

    Version:  2.1.0


# Examples of Keras callback applications
The following section will guide you through creating simple [Callback](https://keras.io/api/callbacks/) applications.


```python
# Download and prepare the horses or humans dataset

# horses_or_humans 3.0.0 has already been downloaded for you
path = "./tensorflow_datasets"
splits, info = tfds.load('horses_or_humans', data_dir=path, as_supervised=True, with_info=True, split=['train[:80%]', 'train[80%:]', 'test'])

(train_examples, validation_examples, test_examples) = splits

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
```


```python
SIZE = 150 #@param {type:"slider", min:64, max:300, step:1}
IMAGE_SIZE = (SIZE, SIZE)
```


```python
def format_image(image, label):
  image = tf.image.resize(image, IMAGE_SIZE) / 255.0
  return  image, label
```


```python
BATCH_SIZE = 32 #@param {type:"integer"}
```


```python
train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test_examples.map(format_image).batch(1)
```


```python
for image_batch, label_batch in train_batches.take(1):
  pass

image_batch.shape
```




    TensorShape([32, 150, 150, 3])




```python
def build_model(dense_units, input_shape=IMAGE_SIZE + (3,)):
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(dense_units, activation='relu'),
      tf.keras.layers.Dense(2, activation='softmax')
  ])
  return model
```

## [TensorBoard](https://keras.io/api/callbacks/tensorboard/)

Enable visualizations for TensorBoard.


```python
!rm -rf logs
```


```python
model = build_model(dense_units=256)
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)

model.fit(train_batches,
          epochs=10,
          validation_data=validation_batches,
          callbacks=[tensorboard_callback])
```

    Epoch 1/10
    26/26 [==============================] - 21s 806ms/step - loss: 0.6718 - accuracy: 0.5949 - val_loss: 0.6664 - val_accuracy: 0.5122
    Epoch 2/10
    26/26 [==============================] - 20s 769ms/step - loss: 0.6190 - accuracy: 0.7117 - val_loss: 0.5923 - val_accuracy: 0.7415
    Epoch 3/10
    26/26 [==============================] - 20s 777ms/step - loss: 0.5548 - accuracy: 0.7591 - val_loss: 0.5761 - val_accuracy: 0.7220
    Epoch 4/10
    26/26 [==============================] - 20s 765ms/step - loss: 0.4943 - accuracy: 0.7737 - val_loss: 0.4535 - val_accuracy: 0.7805
    Epoch 5/10
    26/26 [==============================] - 20s 769ms/step - loss: 0.4654 - accuracy: 0.7895 - val_loss: 0.4283 - val_accuracy: 0.8488
    Epoch 6/10
    26/26 [==============================] - 20s 762ms/step - loss: 0.3895 - accuracy: 0.8370 - val_loss: 0.3737 - val_accuracy: 0.8780
    Epoch 7/10
    26/26 [==============================] - 20s 769ms/step - loss: 0.3378 - accuracy: 0.8564 - val_loss: 0.2947 - val_accuracy: 0.8927
    Epoch 8/10
    26/26 [==============================] - 20s 765ms/step - loss: 0.2828 - accuracy: 0.8881 - val_loss: 0.2158 - val_accuracy: 0.9610
    Epoch 9/10
    26/26 [==============================] - 20s 766ms/step - loss: 0.2442 - accuracy: 0.9246 - val_loss: 0.1730 - val_accuracy: 0.9756
    Epoch 10/10
    26/26 [==============================] - 20s 769ms/step - loss: 0.1830 - accuracy: 0.9465 - val_loss: 0.1372 - val_accuracy: 0.9805





    <tensorflow.python.keras.callbacks.History at 0x7fc579733210>




```python
%tensorboard --logdir logs
```



<iframe id="tensorboard-frame-72cee4c65981c5b1" width="100%" height="800" frameborder="0">
</iframe>
<script>
  (function() {
    const frame = document.getElementById("tensorboard-frame-72cee4c65981c5b1");
    const url = new URL("/", window.location);
    url.port = 6006;
    frame.src = url;
  })();
</script>



## [Model Checkpoint](https://keras.io/api/callbacks/model_checkpoint/)

Callback to save the Keras model or model weights at some frequency.


```python
model = build_model(dense_units=256)
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_batches,
          epochs=5,
          validation_data=validation_batches,
          verbose=2,
          callbacks=[ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.h5', verbose=1),
          ])
```

    Epoch 1/5

    Epoch 00001: saving model to weights.01-0.64.h5
    26/26 - 21s - loss: 0.6654 - accuracy: 0.5888 - val_loss: 0.6405 - val_accuracy: 0.6244
    Epoch 2/5

    Epoch 00002: saving model to weights.02-0.65.h5
    26/26 - 20s - loss: 0.6199 - accuracy: 0.6715 - val_loss: 0.6502 - val_accuracy: 0.5171
    Epoch 3/5

    Epoch 00003: saving model to weights.03-0.52.h5
    26/26 - 20s - loss: 0.5680 - accuracy: 0.7044 - val_loss: 0.5155 - val_accuracy: 0.7902
    Epoch 4/5

    Epoch 00004: saving model to weights.04-0.53.h5
    26/26 - 20s - loss: 0.5069 - accuracy: 0.7603 - val_loss: 0.5274 - val_accuracy: 0.7854
    Epoch 5/5

    Epoch 00005: saving model to weights.05-0.55.h5
    26/26 - 20s - loss: 0.4591 - accuracy: 0.8102 - val_loss: 0.5500 - val_accuracy: 0.7024





    <tensorflow.python.keras.callbacks.History at 0x7fc5795823d0>




```python
model = build_model(dense_units=256)
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_batches,
          epochs=1,
          validation_data=validation_batches,
          verbose=2,
          callbacks=[ModelCheckpoint('saved_model', verbose=1)
          ])
```


    Epoch 00001: saving model to saved_model
    WARNING:tensorflow:From /opt/conda/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1786: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.


    WARNING:tensorflow:From /opt/conda/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1786: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.


    INFO:tensorflow:Assets written to: saved_model/assets


    INFO:tensorflow:Assets written to: saved_model/assets


    26/26 - 21s - loss: 0.6704 - accuracy: 0.5754 - val_loss: 0.6482 - val_accuracy: 0.7122





    <tensorflow.python.keras.callbacks.History at 0x7fc5793f8f50>




```python
model = build_model(dense_units=256)
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_batches,
          epochs=2,
          validation_data=validation_batches,
          verbose=2,
          callbacks=[ModelCheckpoint('model.h5', verbose=1)
          ])
```

    Epoch 1/2

    Epoch 00001: saving model to model.h5
    26/26 - 20s - loss: 0.6813 - accuracy: 0.5414 - val_loss: 0.6890 - val_accuracy: 0.4341
    Epoch 2/2

    Epoch 00002: saving model to model.h5
    26/26 - 19s - loss: 0.6605 - accuracy: 0.6083 - val_loss: 0.6637 - val_accuracy: 0.4878





    <tensorflow.python.keras.callbacks.History at 0x7fc5791f4190>



## [Early stopping](https://keras.io/api/callbacks/early_stopping/)

Stop training when a monitored metric has stopped improving.


```python
model = build_model(dense_units=256)
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_batches,
          epochs=50,
          validation_data=validation_batches,
          verbose=2,
          callbacks=[EarlyStopping(
              patience=3,
              min_delta=0.05,
              baseline=0.8,
              mode='min',
              monitor='val_loss',
              restore_best_weights=True,
              verbose=1)
          ])
```

    Epoch 1/50
    26/26 - 20s - loss: 0.6586 - accuracy: 0.6314 - val_loss: 0.6269 - val_accuracy: 0.6927
    Epoch 2/50
    26/26 - 22s - loss: 0.5968 - accuracy: 0.7105 - val_loss: 0.5662 - val_accuracy: 0.7463
    Epoch 3/50
    26/26 - 20s - loss: 0.5713 - accuracy: 0.7263 - val_loss: 0.5166 - val_accuracy: 0.7756
    Epoch 4/50
    26/26 - 20s - loss: 0.4942 - accuracy: 0.7944 - val_loss: 0.4573 - val_accuracy: 0.7854
    Epoch 5/50
    26/26 - 20s - loss: 0.4476 - accuracy: 0.8139 - val_loss: 0.5552 - val_accuracy: 0.6634
    Epoch 6/50
    26/26 - 20s - loss: 0.3800 - accuracy: 0.8382 - val_loss: 0.4258 - val_accuracy: 0.8293
    Epoch 7/50
    26/26 - 20s - loss: 0.3373 - accuracy: 0.8625 - val_loss: 0.2768 - val_accuracy: 0.9122
    Epoch 8/50
    26/26 - 20s - loss: 0.2722 - accuracy: 0.9002 - val_loss: 0.2535 - val_accuracy: 0.9122
    Epoch 9/50
    26/26 - 21s - loss: 0.2287 - accuracy: 0.9307 - val_loss: 0.2443 - val_accuracy: 0.9171
    Epoch 10/50
    26/26 - 20s - loss: 0.1676 - accuracy: 0.9574 - val_loss: 0.1655 - val_accuracy: 0.9463
    Epoch 11/50
    26/26 - 20s - loss: 0.1416 - accuracy: 0.9659 - val_loss: 0.1579 - val_accuracy: 0.9415
    Epoch 12/50
    26/26 - 20s - loss: 0.1125 - accuracy: 0.9769 - val_loss: 0.0947 - val_accuracy: 0.9902
    Epoch 13/50
    26/26 - 20s - loss: 0.0980 - accuracy: 0.9720 - val_loss: 0.0752 - val_accuracy: 0.9902
    Epoch 14/50
    26/26 - 20s - loss: 0.0950 - accuracy: 0.9757 - val_loss: 0.0765 - val_accuracy: 0.9805
    Epoch 15/50
    Restoring model weights from the end of the best epoch.
    26/26 - 20s - loss: 0.0722 - accuracy: 0.9891 - val_loss: 0.0712 - val_accuracy: 0.9805
    Epoch 00015: early stopping





    <tensorflow.python.keras.callbacks.History at 0x7fc579044410>



## [CSV Logger](https://keras.io/api/callbacks/csv_logger/)

Callback that streams epoch results to a CSV file.


```python
model = build_model(dense_units=256)
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

csv_file = 'training.csv'

model.fit(train_batches,
          epochs=5,
          validation_data=validation_batches,
          callbacks=[CSVLogger(csv_file)
          ])
```

    Epoch 1/5
    26/26 [==============================] - 21s 801ms/step - loss: 0.6772 - accuracy: 0.5474 - val_loss: 0.6591 - val_accuracy: 0.8000
    Epoch 2/5
    26/26 [==============================] - 21s 801ms/step - loss: 0.6319 - accuracy: 0.7470 - val_loss: 0.6453 - val_accuracy: 0.5366
    Epoch 3/5
    26/26 [==============================] - 22s 862ms/step - loss: 0.5733 - accuracy: 0.7616 - val_loss: 0.6296 - val_accuracy: 0.5512
    Epoch 4/5
    26/26 [==============================] - 23s 869ms/step - loss: 0.4903 - accuracy: 0.8005 - val_loss: 0.4541 - val_accuracy: 0.8585
    Epoch 5/5
    26/26 [==============================] - 22s 839ms/step - loss: 0.4441 - accuracy: 0.8224 - val_loss: 0.3586 - val_accuracy: 0.8780





    <tensorflow.python.keras.callbacks.History at 0x7fc578fb1e10>




```python
pd.read_csv(csv_file).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epoch</th>
      <th>accuracy</th>
      <th>loss</th>
      <th>val_accuracy</th>
      <th>val_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.547445</td>
      <td>0.677462</td>
      <td>0.800000</td>
      <td>0.659132</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.746959</td>
      <td>0.631845</td>
      <td>0.536585</td>
      <td>0.645303</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.761557</td>
      <td>0.573608</td>
      <td>0.551219</td>
      <td>0.629637</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.800487</td>
      <td>0.491448</td>
      <td>0.858537</td>
      <td>0.454133</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.822384</td>
      <td>0.444714</td>
      <td>0.878049</td>
      <td>0.358627</td>
    </tr>
  </tbody>
</table>
</div>



## [Learning Rate Scheduler](https://keras.io/api/callbacks/learning_rate_scheduler/)

Updates the learning rate during training.


```python
model = build_model(dense_units=256)
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

def step_decay(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 1
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

model.fit(train_batches,
          epochs=5,
          validation_data=validation_batches,
          callbacks=[LearningRateScheduler(step_decay, verbose=1),
                    TensorBoard(log_dir='./log_dir')])
```


    Epoch 00001: LearningRateScheduler reducing learning rate to 0.005.
    Epoch 1/5
    26/26 [==============================] - 35s 1s/step - loss: 0.6792 - accuracy: 0.5499 - val_loss: 0.6319 - val_accuracy: 0.7415

    Epoch 00002: LearningRateScheduler reducing learning rate to 0.0025.
    Epoch 2/5
    26/26 [==============================] - 23s 887ms/step - loss: 0.6218 - accuracy: 0.6959 - val_loss: 0.6183 - val_accuracy: 0.7024

    Epoch 00003: LearningRateScheduler reducing learning rate to 0.00125.
    Epoch 3/5
    26/26 [==============================] - 22s 846ms/step - loss: 0.5996 - accuracy: 0.7384 - val_loss: 0.6067 - val_accuracy: 0.7171

    Epoch 00004: LearningRateScheduler reducing learning rate to 0.000625.
    Epoch 4/5
    26/26 [==============================] - 23s 892ms/step - loss: 0.5879 - accuracy: 0.7567 - val_loss: 0.6009 - val_accuracy: 0.7171

    Epoch 00005: LearningRateScheduler reducing learning rate to 0.0003125.
    Epoch 5/5
    26/26 [==============================] - 22s 857ms/step - loss: 0.5821 - accuracy: 0.7628 - val_loss: 0.5996 - val_accuracy: 0.7220





    <tensorflow.python.keras.callbacks.History at 0x7fc571fe9490>




```python
%tensorboard --logdir log_dir
```



<iframe id="tensorboard-frame-5753740d2c1898ea" width="100%" height="800" frameborder="0">
</iframe>
<script>
  (function() {
    const frame = document.getElementById("tensorboard-frame-5753740d2c1898ea");
    const url = new URL("/", window.location);
    url.port = 6007;
    frame.src = url;
  })();
</script>



## [ReduceLROnPlateau](https://keras.io/api/callbacks/reduce_lr_on_plateau/)

Reduce learning rate when a metric has stopped improving.


```python
model = build_model(dense_units=256)
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_batches,
          epochs=50,
          validation_data=validation_batches,
          callbacks=[ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.2, verbose=1,
                                       patience=1, min_lr=0.001),
                     TensorBoard(log_dir='./log_dir')])
```

    Epoch 1/50
    26/26 [==============================] - 25s 952ms/step - loss: 0.6692 - accuracy: 0.5888 - val_loss: 0.6434 - val_accuracy: 0.6537
    Epoch 2/50
    26/26 [==============================] - 21s 806ms/step - loss: 0.6077 - accuracy: 0.6922 - val_loss: 0.5818 - val_accuracy: 0.7366
    Epoch 3/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.5640 - accuracy: 0.7138
    Epoch 00003: ReduceLROnPlateau reducing learning rate to 0.0019999999552965165.
    26/26 [==============================] - 20s 786ms/step - loss: 0.5620 - accuracy: 0.7178 - val_loss: 0.6412 - val_accuracy: 0.5854
    Epoch 4/50
    26/26 [==============================] - 20s 766ms/step - loss: 0.5133 - accuracy: 0.7555 - val_loss: 0.5252 - val_accuracy: 0.7463
    Epoch 5/50
    26/26 [==============================] - 20s 776ms/step - loss: 0.4982 - accuracy: 0.7701 - val_loss: 0.5247 - val_accuracy: 0.7805
    Epoch 6/50
    26/26 [==============================] - 20s 781ms/step - loss: 0.4891 - accuracy: 0.7749 - val_loss: 0.5109 - val_accuracy: 0.7854
    Epoch 7/50
    26/26 [==============================] - 21s 804ms/step - loss: 0.4761 - accuracy: 0.7859 - val_loss: 0.4888 - val_accuracy: 0.7463
    Epoch 8/50
    26/26 [==============================] - 21s 796ms/step - loss: 0.4653 - accuracy: 0.7810 - val_loss: 0.4840 - val_accuracy: 0.7951
    Epoch 9/50
    26/26 [==============================] - 20s 781ms/step - loss: 0.4568 - accuracy: 0.7883 - val_loss: 0.4729 - val_accuracy: 0.8049
    Epoch 10/50
    26/26 [==============================] - 20s 761ms/step - loss: 0.4432 - accuracy: 0.8029 - val_loss: 0.4569 - val_accuracy: 0.7561
    Epoch 11/50
    26/26 [==============================] - 20s 758ms/step - loss: 0.4326 - accuracy: 0.8090 - val_loss: 0.4539 - val_accuracy: 0.8098
    Epoch 12/50
    26/26 [==============================] - 20s 766ms/step - loss: 0.4205 - accuracy: 0.8200 - val_loss: 0.4497 - val_accuracy: 0.8293
    Epoch 13/50
    26/26 [==============================] - 21s 800ms/step - loss: 0.4140 - accuracy: 0.8224 - val_loss: 0.4417 - val_accuracy: 0.8341
    Epoch 14/50
    26/26 [==============================] - 21s 795ms/step - loss: 0.4039 - accuracy: 0.8345 - val_loss: 0.4099 - val_accuracy: 0.8195
    Epoch 15/50
    26/26 [==============================] - 20s 782ms/step - loss: 0.3901 - accuracy: 0.8358 - val_loss: 0.3976 - val_accuracy: 0.8244
    Epoch 16/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.3849 - accuracy: 0.8438
    Epoch 00016: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 20s 784ms/step - loss: 0.3801 - accuracy: 0.8467 - val_loss: 0.4099 - val_accuracy: 0.8488
    Epoch 17/50
    26/26 [==============================] - 20s 785ms/step - loss: 0.3714 - accuracy: 0.8674 - val_loss: 0.3933 - val_accuracy: 0.8537
    Epoch 18/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.3599 - accuracy: 0.8687
    Epoch 00018: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 21s 800ms/step - loss: 0.3625 - accuracy: 0.8674 - val_loss: 0.3957 - val_accuracy: 0.8585
    Epoch 19/50
    26/26 [==============================] - 21s 800ms/step - loss: 0.3552 - accuracy: 0.8796 - val_loss: 0.3776 - val_accuracy: 0.8732
    Epoch 20/50
    26/26 [==============================] - 21s 816ms/step - loss: 0.3517 - accuracy: 0.8674 - val_loss: 0.3680 - val_accuracy: 0.8829
    Epoch 21/50
    26/26 [==============================] - 20s 788ms/step - loss: 0.3437 - accuracy: 0.8893 - val_loss: 0.3549 - val_accuracy: 0.8488
    Epoch 22/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.3409 - accuracy: 0.8737
    Epoch 00022: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 21s 816ms/step - loss: 0.3383 - accuracy: 0.8747 - val_loss: 0.3724 - val_accuracy: 0.8829
    Epoch 23/50
    26/26 [==============================] - 21s 820ms/step - loss: 0.3333 - accuracy: 0.8942 - val_loss: 0.3431 - val_accuracy: 0.8585
    Epoch 24/50
    26/26 [==============================] - 21s 814ms/step - loss: 0.3288 - accuracy: 0.8832 - val_loss: 0.3375 - val_accuracy: 0.8537
    Epoch 25/50
    26/26 [==============================] - 20s 786ms/step - loss: 0.3207 - accuracy: 0.8881 - val_loss: 0.3365 - val_accuracy: 0.8829
    Epoch 26/50
    26/26 [==============================] - 20s 784ms/step - loss: 0.3141 - accuracy: 0.8978 - val_loss: 0.3328 - val_accuracy: 0.8927
    Epoch 27/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.3108 - accuracy: 0.9013
    Epoch 00027: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 21s 800ms/step - loss: 0.3084 - accuracy: 0.9027 - val_loss: 0.3376 - val_accuracy: 0.9024
    Epoch 28/50
    26/26 [==============================] - 20s 781ms/step - loss: 0.3041 - accuracy: 0.9039 - val_loss: 0.3142 - val_accuracy: 0.8780
    Epoch 29/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.2942 - accuracy: 0.9000
    Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 19s 750ms/step - loss: 0.2967 - accuracy: 0.8990 - val_loss: 0.3159 - val_accuracy: 0.8976
    Epoch 30/50
    26/26 [==============================] - 21s 796ms/step - loss: 0.2902 - accuracy: 0.9051 - val_loss: 0.3100 - val_accuracy: 0.8976
    Epoch 31/50
    26/26 [==============================] - 20s 754ms/step - loss: 0.2859 - accuracy: 0.8990 - val_loss: 0.2977 - val_accuracy: 0.8878
    Epoch 32/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.2809 - accuracy: 0.9112
    Epoch 00032: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 20s 769ms/step - loss: 0.2773 - accuracy: 0.9124 - val_loss: 0.3508 - val_accuracy: 0.8829
    Epoch 33/50
    26/26 [==============================] - 21s 823ms/step - loss: 0.2780 - accuracy: 0.9100 - val_loss: 0.2884 - val_accuracy: 0.8878
    Epoch 34/50
    26/26 [==============================] - 20s 769ms/step - loss: 0.2696 - accuracy: 0.9234 - val_loss: 0.2811 - val_accuracy: 0.8976
    Epoch 35/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.2660 - accuracy: 0.9150
    Epoch 00035: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 20s 758ms/step - loss: 0.2628 - accuracy: 0.9161 - val_loss: 0.2932 - val_accuracy: 0.9122
    Epoch 36/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.2594 - accuracy: 0.9250
    Epoch 00036: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 20s 757ms/step - loss: 0.2589 - accuracy: 0.9246 - val_loss: 0.3366 - val_accuracy: 0.8829
    Epoch 37/50
    26/26 [==============================] - 21s 789ms/step - loss: 0.2549 - accuracy: 0.9270 - val_loss: 0.2710 - val_accuracy: 0.9171
    Epoch 38/50
    26/26 [==============================] - 20s 773ms/step - loss: 0.2498 - accuracy: 0.9197 - val_loss: 0.2669 - val_accuracy: 0.9268
    Epoch 39/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.2430 - accuracy: 0.9275
    Epoch 00039: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 20s 765ms/step - loss: 0.2455 - accuracy: 0.9282 - val_loss: 0.2930 - val_accuracy: 0.9024
    Epoch 40/50
    26/26 [==============================] - 20s 761ms/step - loss: 0.2375 - accuracy: 0.9294 - val_loss: 0.2521 - val_accuracy: 0.9024
    Epoch 41/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.2384 - accuracy: 0.9312
    Epoch 00041: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 20s 781ms/step - loss: 0.2412 - accuracy: 0.9282 - val_loss: 0.2545 - val_accuracy: 0.8976
    Epoch 42/50
    26/26 [==============================] - 20s 757ms/step - loss: 0.2262 - accuracy: 0.9453 - val_loss: 0.2426 - val_accuracy: 0.9317
    Epoch 43/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.2252 - accuracy: 0.9388
    Epoch 00043: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 20s 777ms/step - loss: 0.2228 - accuracy: 0.9404 - val_loss: 0.2525 - val_accuracy: 0.9268
    Epoch 44/50
    26/26 [==============================] - 20s 754ms/step - loss: 0.2160 - accuracy: 0.9428 - val_loss: 0.2362 - val_accuracy: 0.9024
    Epoch 45/50
    26/26 [==============================] - 20s 761ms/step - loss: 0.2170 - accuracy: 0.9392 - val_loss: 0.2292 - val_accuracy: 0.9268
    Epoch 46/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.2054 - accuracy: 0.9500
    Epoch 00046: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 20s 765ms/step - loss: 0.2045 - accuracy: 0.9501 - val_loss: 0.2759 - val_accuracy: 0.9073
    Epoch 47/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.2067 - accuracy: 0.9425
    Epoch 00047: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 20s 761ms/step - loss: 0.2032 - accuracy: 0.9440 - val_loss: 0.2533 - val_accuracy: 0.9268
    Epoch 48/50
    26/26 [==============================] - 20s 785ms/step - loss: 0.1993 - accuracy: 0.9501 - val_loss: 0.2170 - val_accuracy: 0.9317
    Epoch 49/50
    25/26 [===========================>..] - ETA: 0s - loss: 0.1953 - accuracy: 0.9475
    Epoch 00049: ReduceLROnPlateau reducing learning rate to 0.001.
    26/26 [==============================] - 20s 758ms/step - loss: 0.1933 - accuracy: 0.9489 - val_loss: 0.3325 - val_accuracy: 0.8927
    Epoch 50/50
    26/26 [==============================] - 20s 758ms/step - loss: 0.1887 - accuracy: 0.9489 - val_loss: 0.2121 - val_accuracy: 0.9415





    <tensorflow.python.keras.callbacks.History at 0x7fc57901c650>




```python
%tensorboard --logdir log_dir
```


    Reusing TensorBoard on port 6007 (pid 4870), started 0:17:01 ago. (Use '!kill 4870' to kill it.)




<iframe id="tensorboard-frame-c0f7f177bac7d4bb" width="100%" height="800" frameborder="0">
</iframe>
<script>
  (function() {
    const frame = document.getElementById("tensorboard-frame-c0f7f177bac7d4bb");
    const url = new URL("/", window.location);
    url.port = 6007;
    frame.src = url;
  })();
</script>


