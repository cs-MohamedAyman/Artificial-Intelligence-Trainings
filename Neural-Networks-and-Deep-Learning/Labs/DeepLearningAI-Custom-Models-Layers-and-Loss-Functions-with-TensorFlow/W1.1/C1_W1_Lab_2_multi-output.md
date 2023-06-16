# Ungraded Lab: Build a Multi-output Model

In this lab, we'll show how you can build models with more than one output. The dataset we will be working on is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency). It is an Energy Efficiency dataset which uses the bulding features (e.g. wall area, roof area) as inputs and has two outputs: Cooling Load and Heating Load. Let's see how we can build a model to train on this data.

## Imports


```python
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
```

## Utilities

We define a few utilities for data conversion and visualization to make our code more neat.


```python
def format_output(data):
    y1 = data.pop('Y1')
    y1 = np.array(y1)
    y2 = data.pop('Y2')
    y2 = np.array(y2)
    return y1, y2


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


def plot_diff(y_true, y_pred, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
    plt.show()
```

## Prepare the Data

We download the dataset and format it for training.


```python
# Specify data URI
URI = './data/ENB2012_data.xlsx'

# Use pandas excel reader
df = pd.read_excel(URI)
df = df.sample(frac=1).reset_index(drop=True)

# Split the data into train and test with 80 train / 20 test
train, test = train_test_split(df, test_size=0.2)
train_stats = train.describe()

# Get Y1 and Y2 as the 2 outputs and format them as np arrays
train_stats.pop('Y1')
train_stats.pop('Y2')
train_stats = train_stats.transpose()
train_Y = format_output(train)
test_Y = format_output(test)

# Normalize the training and test data
norm_train_X = norm(train)
norm_test_X = norm(test)
```

## Build the Model

Here is how we'll build the model using the functional syntax. Notice that we can specify a list of outputs (i.e. `[y1_output, y2_output]`) when we instantiate the `Model()` class.


```python
# Define model layers.
input_layer = Input(shape=(len(train .columns),))
first_dense = Dense(units='128', activation='relu')(input_layer)
second_dense = Dense(units='128', activation='relu')(first_dense)

# Y1 output will be fed directly from the second dense
y1_output = Dense(units='1', name='y1_output')(second_dense)
third_dense = Dense(units='64', activation='relu')(second_dense)

# Y2 output will come via the third dense
y2_output = Dense(units='1', name='y2_output')(third_dense)

# Define the model with the input layer and a list of output layers
model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

print(model.summary())
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_1 (InputLayer)            [(None, 8)]          0
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 128)          1152        input_1[0][0]
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 128)          16512       dense[0][0]
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 64)           8256        dense_1[0][0]
    __________________________________________________________________________________________________
    y1_output (Dense)               (None, 1)            129         dense_1[0][0]
    __________________________________________________________________________________________________
    y2_output (Dense)               (None, 1)            65          dense_2[0][0]
    ==================================================================================================
    Total params: 26,114
    Trainable params: 26,114
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None


## Configure parameters

We specify the optimizer as well as the loss and metrics for each output.


```python
# Specify the optimizer, and compile the model with loss functions for both outputs
optimizer = tf.keras.optimizers.SGD(lr=0.001)
model.compile(optimizer=optimizer,
              loss={'y1_output': 'mse', 'y2_output': 'mse'},
              metrics={'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                       'y2_output': tf.keras.metrics.RootMeanSquaredError()})
```

## Train the Model


```python
# Train the model for 500 epochs
history = model.fit(norm_train_X, train_Y,
                    epochs=500, batch_size=10, validation_data=(norm_test_X, test_Y))
```

    Train on 614 samples, validate on 154 samples
    Epoch 1/500
    614/614 [==============================] - 1s 950us/sample - loss: 237.8915 - y1_output_loss: 111.5118 - y2_output_loss: 124.1414 - y1_output_root_mean_squared_error: 10.6106 - y2_output_root_mean_squared_error: 11.1941 - val_loss: 37.7240 - val_y1_output_loss: 16.0065 - val_y2_output_loss: 20.6094 - val_y1_output_root_mean_squared_error: 4.0677 - val_y2_output_root_mean_squared_error: 4.6020
    Epoch 2/500
    614/614 [==============================] - 0s 162us/sample - loss: 26.5016 - y1_output_loss: 10.4998 - y2_output_loss: 15.8086 - y1_output_root_mean_squared_error: 3.2556 - y2_output_root_mean_squared_error: 3.9879 - val_loss: 29.6273 - val_y1_output_loss: 12.7356 - val_y2_output_loss: 16.5970 - val_y1_output_root_mean_squared_error: 3.5903 - val_y2_output_root_mean_squared_error: 4.0911
    Epoch 3/500
    614/614 [==============================] - 0s 148us/sample - loss: 25.9230 - y1_output_loss: 9.8532 - y2_output_loss: 15.9298 - y1_output_root_mean_squared_error: 3.1463 - y2_output_root_mean_squared_error: 4.0029 - val_loss: 41.2080 - val_y1_output_loss: 12.6500 - val_y2_output_loss: 27.8041 - val_y1_output_root_mean_squared_error: 3.6091 - val_y2_output_root_mean_squared_error: 5.3087
    Epoch 4/500
    614/614 [==============================] - 0s 141us/sample - loss: 26.6826 - y1_output_loss: 9.9178 - y2_output_loss: 16.6017 - y1_output_root_mean_squared_error: 3.1565 - y2_output_root_mean_squared_error: 4.0889 - val_loss: 24.4576 - val_y1_output_loss: 10.2043 - val_y2_output_loss: 13.6043 - val_y1_output_root_mean_squared_error: 3.2341 - val_y2_output_root_mean_squared_error: 3.7414
    Epoch 5/500
    614/614 [==============================] - 0s 133us/sample - loss: 25.5290 - y1_output_loss: 9.0589 - y2_output_loss: 16.3812 - y1_output_root_mean_squared_error: 3.0148 - y2_output_root_mean_squared_error: 4.0546 - val_loss: 29.7412 - val_y1_output_loss: 10.1355 - val_y2_output_loss: 18.9563 - val_y1_output_root_mean_squared_error: 3.2250 - val_y2_output_root_mean_squared_error: 4.3978
    Epoch 6/500
    614/614 [==============================] - 0s 132us/sample - loss: 23.2809 - y1_output_loss: 8.8582 - y2_output_loss: 14.4251 - y1_output_root_mean_squared_error: 2.9688 - y2_output_root_mean_squared_error: 3.8036 - val_loss: 34.0977 - val_y1_output_loss: 11.2295 - val_y2_output_loss: 22.3665 - val_y1_output_root_mean_squared_error: 3.3930 - val_y2_output_root_mean_squared_error: 4.7524
    Epoch 7/500
    614/614 [==============================] - 0s 137us/sample - loss: 20.8362 - y1_output_loss: 8.4792 - y2_output_loss: 12.4101 - y1_output_root_mean_squared_error: 2.8971 - y2_output_root_mean_squared_error: 3.5275 - val_loss: 27.4719 - val_y1_output_loss: 10.4849 - val_y2_output_loss: 16.5407 - val_y1_output_root_mean_squared_error: 3.2682 - val_y2_output_root_mean_squared_error: 4.0977
    Epoch 8/500
    614/614 [==============================] - 0s 140us/sample - loss: 21.9250 - y1_output_loss: 8.2946 - y2_output_loss: 13.9670 - y1_output_root_mean_squared_error: 2.8786 - y2_output_root_mean_squared_error: 3.6931 - val_loss: 32.8670 - val_y1_output_loss: 10.8663 - val_y2_output_loss: 21.9988 - val_y1_output_root_mean_squared_error: 3.3055 - val_y2_output_root_mean_squared_error: 4.6841
    Epoch 9/500
    614/614 [==============================] - 0s 131us/sample - loss: 20.2990 - y1_output_loss: 7.8697 - y2_output_loss: 12.2450 - y1_output_root_mean_squared_error: 2.8187 - y2_output_root_mean_squared_error: 3.5148 - val_loss: 21.5097 - val_y1_output_loss: 9.3873 - val_y2_output_loss: 11.6250 - val_y1_output_root_mean_squared_error: 3.1027 - val_y2_output_root_mean_squared_error: 3.4472
    Epoch 10/500
    614/614 [==============================] - 0s 131us/sample - loss: 20.3889 - y1_output_loss: 7.8846 - y2_output_loss: 12.3493 - y1_output_root_mean_squared_error: 2.8160 - y2_output_root_mean_squared_error: 3.5298 - val_loss: 22.0462 - val_y1_output_loss: 9.3005 - val_y2_output_loss: 12.3548 - val_y1_output_root_mean_squared_error: 3.0823 - val_y2_output_root_mean_squared_error: 3.5420
    Epoch 11/500
    614/614 [==============================] - 0s 131us/sample - loss: 19.9037 - y1_output_loss: 7.7333 - y2_output_loss: 12.1263 - y1_output_root_mean_squared_error: 2.7761 - y2_output_root_mean_squared_error: 3.4924 - val_loss: 23.1453 - val_y1_output_loss: 9.7570 - val_y2_output_loss: 12.8417 - val_y1_output_root_mean_squared_error: 3.1665 - val_y2_output_root_mean_squared_error: 3.6220
    Epoch 12/500
    614/614 [==============================] - 0s 132us/sample - loss: 21.4352 - y1_output_loss: 8.0548 - y2_output_loss: 13.2294 - y1_output_root_mean_squared_error: 2.8496 - y2_output_root_mean_squared_error: 3.6490 - val_loss: 21.9632 - val_y1_output_loss: 9.2202 - val_y2_output_loss: 12.4362 - val_y1_output_root_mean_squared_error: 3.0695 - val_y2_output_root_mean_squared_error: 3.5414
    Epoch 13/500
    614/614 [==============================] - 0s 131us/sample - loss: 19.7328 - y1_output_loss: 7.3169 - y2_output_loss: 12.3461 - y1_output_root_mean_squared_error: 2.7132 - y2_output_root_mean_squared_error: 3.5173 - val_loss: 21.9941 - val_y1_output_loss: 10.2303 - val_y2_output_loss: 11.3067 - val_y1_output_root_mean_squared_error: 3.2312 - val_y2_output_root_mean_squared_error: 3.3991
    Epoch 14/500
    614/614 [==============================] - 0s 141us/sample - loss: 19.2220 - y1_output_loss: 7.1591 - y2_output_loss: 11.8969 - y1_output_root_mean_squared_error: 2.6879 - y2_output_root_mean_squared_error: 3.4637 - val_loss: 19.2671 - val_y1_output_loss: 8.3084 - val_y2_output_loss: 10.4414 - val_y1_output_root_mean_squared_error: 2.9316 - val_y2_output_root_mean_squared_error: 3.2669
    Epoch 15/500
    614/614 [==============================] - 0s 143us/sample - loss: 19.9567 - y1_output_loss: 7.5775 - y2_output_loss: 12.2636 - y1_output_root_mean_squared_error: 2.7546 - y2_output_root_mean_squared_error: 3.5170 - val_loss: 20.5039 - val_y1_output_loss: 8.5980 - val_y2_output_loss: 11.4649 - val_y1_output_root_mean_squared_error: 2.9714 - val_y2_output_root_mean_squared_error: 3.4169
    Epoch 16/500
    614/614 [==============================] - 0s 143us/sample - loss: 15.6734 - y1_output_loss: 6.1412 - y2_output_loss: 9.5403 - y1_output_root_mean_squared_error: 2.4748 - y2_output_root_mean_squared_error: 3.0901 - val_loss: 33.8430 - val_y1_output_loss: 10.5806 - val_y2_output_loss: 23.2360 - val_y1_output_root_mean_squared_error: 3.2768 - val_y2_output_root_mean_squared_error: 4.8068
    Epoch 17/500
    614/614 [==============================] - 0s 132us/sample - loss: 17.9742 - y1_output_loss: 6.8557 - y2_output_loss: 11.1356 - y1_output_root_mean_squared_error: 2.6187 - y2_output_root_mean_squared_error: 3.3342 - val_loss: 21.5014 - val_y1_output_loss: 9.1826 - val_y2_output_loss: 11.9812 - val_y1_output_root_mean_squared_error: 3.0644 - val_y2_output_root_mean_squared_error: 3.4801
    Epoch 18/500
    614/614 [==============================] - 0s 130us/sample - loss: 15.2777 - y1_output_loss: 6.0309 - y2_output_loss: 9.1712 - y1_output_root_mean_squared_error: 2.4608 - y2_output_root_mean_squared_error: 3.0368 - val_loss: 15.8824 - val_y1_output_loss: 6.8318 - val_y2_output_loss: 8.7080 - val_y1_output_root_mean_squared_error: 2.6442 - val_y2_output_root_mean_squared_error: 2.9817
    Epoch 19/500
    614/614 [==============================] - 0s 138us/sample - loss: 17.7472 - y1_output_loss: 6.7009 - y2_output_loss: 11.0357 - y1_output_root_mean_squared_error: 2.5846 - y2_output_root_mean_squared_error: 3.3267 - val_loss: 19.4894 - val_y1_output_loss: 7.8689 - val_y2_output_loss: 11.1628 - val_y1_output_root_mean_squared_error: 2.8469 - val_y2_output_root_mean_squared_error: 3.3741
    Epoch 20/500
    614/614 [==============================] - 0s 132us/sample - loss: 15.2773 - y1_output_loss: 5.7175 - y2_output_loss: 9.7339 - y1_output_root_mean_squared_error: 2.3844 - y2_output_root_mean_squared_error: 3.0971 - val_loss: 40.5794 - val_y1_output_loss: 10.1509 - val_y2_output_loss: 29.8951 - val_y1_output_root_mean_squared_error: 3.2230 - val_y2_output_root_mean_squared_error: 5.4947
    Epoch 21/500
    614/614 [==============================] - 0s 132us/sample - loss: 15.1425 - y1_output_loss: 5.5240 - y2_output_loss: 9.8068 - y1_output_root_mean_squared_error: 2.3509 - y2_output_root_mean_squared_error: 3.1010 - val_loss: 24.3496 - val_y1_output_loss: 6.6967 - val_y2_output_loss: 17.2000 - val_y1_output_root_mean_squared_error: 2.6303 - val_y2_output_root_mean_squared_error: 4.1751
    Epoch 22/500
    614/614 [==============================] - 0s 131us/sample - loss: 15.6936 - y1_output_loss: 5.4642 - y2_output_loss: 10.0819 - y1_output_root_mean_squared_error: 2.3488 - y2_output_root_mean_squared_error: 3.1901 - val_loss: 14.7024 - val_y1_output_loss: 5.8010 - val_y2_output_loss: 8.6610 - val_y1_output_root_mean_squared_error: 2.4450 - val_y2_output_root_mean_squared_error: 2.9537
    Epoch 23/500
    614/614 [==============================] - 0s 137us/sample - loss: 12.4393 - y1_output_loss: 5.0115 - y2_output_loss: 7.5683 - y1_output_root_mean_squared_error: 2.2346 - y2_output_root_mean_squared_error: 2.7287 - val_loss: 16.3736 - val_y1_output_loss: 6.3528 - val_y2_output_loss: 9.6703 - val_y1_output_root_mean_squared_error: 2.5529 - val_y2_output_root_mean_squared_error: 3.1395
    Epoch 24/500
    614/614 [==============================] - 0s 140us/sample - loss: 14.0673 - y1_output_loss: 5.2808 - y2_output_loss: 8.6789 - y1_output_root_mean_squared_error: 2.3086 - y2_output_root_mean_squared_error: 2.9560 - val_loss: 14.1108 - val_y1_output_loss: 6.0291 - val_y2_output_loss: 7.7899 - val_y1_output_root_mean_squared_error: 2.4931 - val_y2_output_root_mean_squared_error: 2.8098
    Epoch 25/500
    614/614 [==============================] - 0s 140us/sample - loss: 14.3004 - y1_output_loss: 5.1612 - y2_output_loss: 9.0049 - y1_output_root_mean_squared_error: 2.2826 - y2_output_root_mean_squared_error: 3.0150 - val_loss: 13.0485 - val_y1_output_loss: 5.4062 - val_y2_output_loss: 7.2936 - val_y1_output_root_mean_squared_error: 2.3668 - val_y2_output_root_mean_squared_error: 2.7288
    Epoch 26/500
    614/614 [==============================] - 0s 138us/sample - loss: 13.2762 - y1_output_loss: 4.9794 - y2_output_loss: 8.3905 - y1_output_root_mean_squared_error: 2.2079 - y2_output_root_mean_squared_error: 2.8985 - val_loss: 19.9519 - val_y1_output_loss: 6.8038 - val_y2_output_loss: 13.0216 - val_y1_output_root_mean_squared_error: 2.6387 - val_y2_output_root_mean_squared_error: 3.6041
    Epoch 27/500
    614/614 [==============================] - 0s 140us/sample - loss: 12.6419 - y1_output_loss: 4.5751 - y2_output_loss: 7.9778 - y1_output_root_mean_squared_error: 2.1466 - y2_output_root_mean_squared_error: 2.8344 - val_loss: 12.8147 - val_y1_output_loss: 5.2030 - val_y2_output_loss: 7.4058 - val_y1_output_root_mean_squared_error: 2.3162 - val_y2_output_root_mean_squared_error: 2.7294
    Epoch 28/500
    614/614 [==============================] - 0s 141us/sample - loss: 12.7061 - y1_output_loss: 4.6769 - y2_output_loss: 8.1030 - y1_output_root_mean_squared_error: 2.1626 - y2_output_root_mean_squared_error: 2.8336 - val_loss: 32.8134 - val_y1_output_loss: 8.0424 - val_y2_output_loss: 24.0996 - val_y1_output_root_mean_squared_error: 2.8811 - val_y2_output_root_mean_squared_error: 4.9510
    Epoch 29/500
    614/614 [==============================] - 0s 137us/sample - loss: 11.2208 - y1_output_loss: 4.2490 - y2_output_loss: 6.9297 - y1_output_root_mean_squared_error: 2.0693 - y2_output_root_mean_squared_error: 2.6341 - val_loss: 13.4968 - val_y1_output_loss: 5.0693 - val_y2_output_loss: 8.0718 - val_y1_output_root_mean_squared_error: 2.2913 - val_y2_output_root_mean_squared_error: 2.8717
    Epoch 30/500
    614/614 [==============================] - 0s 132us/sample - loss: 10.1197 - y1_output_loss: 3.9710 - y2_output_loss: 6.0754 - y1_output_root_mean_squared_error: 1.9983 - y2_output_root_mean_squared_error: 2.4752 - val_loss: 11.5494 - val_y1_output_loss: 4.7391 - val_y2_output_loss: 6.5247 - val_y1_output_root_mean_squared_error: 2.2145 - val_y2_output_root_mean_squared_error: 2.5779
    Epoch 31/500
    614/614 [==============================] - 0s 132us/sample - loss: 11.6539 - y1_output_loss: 4.2451 - y2_output_loss: 7.4169 - y1_output_root_mean_squared_error: 2.0595 - y2_output_root_mean_squared_error: 2.7226 - val_loss: 16.3176 - val_y1_output_loss: 6.0623 - val_y2_output_loss: 10.0081 - val_y1_output_root_mean_squared_error: 2.4888 - val_y2_output_root_mean_squared_error: 3.1817
    Epoch 32/500
    614/614 [==============================] - 0s 130us/sample - loss: 11.0092 - y1_output_loss: 4.1258 - y2_output_loss: 7.0125 - y1_output_root_mean_squared_error: 2.0161 - y2_output_root_mean_squared_error: 2.6352 - val_loss: 25.2745 - val_y1_output_loss: 7.5367 - val_y2_output_loss: 17.4392 - val_y1_output_root_mean_squared_error: 2.7720 - val_y2_output_root_mean_squared_error: 4.1941
    Epoch 33/500
    614/614 [==============================] - 0s 132us/sample - loss: 10.5722 - y1_output_loss: 3.9115 - y2_output_loss: 6.6180 - y1_output_root_mean_squared_error: 1.9868 - y2_output_root_mean_squared_error: 2.5739 - val_loss: 24.9634 - val_y1_output_loss: 7.2896 - val_y2_output_loss: 17.6441 - val_y1_output_root_mean_squared_error: 2.7130 - val_y2_output_root_mean_squared_error: 4.1956
    Epoch 34/500
    614/614 [==============================] - 0s 139us/sample - loss: 10.3746 - y1_output_loss: 3.7468 - y2_output_loss: 6.7855 - y1_output_root_mean_squared_error: 1.9397 - y2_output_root_mean_squared_error: 2.5715 - val_loss: 84.2036 - val_y1_output_loss: 13.7190 - val_y2_output_loss: 70.8543 - val_y1_output_root_mean_squared_error: 3.7060 - val_y2_output_root_mean_squared_error: 8.3946
    Epoch 35/500
    614/614 [==============================] - 0s 140us/sample - loss: 10.6561 - y1_output_loss: 3.6687 - y2_output_loss: 7.0476 - y1_output_root_mean_squared_error: 1.9172 - y2_output_root_mean_squared_error: 2.6421 - val_loss: 10.5519 - val_y1_output_loss: 4.1163 - val_y2_output_loss: 6.2927 - val_y1_output_root_mean_squared_error: 2.0533 - val_y2_output_root_mean_squared_error: 2.5171
    Epoch 36/500
    614/614 [==============================] - 0s 139us/sample - loss: 9.5169 - y1_output_loss: 3.1544 - y2_output_loss: 6.3103 - y1_output_root_mean_squared_error: 1.7827 - y2_output_root_mean_squared_error: 2.5177 - val_loss: 16.8315 - val_y1_output_loss: 4.8144 - val_y2_output_loss: 11.6759 - val_y1_output_root_mean_squared_error: 2.2291 - val_y2_output_root_mean_squared_error: 3.4442
    Epoch 37/500
    614/614 [==============================] - 0s 137us/sample - loss: 9.9858 - y1_output_loss: 3.5550 - y2_output_loss: 6.4513 - y1_output_root_mean_squared_error: 1.8901 - y2_output_root_mean_squared_error: 2.5325 - val_loss: 10.1678 - val_y1_output_loss: 3.5649 - val_y2_output_loss: 6.3685 - val_y1_output_root_mean_squared_error: 1.9171 - val_y2_output_root_mean_squared_error: 2.5481
    Epoch 38/500
    614/614 [==============================] - 0s 130us/sample - loss: 8.8757 - y1_output_loss: 3.0644 - y2_output_loss: 5.7323 - y1_output_root_mean_squared_error: 1.7585 - y2_output_root_mean_squared_error: 2.4049 - val_loss: 9.1133 - val_y1_output_loss: 3.4843 - val_y2_output_loss: 5.5384 - val_y1_output_root_mean_squared_error: 1.8870 - val_y2_output_root_mean_squared_error: 2.3564
    Epoch 39/500
    614/614 [==============================] - 0s 129us/sample - loss: 7.6853 - y1_output_loss: 2.8064 - y2_output_loss: 4.8676 - y1_output_root_mean_squared_error: 1.6822 - y2_output_root_mean_squared_error: 2.2035 - val_loss: 9.9117 - val_y1_output_loss: 3.6723 - val_y2_output_loss: 5.9925 - val_y1_output_root_mean_squared_error: 1.9499 - val_y2_output_root_mean_squared_error: 2.4717
    Epoch 40/500
    614/614 [==============================] - 0s 127us/sample - loss: 7.5416 - y1_output_loss: 2.6598 - y2_output_loss: 4.9153 - y1_output_root_mean_squared_error: 1.6259 - y2_output_root_mean_squared_error: 2.2132 - val_loss: 14.7302 - val_y1_output_loss: 5.0536 - val_y2_output_loss: 9.3504 - val_y1_output_root_mean_squared_error: 2.2828 - val_y2_output_root_mean_squared_error: 3.0853
    Epoch 41/500
    614/614 [==============================] - 0s 131us/sample - loss: 7.7524 - y1_output_loss: 2.6683 - y2_output_loss: 5.0639 - y1_output_root_mean_squared_error: 1.6376 - y2_output_root_mean_squared_error: 2.2518 - val_loss: 8.3376 - val_y1_output_loss: 2.9197 - val_y2_output_loss: 5.2964 - val_y1_output_root_mean_squared_error: 1.7336 - val_y2_output_root_mean_squared_error: 2.3091
    Epoch 42/500
    614/614 [==============================] - 0s 130us/sample - loss: 7.2971 - y1_output_loss: 2.6251 - y2_output_loss: 4.6803 - y1_output_root_mean_squared_error: 1.6185 - y2_output_root_mean_squared_error: 2.1628 - val_loss: 8.0041 - val_y1_output_loss: 2.8007 - val_y2_output_loss: 5.0336 - val_y1_output_root_mean_squared_error: 1.7032 - val_y2_output_root_mean_squared_error: 2.2591
    Epoch 43/500
    614/614 [==============================] - 0s 130us/sample - loss: 9.4482 - y1_output_loss: 3.0933 - y2_output_loss: 6.3342 - y1_output_root_mean_squared_error: 1.7652 - y2_output_root_mean_squared_error: 2.5164 - val_loss: 8.3575 - val_y1_output_loss: 2.9654 - val_y2_output_loss: 5.2732 - val_y1_output_root_mean_squared_error: 1.7434 - val_y2_output_root_mean_squared_error: 2.3061
    Epoch 44/500
    614/614 [==============================] - 0s 130us/sample - loss: 10.1021 - y1_output_loss: 3.4951 - y2_output_loss: 6.5884 - y1_output_root_mean_squared_error: 1.8751 - y2_output_root_mean_squared_error: 2.5664 - val_loss: 15.0536 - val_y1_output_loss: 3.8170 - val_y2_output_loss: 11.2229 - val_y1_output_root_mean_squared_error: 1.9776 - val_y2_output_root_mean_squared_error: 3.3381
    Epoch 45/500
    614/614 [==============================] - 0s 132us/sample - loss: 12.3573 - y1_output_loss: 3.7721 - y2_output_loss: 8.6368 - y1_output_root_mean_squared_error: 1.9325 - y2_output_root_mean_squared_error: 2.9364 - val_loss: 26.4975 - val_y1_output_loss: 7.4209 - val_y2_output_loss: 18.7742 - val_y1_output_root_mean_squared_error: 2.7617 - val_y2_output_root_mean_squared_error: 4.3440
    Epoch 46/500
    614/614 [==============================] - 0s 131us/sample - loss: 9.6181 - y1_output_loss: 3.3322 - y2_output_loss: 6.3546 - y1_output_root_mean_squared_error: 1.8028 - y2_output_root_mean_squared_error: 2.5235 - val_loss: 22.1681 - val_y1_output_loss: 5.6839 - val_y2_output_loss: 16.8504 - val_y1_output_root_mean_squared_error: 2.3782 - val_y2_output_root_mean_squared_error: 4.0636
    Epoch 47/500
    614/614 [==============================] - 0s 129us/sample - loss: 8.4161 - y1_output_loss: 2.9319 - y2_output_loss: 5.6706 - y1_output_root_mean_squared_error: 1.7070 - y2_output_root_mean_squared_error: 2.3456 - val_loss: 14.5839 - val_y1_output_loss: 4.2318 - val_y2_output_loss: 10.4772 - val_y1_output_root_mean_squared_error: 2.0672 - val_y2_output_root_mean_squared_error: 3.2110
    Epoch 48/500
    614/614 [==============================] - 0s 129us/sample - loss: 9.4530 - y1_output_loss: 2.9082 - y2_output_loss: 6.5078 - y1_output_root_mean_squared_error: 1.7043 - y2_output_root_mean_squared_error: 2.5590 - val_loss: 13.8313 - val_y1_output_loss: 3.5586 - val_y2_output_loss: 10.1966 - val_y1_output_root_mean_squared_error: 1.9096 - val_y2_output_root_mean_squared_error: 3.1914
    Epoch 49/500
    614/614 [==============================] - 0s 128us/sample - loss: 8.8975 - y1_output_loss: 2.7610 - y2_output_loss: 6.0551 - y1_output_root_mean_squared_error: 1.6687 - y2_output_root_mean_squared_error: 2.4724 - val_loss: 10.5295 - val_y1_output_loss: 2.7283 - val_y2_output_loss: 7.7385 - val_y1_output_root_mean_squared_error: 1.6747 - val_y2_output_root_mean_squared_error: 2.7794
    Epoch 50/500
    614/614 [==============================] - 0s 129us/sample - loss: 7.8998 - y1_output_loss: 2.7167 - y2_output_loss: 5.2067 - y1_output_root_mean_squared_error: 1.6391 - y2_output_root_mean_squared_error: 2.2832 - val_loss: 9.1386 - val_y1_output_loss: 2.7926 - val_y2_output_loss: 6.4219 - val_y1_output_root_mean_squared_error: 1.6830 - val_y2_output_root_mean_squared_error: 2.5112
    Epoch 51/500
    614/614 [==============================] - 0s 131us/sample - loss: 7.8666 - y1_output_loss: 2.6561 - y2_output_loss: 5.6697 - y1_output_root_mean_squared_error: 1.6012 - y2_output_root_mean_squared_error: 2.3028 - val_loss: 43.0557 - val_y1_output_loss: 8.9650 - val_y2_output_loss: 33.3210 - val_y1_output_root_mean_squared_error: 3.0348 - val_y2_output_root_mean_squared_error: 5.8177
    Epoch 52/500
    614/614 [==============================] - 0s 130us/sample - loss: 12.8199 - y1_output_loss: 4.0424 - y2_output_loss: 8.7957 - y1_output_root_mean_squared_error: 2.0097 - y2_output_root_mean_squared_error: 2.9633 - val_loss: 16.9210 - val_y1_output_loss: 3.7921 - val_y2_output_loss: 13.3023 - val_y1_output_root_mean_squared_error: 1.9534 - val_y2_output_root_mean_squared_error: 3.6201
    Epoch 53/500
    614/614 [==============================] - 0s 130us/sample - loss: 6.0578 - y1_output_loss: 1.9579 - y2_output_loss: 4.0462 - y1_output_root_mean_squared_error: 1.4058 - y2_output_root_mean_squared_error: 2.0203 - val_loss: 8.4333 - val_y1_output_loss: 2.6034 - val_y2_output_loss: 5.7328 - val_y1_output_root_mean_squared_error: 1.6279 - val_y2_output_root_mean_squared_error: 2.4048
    Epoch 54/500
    614/614 [==============================] - 0s 131us/sample - loss: 5.4121 - y1_output_loss: 1.6417 - y2_output_loss: 3.7652 - y1_output_root_mean_squared_error: 1.2869 - y2_output_root_mean_squared_error: 1.9380 - val_loss: 7.9497 - val_y1_output_loss: 2.3606 - val_y2_output_loss: 5.4988 - val_y1_output_root_mean_squared_error: 1.5527 - val_y2_output_root_mean_squared_error: 2.3535
    Epoch 55/500
    614/614 [==============================] - 0s 131us/sample - loss: 5.6716 - y1_output_loss: 1.7385 - y2_output_loss: 4.0546 - y1_output_root_mean_squared_error: 1.3220 - y2_output_root_mean_squared_error: 1.9809 - val_loss: 8.5882 - val_y1_output_loss: 2.4679 - val_y2_output_loss: 5.9498 - val_y1_output_root_mean_squared_error: 1.5907 - val_y2_output_root_mean_squared_error: 2.4613
    Epoch 56/500
    614/614 [==============================] - 0s 131us/sample - loss: 6.5177 - y1_output_loss: 1.9123 - y2_output_loss: 4.5528 - y1_output_root_mean_squared_error: 1.3883 - y2_output_root_mean_squared_error: 2.1425 - val_loss: 6.1958 - val_y1_output_loss: 2.0050 - val_y2_output_loss: 4.0729 - val_y1_output_root_mean_squared_error: 1.4395 - val_y2_output_root_mean_squared_error: 2.0306
    Epoch 57/500
    614/614 [==============================] - 0s 130us/sample - loss: 5.1883 - y1_output_loss: 1.6099 - y2_output_loss: 3.5351 - y1_output_root_mean_squared_error: 1.2732 - y2_output_root_mean_squared_error: 1.8887 - val_loss: 6.5854 - val_y1_output_loss: 1.8990 - val_y2_output_loss: 4.5819 - val_y1_output_root_mean_squared_error: 1.4003 - val_y2_output_root_mean_squared_error: 2.1505
    Epoch 58/500
    614/614 [==============================] - 0s 131us/sample - loss: 9.8717 - y1_output_loss: 2.9589 - y2_output_loss: 6.8285 - y1_output_root_mean_squared_error: 1.7282 - y2_output_root_mean_squared_error: 2.6239 - val_loss: 7.3061 - val_y1_output_loss: 2.2910 - val_y2_output_loss: 4.8930 - val_y1_output_root_mean_squared_error: 1.5368 - val_y2_output_root_mean_squared_error: 2.2236
    Epoch 59/500
    614/614 [==============================] - 0s 131us/sample - loss: 6.2381 - y1_output_loss: 1.8468 - y2_output_loss: 4.3686 - y1_output_root_mean_squared_error: 1.3599 - y2_output_root_mean_squared_error: 2.0949 - val_loss: 6.3988 - val_y1_output_loss: 1.7996 - val_y2_output_loss: 4.4221 - val_y1_output_root_mean_squared_error: 1.3663 - val_y2_output_root_mean_squared_error: 2.1289
    Epoch 60/500
    614/614 [==============================] - 0s 129us/sample - loss: 5.1163 - y1_output_loss: 1.6449 - y2_output_loss: 3.4694 - y1_output_root_mean_squared_error: 1.2852 - y2_output_root_mean_squared_error: 1.8613 - val_loss: 11.4479 - val_y1_output_loss: 4.8973 - val_y2_output_loss: 6.3176 - val_y1_output_root_mean_squared_error: 2.2373 - val_y2_output_root_mean_squared_error: 2.5382
    Epoch 61/500
    614/614 [==============================] - 0s 130us/sample - loss: 5.9284 - y1_output_loss: 1.7894 - y2_output_loss: 4.1375 - y1_output_root_mean_squared_error: 1.3393 - y2_output_root_mean_squared_error: 2.0334 - val_loss: 8.3133 - val_y1_output_loss: 2.2175 - val_y2_output_loss: 5.9056 - val_y1_output_root_mean_squared_error: 1.5145 - val_y2_output_root_mean_squared_error: 2.4535
    Epoch 62/500
    614/614 [==============================] - 0s 134us/sample - loss: 6.7040 - y1_output_loss: 2.2062 - y2_output_loss: 4.5044 - y1_output_root_mean_squared_error: 1.4913 - y2_output_root_mean_squared_error: 2.1166 - val_loss: 6.0566 - val_y1_output_loss: 1.8313 - val_y2_output_loss: 4.0869 - val_y1_output_root_mean_squared_error: 1.3761 - val_y2_output_root_mean_squared_error: 2.0403
    Epoch 63/500
    614/614 [==============================] - 0s 131us/sample - loss: 6.5562 - y1_output_loss: 1.6603 - y2_output_loss: 4.8359 - y1_output_root_mean_squared_error: 1.2942 - y2_output_root_mean_squared_error: 2.2094 - val_loss: 5.0665 - val_y1_output_loss: 1.4187 - val_y2_output_loss: 3.5376 - val_y1_output_root_mean_squared_error: 1.2094 - val_y2_output_root_mean_squared_error: 1.8984
    Epoch 64/500
    614/614 [==============================] - 0s 130us/sample - loss: 4.1777 - y1_output_loss: 1.1851 - y2_output_loss: 3.0650 - y1_output_root_mean_squared_error: 1.0904 - y2_output_root_mean_squared_error: 1.7288 - val_loss: 9.1101 - val_y1_output_loss: 1.8834 - val_y2_output_loss: 7.0261 - val_y1_output_root_mean_squared_error: 1.3890 - val_y2_output_root_mean_squared_error: 2.6797
    Epoch 65/500
    614/614 [==============================] - 0s 139us/sample - loss: 4.1986 - y1_output_loss: 1.2111 - y2_output_loss: 3.0229 - y1_output_root_mean_squared_error: 1.1034 - y2_output_root_mean_squared_error: 1.7266 - val_loss: 6.8212 - val_y1_output_loss: 1.7415 - val_y2_output_loss: 5.0112 - val_y1_output_root_mean_squared_error: 1.3267 - val_y2_output_root_mean_squared_error: 2.2497
    Epoch 66/500
    614/614 [==============================] - 0s 139us/sample - loss: 4.1745 - y1_output_loss: 1.1390 - y2_output_loss: 3.0932 - y1_output_root_mean_squared_error: 1.0589 - y2_output_root_mean_squared_error: 1.7473 - val_loss: 5.9974 - val_y1_output_loss: 1.4774 - val_y2_output_loss: 4.4071 - val_y1_output_root_mean_squared_error: 1.2363 - val_y2_output_root_mean_squared_error: 2.1140
    Epoch 67/500
    614/614 [==============================] - 0s 134us/sample - loss: 3.9795 - y1_output_loss: 1.1619 - y2_output_loss: 2.8257 - y1_output_root_mean_squared_error: 1.0811 - y2_output_root_mean_squared_error: 1.6765 - val_loss: 7.6839 - val_y1_output_loss: 1.5569 - val_y2_output_loss: 6.0490 - val_y1_output_root_mean_squared_error: 1.2587 - val_y2_output_root_mean_squared_error: 2.4697
    Epoch 68/500
    614/614 [==============================] - 0s 131us/sample - loss: 5.4101 - y1_output_loss: 1.5464 - y2_output_loss: 3.8217 - y1_output_root_mean_squared_error: 1.2474 - y2_output_root_mean_squared_error: 1.9632 - val_loss: 5.5725 - val_y1_output_loss: 1.7447 - val_y2_output_loss: 3.7299 - val_y1_output_root_mean_squared_error: 1.3405 - val_y2_output_root_mean_squared_error: 1.9431
    Epoch 69/500
    614/614 [==============================] - 0s 135us/sample - loss: 3.8396 - y1_output_loss: 1.0682 - y2_output_loss: 2.7589 - y1_output_root_mean_squared_error: 1.0357 - y2_output_root_mean_squared_error: 1.6634 - val_loss: 6.0774 - val_y1_output_loss: 1.8187 - val_y2_output_loss: 4.1921 - val_y1_output_root_mean_squared_error: 1.3553 - val_y2_output_root_mean_squared_error: 2.0593
    Epoch 70/500
    614/614 [==============================] - 0s 139us/sample - loss: 4.3120 - y1_output_loss: 1.1240 - y2_output_loss: 3.2039 - y1_output_root_mean_squared_error: 1.0576 - y2_output_root_mean_squared_error: 1.7870 - val_loss: 7.1577 - val_y1_output_loss: 2.1330 - val_y2_output_loss: 5.0884 - val_y1_output_root_mean_squared_error: 1.4508 - val_y2_output_root_mean_squared_error: 2.2478
    Epoch 71/500
    614/614 [==============================] - 0s 129us/sample - loss: 4.0124 - y1_output_loss: 1.0877 - y2_output_loss: 2.9500 - y1_output_root_mean_squared_error: 1.0449 - y2_output_root_mean_squared_error: 1.7090 - val_loss: 12.8695 - val_y1_output_loss: 1.4496 - val_y2_output_loss: 11.4954 - val_y1_output_root_mean_squared_error: 1.2107 - val_y2_output_root_mean_squared_error: 3.3769
    Epoch 72/500
    614/614 [==============================] - 0s 138us/sample - loss: 4.4485 - y1_output_loss: 1.1782 - y2_output_loss: 3.3987 - y1_output_root_mean_squared_error: 1.0854 - y2_output_root_mean_squared_error: 1.8084 - val_loss: 32.2117 - val_y1_output_loss: 9.1000 - val_y2_output_loss: 22.1393 - val_y1_output_root_mean_squared_error: 3.0716 - val_y2_output_root_mean_squared_error: 4.7725
    Epoch 73/500
    614/614 [==============================] - 0s 139us/sample - loss: 7.7580 - y1_output_loss: 2.3585 - y2_output_loss: 5.3510 - y1_output_root_mean_squared_error: 1.5416 - y2_output_root_mean_squared_error: 2.3198 - val_loss: 5.3498 - val_y1_output_loss: 1.1882 - val_y2_output_loss: 4.0395 - val_y1_output_root_mean_squared_error: 1.1053 - val_y2_output_root_mean_squared_error: 2.0318
    Epoch 74/500
    614/614 [==============================] - 0s 139us/sample - loss: 12.8587 - y1_output_loss: 4.2574 - y2_output_loss: 8.4879 - y1_output_root_mean_squared_error: 2.0720 - y2_output_root_mean_squared_error: 2.9267 - val_loss: 6.4824 - val_y1_output_loss: 1.8460 - val_y2_output_loss: 4.4801 - val_y1_output_root_mean_squared_error: 1.3807 - val_y2_output_root_mean_squared_error: 2.1392
    Epoch 75/500
    614/614 [==============================] - 0s 134us/sample - loss: 4.9586 - y1_output_loss: 1.1371 - y2_output_loss: 3.7806 - y1_output_root_mean_squared_error: 1.0713 - y2_output_root_mean_squared_error: 1.9522 - val_loss: 12.1681 - val_y1_output_loss: 2.8909 - val_y2_output_loss: 9.1492 - val_y1_output_root_mean_squared_error: 1.7211 - val_y2_output_root_mean_squared_error: 3.0342
    Epoch 76/500
    614/614 [==============================] - 0s 135us/sample - loss: 3.4377 - y1_output_loss: 0.9349 - y2_output_loss: 2.4780 - y1_output_root_mean_squared_error: 0.9693 - y2_output_root_mean_squared_error: 1.5806 - val_loss: 4.6813 - val_y1_output_loss: 0.8705 - val_y2_output_loss: 3.6764 - val_y1_output_root_mean_squared_error: 0.9466 - val_y2_output_root_mean_squared_error: 1.9456
    Epoch 77/500
    614/614 [==============================] - 0s 147us/sample - loss: 4.0986 - y1_output_loss: 0.9748 - y2_output_loss: 3.2723 - y1_output_root_mean_squared_error: 0.9627 - y2_output_root_mean_squared_error: 1.7809 - val_loss: 29.0122 - val_y1_output_loss: 9.8058 - val_y2_output_loss: 18.8913 - val_y1_output_root_mean_squared_error: 3.1514 - val_y2_output_root_mean_squared_error: 4.3682
    Epoch 78/500
    614/614 [==============================] - 0s 137us/sample - loss: 4.4672 - y1_output_loss: 1.1098 - y2_output_loss: 3.3861 - y1_output_root_mean_squared_error: 1.0511 - y2_output_root_mean_squared_error: 1.8337 - val_loss: 4.2575 - val_y1_output_loss: 0.8875 - val_y2_output_loss: 3.2589 - val_y1_output_root_mean_squared_error: 0.9582 - val_y2_output_root_mean_squared_error: 1.8274
    Epoch 79/500
    614/614 [==============================] - 0s 137us/sample - loss: 3.5672 - y1_output_loss: 0.9061 - y2_output_loss: 2.6892 - y1_output_root_mean_squared_error: 0.9559 - y2_output_root_mean_squared_error: 1.6289 - val_loss: 4.8883 - val_y1_output_loss: 1.0379 - val_y2_output_loss: 3.7599 - val_y1_output_root_mean_squared_error: 1.0271 - val_y2_output_root_mean_squared_error: 1.9579
    Epoch 80/500
    614/614 [==============================] - 0s 130us/sample - loss: 4.3092 - y1_output_loss: 1.1005 - y2_output_loss: 3.1947 - y1_output_root_mean_squared_error: 1.0521 - y2_output_root_mean_squared_error: 1.7895 - val_loss: 6.2380 - val_y1_output_loss: 1.9315 - val_y2_output_loss: 4.2258 - val_y1_output_root_mean_squared_error: 1.3982 - val_y2_output_root_mean_squared_error: 2.0696
    Epoch 81/500
    614/614 [==============================] - 0s 131us/sample - loss: 4.0148 - y1_output_loss: 1.0699 - y2_output_loss: 2.9113 - y1_output_root_mean_squared_error: 1.0374 - y2_output_root_mean_squared_error: 1.7142 - val_loss: 3.9407 - val_y1_output_loss: 0.7913 - val_y2_output_loss: 3.0633 - val_y1_output_root_mean_squared_error: 0.8977 - val_y2_output_root_mean_squared_error: 1.7705
    Epoch 82/500
    614/614 [==============================] - 0s 139us/sample - loss: 3.4123 - y1_output_loss: 0.8346 - y2_output_loss: 2.6045 - y1_output_root_mean_squared_error: 0.9158 - y2_output_root_mean_squared_error: 1.6042 - val_loss: 7.5686 - val_y1_output_loss: 1.4710 - val_y2_output_loss: 5.9907 - val_y1_output_root_mean_squared_error: 1.2163 - val_y2_output_root_mean_squared_error: 2.4676
    Epoch 83/500
    614/614 [==============================] - 0s 128us/sample - loss: 3.5277 - y1_output_loss: 0.8008 - y2_output_loss: 2.7551 - y1_output_root_mean_squared_error: 0.8974 - y2_output_root_mean_squared_error: 1.6500 - val_loss: 11.0437 - val_y1_output_loss: 1.3469 - val_y2_output_loss: 9.5912 - val_y1_output_root_mean_squared_error: 1.1700 - val_y2_output_root_mean_squared_error: 3.1104
    Epoch 84/500
    614/614 [==============================] - 0s 130us/sample - loss: 4.1589 - y1_output_loss: 0.9096 - y2_output_loss: 3.2343 - y1_output_root_mean_squared_error: 0.9562 - y2_output_root_mean_squared_error: 1.8013 - val_loss: 6.4258 - val_y1_output_loss: 1.4109 - val_y2_output_loss: 5.1273 - val_y1_output_root_mean_squared_error: 1.1751 - val_y2_output_root_mean_squared_error: 2.2461
    Epoch 85/500
    614/614 [==============================] - 0s 137us/sample - loss: 3.2904 - y1_output_loss: 0.8740 - y2_output_loss: 2.4116 - y1_output_root_mean_squared_error: 0.9331 - y2_output_root_mean_squared_error: 1.5555 - val_loss: 3.8342 - val_y1_output_loss: 0.9717 - val_y2_output_loss: 2.8115 - val_y1_output_root_mean_squared_error: 0.9754 - val_y2_output_root_mean_squared_error: 1.6979
    Epoch 86/500
    614/614 [==============================] - 0s 147us/sample - loss: 4.1972 - y1_output_loss: 1.1497 - y2_output_loss: 3.0301 - y1_output_root_mean_squared_error: 1.0756 - y2_output_root_mean_squared_error: 1.7436 - val_loss: 6.3965 - val_y1_output_loss: 1.3252 - val_y2_output_loss: 5.0582 - val_y1_output_root_mean_squared_error: 1.1431 - val_y2_output_root_mean_squared_error: 2.2561
    Epoch 87/500
    614/614 [==============================] - 0s 143us/sample - loss: 3.6478 - y1_output_loss: 0.8405 - y2_output_loss: 2.8082 - y1_output_root_mean_squared_error: 0.9159 - y2_output_root_mean_squared_error: 1.6760 - val_loss: 6.5749 - val_y1_output_loss: 1.2024 - val_y2_output_loss: 5.3743 - val_y1_output_root_mean_squared_error: 1.0863 - val_y2_output_root_mean_squared_error: 2.3227
    Epoch 88/500
    614/614 [==============================] - 0s 137us/sample - loss: 3.2684 - y1_output_loss: 0.8212 - y2_output_loss: 2.4230 - y1_output_root_mean_squared_error: 0.9088 - y2_output_root_mean_squared_error: 1.5628 - val_loss: 4.4718 - val_y1_output_loss: 0.8358 - val_y2_output_loss: 3.5290 - val_y1_output_root_mean_squared_error: 0.9264 - val_y2_output_root_mean_squared_error: 1.9009
    Epoch 89/500
    614/614 [==============================] - 0s 138us/sample - loss: 8.7035 - y1_output_loss: 2.8690 - y2_output_loss: 5.8227 - y1_output_root_mean_squared_error: 1.6981 - y2_output_root_mean_squared_error: 2.4124 - val_loss: 16.8340 - val_y1_output_loss: 6.2425 - val_y2_output_loss: 10.3888 - val_y1_output_root_mean_squared_error: 2.5191 - val_y2_output_root_mean_squared_error: 3.2386
    Epoch 90/500
    614/614 [==============================] - 0s 130us/sample - loss: 3.5824 - y1_output_loss: 0.8496 - y2_output_loss: 2.7297 - y1_output_root_mean_squared_error: 0.9218 - y2_output_root_mean_squared_error: 1.6531 - val_loss: 4.3430 - val_y1_output_loss: 1.1453 - val_y2_output_loss: 3.1863 - val_y1_output_root_mean_squared_error: 1.0574 - val_y2_output_root_mean_squared_error: 1.7958
    Epoch 91/500
    614/614 [==============================] - 0s 127us/sample - loss: 3.2189 - y1_output_loss: 0.8652 - y2_output_loss: 2.3500 - y1_output_root_mean_squared_error: 0.9293 - y2_output_root_mean_squared_error: 1.5347 - val_loss: 3.5291 - val_y1_output_loss: 0.7548 - val_y2_output_loss: 2.6823 - val_y1_output_root_mean_squared_error: 0.8831 - val_y2_output_root_mean_squared_error: 1.6581
    Epoch 92/500
    614/614 [==============================] - 0s 132us/sample - loss: 3.3420 - y1_output_loss: 0.9114 - y2_output_loss: 2.4220 - y1_output_root_mean_squared_error: 0.9532 - y2_output_root_mean_squared_error: 1.5599 - val_loss: 6.4058 - val_y1_output_loss: 2.2327 - val_y2_output_loss: 3.9950 - val_y1_output_root_mean_squared_error: 1.5175 - val_y2_output_root_mean_squared_error: 2.0256
    Epoch 93/500
    614/614 [==============================] - 0s 133us/sample - loss: 3.0834 - y1_output_loss: 0.7380 - y2_output_loss: 2.3396 - y1_output_root_mean_squared_error: 0.8618 - y2_output_root_mean_squared_error: 1.5300 - val_loss: 3.5237 - val_y1_output_loss: 0.6148 - val_y2_output_loss: 2.8457 - val_y1_output_root_mean_squared_error: 0.7807 - val_y2_output_root_mean_squared_error: 1.7071
    Epoch 94/500
    614/614 [==============================] - 0s 143us/sample - loss: 2.6080 - y1_output_loss: 0.5794 - y2_output_loss: 2.0269 - y1_output_root_mean_squared_error: 0.7630 - y2_output_root_mean_squared_error: 1.4233 - val_loss: 3.3665 - val_y1_output_loss: 0.6049 - val_y2_output_loss: 2.7063 - val_y1_output_root_mean_squared_error: 0.7678 - val_y2_output_root_mean_squared_error: 1.6664
    Epoch 95/500
    614/614 [==============================] - 0s 145us/sample - loss: 3.7835 - y1_output_loss: 0.9318 - y2_output_loss: 2.8546 - y1_output_root_mean_squared_error: 0.9699 - y2_output_root_mean_squared_error: 1.6860 - val_loss: 6.8517 - val_y1_output_loss: 0.7439 - val_y2_output_loss: 6.0428 - val_y1_output_root_mean_squared_error: 0.8598 - val_y2_output_root_mean_squared_error: 2.4723
    Epoch 96/500
    614/614 [==============================] - 0s 137us/sample - loss: 3.4257 - y1_output_loss: 0.8412 - y2_output_loss: 2.5561 - y1_output_root_mean_squared_error: 0.9203 - y2_output_root_mean_squared_error: 1.6059 - val_loss: 3.5549 - val_y1_output_loss: 0.6166 - val_y2_output_loss: 2.8665 - val_y1_output_root_mean_squared_error: 0.7796 - val_y2_output_root_mean_squared_error: 1.7167
    Epoch 97/500
    614/614 [==============================] - 0s 128us/sample - loss: 2.5626 - y1_output_loss: 0.5288 - y2_output_loss: 2.0318 - y1_output_root_mean_squared_error: 0.7266 - y2_output_root_mean_squared_error: 1.4264 - val_loss: 4.3940 - val_y1_output_loss: 0.8057 - val_y2_output_loss: 3.4981 - val_y1_output_root_mean_squared_error: 0.9022 - val_y2_output_root_mean_squared_error: 1.8921
    Epoch 98/500
    614/614 [==============================] - 0s 127us/sample - loss: 2.8786 - y1_output_loss: 0.6974 - y2_output_loss: 2.1668 - y1_output_root_mean_squared_error: 0.8384 - y2_output_root_mean_squared_error: 1.4750 - val_loss: 4.1193 - val_y1_output_loss: 1.0233 - val_y2_output_loss: 3.0382 - val_y1_output_root_mean_squared_error: 1.0120 - val_y2_output_root_mean_squared_error: 1.7593
    Epoch 99/500
    614/614 [==============================] - 0s 130us/sample - loss: 3.8618 - y1_output_loss: 0.9043 - y2_output_loss: 2.9534 - y1_output_root_mean_squared_error: 0.9517 - y2_output_root_mean_squared_error: 1.7193 - val_loss: 3.5291 - val_y1_output_loss: 0.7537 - val_y2_output_loss: 2.7261 - val_y1_output_root_mean_squared_error: 0.8542 - val_y2_output_root_mean_squared_error: 1.6732
    Epoch 100/500
    614/614 [==============================] - 0s 129us/sample - loss: 2.6438 - y1_output_loss: 0.6678 - y2_output_loss: 1.9761 - y1_output_root_mean_squared_error: 0.8127 - y2_output_root_mean_squared_error: 1.4083 - val_loss: 4.1185 - val_y1_output_loss: 0.9245 - val_y2_output_loss: 3.1345 - val_y1_output_root_mean_squared_error: 0.9646 - val_y2_output_root_mean_squared_error: 1.7855
    Epoch 101/500
    614/614 [==============================] - 0s 128us/sample - loss: 3.3914 - y1_output_loss: 0.7934 - y2_output_loss: 2.5697 - y1_output_root_mean_squared_error: 0.8941 - y2_output_root_mean_squared_error: 1.6100 - val_loss: 3.2700 - val_y1_output_loss: 0.6020 - val_y2_output_loss: 2.5829 - val_y1_output_root_mean_squared_error: 0.7783 - val_y2_output_root_mean_squared_error: 1.6323
    Epoch 102/500
    614/614 [==============================] - 0s 128us/sample - loss: 2.9127 - y1_output_loss: 0.7043 - y2_output_loss: 2.1952 - y1_output_root_mean_squared_error: 0.8390 - y2_output_root_mean_squared_error: 1.4862 - val_loss: 3.2626 - val_y1_output_loss: 0.6309 - val_y2_output_loss: 2.5767 - val_y1_output_root_mean_squared_error: 0.7835 - val_y2_output_root_mean_squared_error: 1.6275
    Epoch 103/500
    614/614 [==============================] - 0s 131us/sample - loss: 7.7937 - y1_output_loss: 3.2048 - y2_output_loss: 6.3057 - y1_output_root_mean_squared_error: 1.5458 - y2_output_root_mean_squared_error: 2.3247 - val_loss: 158.0061 - val_y1_output_loss: 69.2106 - val_y2_output_loss: 87.5402 - val_y1_output_root_mean_squared_error: 8.3599 - val_y2_output_root_mean_squared_error: 9.3871
    Epoch 104/500
    614/614 [==============================] - 0s 129us/sample - loss: 22.3471 - y1_output_loss: 8.8123 - y2_output_loss: 13.3225 - y1_output_root_mean_squared_error: 2.9830 - y2_output_root_mean_squared_error: 3.6673 - val_loss: 4.2955 - val_y1_output_loss: 0.9008 - val_y2_output_loss: 3.3474 - val_y1_output_root_mean_squared_error: 0.9403 - val_y2_output_root_mean_squared_error: 1.8470
    Epoch 105/500
    614/614 [==============================] - 0s 130us/sample - loss: 3.5608 - y1_output_loss: 0.9143 - y2_output_loss: 2.6620 - y1_output_root_mean_squared_error: 0.9520 - y2_output_root_mean_squared_error: 1.6293 - val_loss: 9.0498 - val_y1_output_loss: 2.4177 - val_y2_output_loss: 6.7061 - val_y1_output_root_mean_squared_error: 1.5378 - val_y2_output_root_mean_squared_error: 2.5855
    Epoch 106/500
    614/614 [==============================] - 0s 127us/sample - loss: 2.4509 - y1_output_loss: 0.6073 - y2_output_loss: 1.8394 - y1_output_root_mean_squared_error: 0.7821 - y2_output_root_mean_squared_error: 1.3562 - val_loss: 4.1313 - val_y1_output_loss: 1.0660 - val_y2_output_loss: 2.9860 - val_y1_output_root_mean_squared_error: 1.0293 - val_y2_output_root_mean_squared_error: 1.7526
    Epoch 107/500
    614/614 [==============================] - 0s 127us/sample - loss: 3.5685 - y1_output_loss: 0.8309 - y2_output_loss: 2.7157 - y1_output_root_mean_squared_error: 0.9153 - y2_output_root_mean_squared_error: 1.6525 - val_loss: 4.3215 - val_y1_output_loss: 0.8715 - val_y2_output_loss: 3.3539 - val_y1_output_root_mean_squared_error: 0.9399 - val_y2_output_root_mean_squared_error: 1.8542
    Epoch 108/500
    614/614 [==============================] - 0s 130us/sample - loss: 2.2741 - y1_output_loss: 0.4761 - y2_output_loss: 1.7859 - y1_output_root_mean_squared_error: 0.6901 - y2_output_root_mean_squared_error: 1.3409 - val_loss: 3.1533 - val_y1_output_loss: 0.5558 - val_y2_output_loss: 2.5085 - val_y1_output_root_mean_squared_error: 0.7538 - val_y2_output_root_mean_squared_error: 1.6078
    Epoch 109/500
    614/614 [==============================] - 0s 130us/sample - loss: 2.1234 - y1_output_loss: 0.4287 - y2_output_loss: 1.6796 - y1_output_root_mean_squared_error: 0.6570 - y2_output_root_mean_squared_error: 1.3007 - val_loss: 3.1515 - val_y1_output_loss: 0.6809 - val_y2_output_loss: 2.4134 - val_y1_output_root_mean_squared_error: 0.8162 - val_y2_output_root_mean_squared_error: 1.5765
    Epoch 110/500
    614/614 [==============================] - 0s 126us/sample - loss: 2.1559 - y1_output_loss: 0.4444 - y2_output_loss: 1.6956 - y1_output_root_mean_squared_error: 0.6689 - y2_output_root_mean_squared_error: 1.3071 - val_loss: 2.8880 - val_y1_output_loss: 0.5113 - val_y2_output_loss: 2.3220 - val_y1_output_root_mean_squared_error: 0.7112 - val_y2_output_root_mean_squared_error: 1.5434
    Epoch 111/500
    614/614 [==============================] - 0s 127us/sample - loss: 2.3092 - y1_output_loss: 0.4842 - y2_output_loss: 1.8362 - y1_output_root_mean_squared_error: 0.6957 - y2_output_root_mean_squared_error: 1.3510 - val_loss: 3.4314 - val_y1_output_loss: 0.6078 - val_y2_output_loss: 2.7611 - val_y1_output_root_mean_squared_error: 0.7761 - val_y2_output_root_mean_squared_error: 1.6820
    Epoch 112/500
    614/614 [==============================] - 0s 128us/sample - loss: 2.6093 - y1_output_loss: 0.4703 - y2_output_loss: 2.1364 - y1_output_root_mean_squared_error: 0.6842 - y2_output_root_mean_squared_error: 1.4633 - val_loss: 8.3751 - val_y1_output_loss: 2.8493 - val_y2_output_loss: 5.3860 - val_y1_output_root_mean_squared_error: 1.6973 - val_y2_output_root_mean_squared_error: 2.3440
    Epoch 113/500
    614/614 [==============================] - 0s 130us/sample - loss: 2.1448 - y1_output_loss: 0.5130 - y2_output_loss: 1.6236 - y1_output_root_mean_squared_error: 0.7192 - y2_output_root_mean_squared_error: 1.2758 - val_loss: 4.1385 - val_y1_output_loss: 0.5559 - val_y2_output_loss: 3.5538 - val_y1_output_root_mean_squared_error: 0.7357 - val_y2_output_root_mean_squared_error: 1.8966
    Epoch 114/500
    614/614 [==============================] - 0s 129us/sample - loss: 3.8126 - y1_output_loss: 1.0292 - y2_output_loss: 2.7543 - y1_output_root_mean_squared_error: 1.0190 - y2_output_root_mean_squared_error: 1.6656 - val_loss: 3.7984 - val_y1_output_loss: 0.7139 - val_y2_output_loss: 3.0116 - val_y1_output_root_mean_squared_error: 0.8443 - val_y2_output_root_mean_squared_error: 1.7566
    Epoch 115/500
    614/614 [==============================] - 0s 128us/sample - loss: 2.1934 - y1_output_loss: 0.5296 - y2_output_loss: 1.6517 - y1_output_root_mean_squared_error: 0.7303 - y2_output_root_mean_squared_error: 1.2885 - val_loss: 4.9142 - val_y1_output_loss: 0.7292 - val_y2_output_loss: 4.0736 - val_y1_output_root_mean_squared_error: 0.8651 - val_y2_output_root_mean_squared_error: 2.0410
    Epoch 116/500
    614/614 [==============================] - 0s 128us/sample - loss: 2.5805 - y1_output_loss: 0.5588 - y2_output_loss: 2.0072 - y1_output_root_mean_squared_error: 0.7457 - y2_output_root_mean_squared_error: 1.4228 - val_loss: 3.1035 - val_y1_output_loss: 0.6819 - val_y2_output_loss: 2.3884 - val_y1_output_root_mean_squared_error: 0.8122 - val_y2_output_root_mean_squared_error: 1.5633
    Epoch 117/500
    614/614 [==============================] - 0s 128us/sample - loss: 2.2188 - y1_output_loss: 0.5188 - y2_output_loss: 1.6896 - y1_output_root_mean_squared_error: 0.7218 - y2_output_root_mean_squared_error: 1.3030 - val_loss: 4.5170 - val_y1_output_loss: 1.0200 - val_y2_output_loss: 3.5270 - val_y1_output_root_mean_squared_error: 0.9969 - val_y2_output_root_mean_squared_error: 1.8770
    Epoch 118/500
    614/614 [==============================] - 0s 129us/sample - loss: 2.2642 - y1_output_loss: 0.5004 - y2_output_loss: 1.7944 - y1_output_root_mean_squared_error: 0.7034 - y2_output_root_mean_squared_error: 1.3302 - val_loss: 13.4534 - val_y1_output_loss: 4.4148 - val_y2_output_loss: 8.8313 - val_y1_output_root_mean_squared_error: 2.1166 - val_y2_output_root_mean_squared_error: 2.9956
    Epoch 119/500
    614/614 [==============================] - 0s 126us/sample - loss: 3.4317 - y1_output_loss: 0.9211 - y2_output_loss: 2.4796 - y1_output_root_mean_squared_error: 0.9641 - y2_output_root_mean_squared_error: 1.5819 - val_loss: 2.7845 - val_y1_output_loss: 0.4619 - val_y2_output_loss: 2.2699 - val_y1_output_root_mean_squared_error: 0.6720 - val_y2_output_root_mean_squared_error: 1.5274
    Epoch 120/500
    614/614 [==============================] - 0s 131us/sample - loss: 2.7194 - y1_output_loss: 0.6395 - y2_output_loss: 2.0567 - y1_output_root_mean_squared_error: 0.8029 - y2_output_root_mean_squared_error: 1.4404 - val_loss: 3.1698 - val_y1_output_loss: 0.5139 - val_y2_output_loss: 2.5761 - val_y1_output_root_mean_squared_error: 0.7171 - val_y2_output_root_mean_squared_error: 1.6296
    Epoch 121/500
    614/614 [==============================] - 0s 128us/sample - loss: 2.4622 - y1_output_loss: 0.5483 - y2_output_loss: 1.8992 - y1_output_root_mean_squared_error: 0.7393 - y2_output_root_mean_squared_error: 1.3841 - val_loss: 3.0393 - val_y1_output_loss: 0.6120 - val_y2_output_loss: 2.3677 - val_y1_output_root_mean_squared_error: 0.7766 - val_y2_output_root_mean_squared_error: 1.5608
    Epoch 122/500
    614/614 [==============================] - 0s 135us/sample - loss: 1.7436 - y1_output_loss: 0.3782 - y2_output_loss: 1.3618 - y1_output_root_mean_squared_error: 0.6109 - y2_output_root_mean_squared_error: 1.1707 - val_loss: 4.8836 - val_y1_output_loss: 1.1739 - val_y2_output_loss: 3.6981 - val_y1_output_root_mean_squared_error: 1.0782 - val_y2_output_root_mean_squared_error: 1.9290
    Epoch 123/500
    614/614 [==============================] - 0s 136us/sample - loss: 3.0457 - y1_output_loss: 0.7464 - y2_output_loss: 2.2793 - y1_output_root_mean_squared_error: 0.8651 - y2_output_root_mean_squared_error: 1.5157 - val_loss: 3.3817 - val_y1_output_loss: 0.6666 - val_y2_output_loss: 2.6238 - val_y1_output_root_mean_squared_error: 0.8221 - val_y2_output_root_mean_squared_error: 1.6449
    Epoch 124/500
    614/614 [==============================] - 0s 129us/sample - loss: 2.4809 - y1_output_loss: 0.5777 - y2_output_loss: 1.9070 - y1_output_root_mean_squared_error: 0.7620 - y2_output_root_mean_squared_error: 1.3785 - val_loss: 4.7847 - val_y1_output_loss: 0.6744 - val_y2_output_loss: 4.0393 - val_y1_output_root_mean_squared_error: 0.8268 - val_y2_output_root_mean_squared_error: 2.0251
    Epoch 125/500
    614/614 [==============================] - 0s 126us/sample - loss: 2.4572 - y1_output_loss: 0.5789 - y2_output_loss: 1.8554 - y1_output_root_mean_squared_error: 0.7644 - y2_output_root_mean_squared_error: 1.3686 - val_loss: 2.5188 - val_y1_output_loss: 0.3893 - val_y2_output_loss: 2.0748 - val_y1_output_root_mean_squared_error: 0.6216 - val_y2_output_root_mean_squared_error: 1.4603
    Epoch 126/500
    614/614 [==============================] - 0s 127us/sample - loss: 2.0650 - y1_output_loss: 0.4304 - y2_output_loss: 1.6663 - y1_output_root_mean_squared_error: 0.6469 - y2_output_root_mean_squared_error: 1.2832 - val_loss: 7.8846 - val_y1_output_loss: 1.9611 - val_y2_output_loss: 5.7005 - val_y1_output_root_mean_squared_error: 1.4252 - val_y2_output_root_mean_squared_error: 2.4194
    Epoch 127/500
    614/614 [==============================] - 0s 140us/sample - loss: 2.8335 - y1_output_loss: 0.6696 - y2_output_loss: 2.2328 - y1_output_root_mean_squared_error: 0.8202 - y2_output_root_mean_squared_error: 1.4700 - val_loss: 2.7360 - val_y1_output_loss: 0.4164 - val_y2_output_loss: 2.2620 - val_y1_output_root_mean_squared_error: 0.6533 - val_y2_output_root_mean_squared_error: 1.5196
    Epoch 128/500
    614/614 [==============================] - 0s 138us/sample - loss: 2.0905 - y1_output_loss: 0.4469 - y2_output_loss: 1.6423 - y1_output_root_mean_squared_error: 0.6642 - y2_output_root_mean_squared_error: 1.2843 - val_loss: 2.7131 - val_y1_output_loss: 0.4883 - val_y2_output_loss: 2.1897 - val_y1_output_root_mean_squared_error: 0.7039 - val_y2_output_root_mean_squared_error: 1.4892
    Epoch 129/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.6881 - y1_output_loss: 0.4160 - y2_output_loss: 1.2810 - y1_output_root_mean_squared_error: 0.6423 - y2_output_root_mean_squared_error: 1.1294 - val_loss: 2.9282 - val_y1_output_loss: 0.5270 - val_y2_output_loss: 2.3539 - val_y1_output_root_mean_squared_error: 0.7266 - val_y2_output_root_mean_squared_error: 1.5493
    Epoch 130/500
    614/614 [==============================] - 0s 130us/sample - loss: 1.9255 - y1_output_loss: 0.4255 - y2_output_loss: 1.4866 - y1_output_root_mean_squared_error: 0.6553 - y2_output_root_mean_squared_error: 1.2231 - val_loss: 2.4993 - val_y1_output_loss: 0.3983 - val_y2_output_loss: 2.0536 - val_y1_output_root_mean_squared_error: 0.6274 - val_y2_output_root_mean_squared_error: 1.4511
    Epoch 131/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.7660 - y1_output_loss: 0.3263 - y2_output_loss: 1.4385 - y1_output_root_mean_squared_error: 0.5705 - y2_output_root_mean_squared_error: 1.2002 - val_loss: 3.1876 - val_y1_output_loss: 0.4507 - val_y2_output_loss: 2.7282 - val_y1_output_root_mean_squared_error: 0.6591 - val_y2_output_root_mean_squared_error: 1.6593
    Epoch 132/500
    614/614 [==============================] - 0s 129us/sample - loss: 1.9486 - y1_output_loss: 0.4716 - y2_output_loss: 1.5557 - y1_output_root_mean_squared_error: 0.6613 - y2_output_root_mean_squared_error: 1.2294 - val_loss: 4.1857 - val_y1_output_loss: 1.3866 - val_y2_output_loss: 2.6864 - val_y1_output_root_mean_squared_error: 1.1943 - val_y2_output_root_mean_squared_error: 1.6611
    Epoch 133/500
    614/614 [==============================] - 0s 128us/sample - loss: 2.4569 - y1_output_loss: 0.4968 - y2_output_loss: 1.9667 - y1_output_root_mean_squared_error: 0.7071 - y2_output_root_mean_squared_error: 1.3989 - val_loss: 6.2426 - val_y1_output_loss: 0.8009 - val_y2_output_loss: 5.3643 - val_y1_output_root_mean_squared_error: 0.8937 - val_y2_output_root_mean_squared_error: 2.3332
    Epoch 134/500
    614/614 [==============================] - 0s 126us/sample - loss: 2.1532 - y1_output_loss: 0.5140 - y2_output_loss: 1.6224 - y1_output_root_mean_squared_error: 0.7182 - y2_output_root_mean_squared_error: 1.2796 - val_loss: 2.7211 - val_y1_output_loss: 0.4155 - val_y2_output_loss: 2.2395 - val_y1_output_root_mean_squared_error: 0.6452 - val_y2_output_root_mean_squared_error: 1.5182
    Epoch 135/500
    614/614 [==============================] - 0s 127us/sample - loss: 1.9109 - y1_output_loss: 0.4328 - y2_output_loss: 1.4621 - y1_output_root_mean_squared_error: 0.6603 - y2_output_root_mean_squared_error: 1.2145 - val_loss: 2.4557 - val_y1_output_loss: 0.4464 - val_y2_output_loss: 1.9518 - val_y1_output_root_mean_squared_error: 0.6702 - val_y2_output_root_mean_squared_error: 1.4165
    Epoch 136/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.9982 - y1_output_loss: 0.4110 - y2_output_loss: 1.5718 - y1_output_root_mean_squared_error: 0.6440 - y2_output_root_mean_squared_error: 1.2584 - val_loss: 2.3826 - val_y1_output_loss: 0.3494 - val_y2_output_loss: 1.9768 - val_y1_output_root_mean_squared_error: 0.5924 - val_y2_output_root_mean_squared_error: 1.4253
    Epoch 137/500
    614/614 [==============================] - 0s 127us/sample - loss: 2.0542 - y1_output_loss: 0.4609 - y2_output_loss: 1.6014 - y1_output_root_mean_squared_error: 0.6724 - y2_output_root_mean_squared_error: 1.2657 - val_loss: 5.8881 - val_y1_output_loss: 1.3723 - val_y2_output_loss: 4.5595 - val_y1_output_root_mean_squared_error: 1.1637 - val_y2_output_root_mean_squared_error: 2.1293
    Epoch 138/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.6780 - y1_output_loss: 0.3529 - y2_output_loss: 1.3204 - y1_output_root_mean_squared_error: 0.5936 - y2_output_root_mean_squared_error: 1.1514 - val_loss: 2.5876 - val_y1_output_loss: 0.4808 - val_y2_output_loss: 2.0754 - val_y1_output_root_mean_squared_error: 0.6905 - val_y2_output_root_mean_squared_error: 1.4529
    Epoch 139/500
    614/614 [==============================] - 0s 134us/sample - loss: 1.7052 - y1_output_loss: 0.3146 - y2_output_loss: 1.3892 - y1_output_root_mean_squared_error: 0.5574 - y2_output_root_mean_squared_error: 1.1809 - val_loss: 2.4733 - val_y1_output_loss: 0.3524 - val_y2_output_loss: 2.0793 - val_y1_output_root_mean_squared_error: 0.5873 - val_y2_output_root_mean_squared_error: 1.4589
    Epoch 140/500
    614/614 [==============================] - 0s 133us/sample - loss: 1.6575 - y1_output_loss: 0.3140 - y2_output_loss: 1.3409 - y1_output_root_mean_squared_error: 0.5608 - y2_output_root_mean_squared_error: 1.1589 - val_loss: 4.5696 - val_y1_output_loss: 0.5401 - val_y2_output_loss: 3.9520 - val_y1_output_root_mean_squared_error: 0.7422 - val_y2_output_root_mean_squared_error: 2.0047
    Epoch 141/500
    614/614 [==============================] - 0s 136us/sample - loss: 1.6680 - y1_output_loss: 0.3274 - y2_output_loss: 1.3523 - y1_output_root_mean_squared_error: 0.5695 - y2_output_root_mean_squared_error: 1.1592 - val_loss: 2.5452 - val_y1_output_loss: 0.3720 - val_y2_output_loss: 2.1212 - val_y1_output_root_mean_squared_error: 0.6086 - val_y2_output_root_mean_squared_error: 1.4747
    Epoch 142/500
    614/614 [==============================] - 0s 131us/sample - loss: 1.7012 - y1_output_loss: 0.3194 - y2_output_loss: 1.3999 - y1_output_root_mean_squared_error: 0.5628 - y2_output_root_mean_squared_error: 1.1766 - val_loss: 7.5650 - val_y1_output_loss: 2.0890 - val_y2_output_loss: 5.2785 - val_y1_output_root_mean_squared_error: 1.4675 - val_y2_output_root_mean_squared_error: 2.3263
    Epoch 143/500
    614/614 [==============================] - 0s 128us/sample - loss: 2.2836 - y1_output_loss: 0.5567 - y2_output_loss: 1.7135 - y1_output_root_mean_squared_error: 0.7480 - y2_output_root_mean_squared_error: 1.3130 - val_loss: 2.8312 - val_y1_output_loss: 0.4030 - val_y2_output_loss: 2.3611 - val_y1_output_root_mean_squared_error: 0.6420 - val_y2_output_root_mean_squared_error: 1.5553
    Epoch 144/500
    614/614 [==============================] - 0s 127us/sample - loss: 1.5396 - y1_output_loss: 0.3240 - y2_output_loss: 1.2323 - y1_output_root_mean_squared_error: 0.5671 - y2_output_root_mean_squared_error: 1.1037 - val_loss: 3.0579 - val_y1_output_loss: 0.3711 - val_y2_output_loss: 2.6281 - val_y1_output_root_mean_squared_error: 0.6082 - val_y2_output_root_mean_squared_error: 1.6395
    Epoch 145/500
    614/614 [==============================] - 0s 132us/sample - loss: 2.0730 - y1_output_loss: 0.4036 - y2_output_loss: 1.6560 - y1_output_root_mean_squared_error: 0.6377 - y2_output_root_mean_squared_error: 1.2909 - val_loss: 2.2889 - val_y1_output_loss: 0.3478 - val_y2_output_loss: 1.8895 - val_y1_output_root_mean_squared_error: 0.5903 - val_y2_output_root_mean_squared_error: 1.3930
    Epoch 146/500
    614/614 [==============================] - 0s 137us/sample - loss: 1.6029 - y1_output_loss: 0.2926 - y2_output_loss: 1.3110 - y1_output_root_mean_squared_error: 0.5415 - y2_output_root_mean_squared_error: 1.1444 - val_loss: 2.9368 - val_y1_output_loss: 0.5693 - val_y2_output_loss: 2.4004 - val_y1_output_root_mean_squared_error: 0.7406 - val_y2_output_root_mean_squared_error: 1.5454
    Epoch 147/500
    614/614 [==============================] - 0s 138us/sample - loss: 2.5843 - y1_output_loss: 0.6258 - y2_output_loss: 1.9388 - y1_output_root_mean_squared_error: 0.7948 - y2_output_root_mean_squared_error: 1.3974 - val_loss: 2.4261 - val_y1_output_loss: 0.4617 - val_y2_output_loss: 1.9377 - val_y1_output_root_mean_squared_error: 0.6746 - val_y2_output_root_mean_squared_error: 1.4039
    Epoch 148/500
    614/614 [==============================] - 0s 132us/sample - loss: 1.5566 - y1_output_loss: 0.3068 - y2_output_loss: 1.2401 - y1_output_root_mean_squared_error: 0.5551 - y2_output_root_mean_squared_error: 1.1174 - val_loss: 2.3766 - val_y1_output_loss: 0.3669 - val_y2_output_loss: 1.9590 - val_y1_output_root_mean_squared_error: 0.6044 - val_y2_output_root_mean_squared_error: 1.4182
    Epoch 149/500
    614/614 [==============================] - 0s 131us/sample - loss: 1.8956 - y1_output_loss: 0.3968 - y2_output_loss: 1.4940 - y1_output_root_mean_squared_error: 0.6290 - y2_output_root_mean_squared_error: 1.2247 - val_loss: 3.6572 - val_y1_output_loss: 0.4885 - val_y2_output_loss: 3.2297 - val_y1_output_root_mean_squared_error: 0.6916 - val_y2_output_root_mean_squared_error: 1.7830
    Epoch 150/500
    614/614 [==============================] - 0s 132us/sample - loss: 1.9477 - y1_output_loss: 0.3693 - y2_output_loss: 1.5642 - y1_output_root_mean_squared_error: 0.6101 - y2_output_root_mean_squared_error: 1.2552 - val_loss: 2.4703 - val_y1_output_loss: 0.5114 - val_y2_output_loss: 1.9022 - val_y1_output_root_mean_squared_error: 0.7235 - val_y2_output_root_mean_squared_error: 1.3953
    Epoch 151/500
    614/614 [==============================] - 0s 130us/sample - loss: 1.9004 - y1_output_loss: 0.3601 - y2_output_loss: 1.5240 - y1_output_root_mean_squared_error: 0.6028 - y2_output_root_mean_squared_error: 1.2397 - val_loss: 2.8787 - val_y1_output_loss: 0.5207 - val_y2_output_loss: 2.3668 - val_y1_output_root_mean_squared_error: 0.7204 - val_y2_output_root_mean_squared_error: 1.5362
    Epoch 152/500
    614/614 [==============================] - 0s 129us/sample - loss: 1.7380 - y1_output_loss: 0.3736 - y2_output_loss: 1.3838 - y1_output_root_mean_squared_error: 0.6089 - y2_output_root_mean_squared_error: 1.1693 - val_loss: 5.0036 - val_y1_output_loss: 0.9661 - val_y2_output_loss: 4.0356 - val_y1_output_root_mean_squared_error: 0.9724 - val_y2_output_root_mean_squared_error: 2.0145
    Epoch 153/500
    614/614 [==============================] - 0s 136us/sample - loss: 1.5086 - y1_output_loss: 0.3197 - y2_output_loss: 1.1765 - y1_output_root_mean_squared_error: 0.5680 - y2_output_root_mean_squared_error: 1.0890 - val_loss: 2.1829 - val_y1_output_loss: 0.3808 - val_y2_output_loss: 1.7846 - val_y1_output_root_mean_squared_error: 0.6105 - val_y2_output_root_mean_squared_error: 1.3454
    Epoch 154/500
    614/614 [==============================] - 0s 131us/sample - loss: 1.4530 - y1_output_loss: 0.3245 - y2_output_loss: 1.1358 - y1_output_root_mean_squared_error: 0.5696 - y2_output_root_mean_squared_error: 1.0623 - val_loss: 2.9581 - val_y1_output_loss: 0.7755 - val_y2_output_loss: 2.2178 - val_y1_output_root_mean_squared_error: 0.8663 - val_y2_output_root_mean_squared_error: 1.4858
    Epoch 155/500
    614/614 [==============================] - 0s 129us/sample - loss: 1.3961 - y1_output_loss: 0.2623 - y2_output_loss: 1.1621 - y1_output_root_mean_squared_error: 0.5090 - y2_output_root_mean_squared_error: 1.0664 - val_loss: 3.9198 - val_y1_output_loss: 0.9870 - val_y2_output_loss: 2.9119 - val_y1_output_root_mean_squared_error: 0.9979 - val_y2_output_root_mean_squared_error: 1.7099
    Epoch 156/500
    614/614 [==============================] - 0s 131us/sample - loss: 1.3764 - y1_output_loss: 0.2773 - y2_output_loss: 1.0909 - y1_output_root_mean_squared_error: 0.5279 - y2_output_root_mean_squared_error: 1.0477 - val_loss: 2.6315 - val_y1_output_loss: 0.4868 - val_y2_output_loss: 2.1527 - val_y1_output_root_mean_squared_error: 0.6896 - val_y2_output_root_mean_squared_error: 1.4683
    Epoch 157/500
    614/614 [==============================] - 0s 130us/sample - loss: 1.3993 - y1_output_loss: 0.2526 - y2_output_loss: 1.1425 - y1_output_root_mean_squared_error: 0.5028 - y2_output_root_mean_squared_error: 1.0707 - val_loss: 2.3579 - val_y1_output_loss: 0.3671 - val_y2_output_loss: 1.9434 - val_y1_output_root_mean_squared_error: 0.6049 - val_y2_output_root_mean_squared_error: 1.4114
    Epoch 158/500
    614/614 [==============================] - 0s 129us/sample - loss: 1.9051 - y1_output_loss: 0.4445 - y2_output_loss: 1.5129 - y1_output_root_mean_squared_error: 0.6685 - y2_output_root_mean_squared_error: 1.2075 - val_loss: 12.5943 - val_y1_output_loss: 1.7840 - val_y2_output_loss: 10.9057 - val_y1_output_root_mean_squared_error: 1.3408 - val_y2_output_root_mean_squared_error: 3.2858
    Epoch 159/500
    614/614 [==============================] - 0s 131us/sample - loss: 1.5082 - y1_output_loss: 0.3293 - y2_output_loss: 1.1715 - y1_output_root_mean_squared_error: 0.5744 - y2_output_root_mean_squared_error: 1.0855 - val_loss: 2.6467 - val_y1_output_loss: 0.4487 - val_y2_output_loss: 2.1537 - val_y1_output_root_mean_squared_error: 0.6711 - val_y2_output_root_mean_squared_error: 1.4820
    Epoch 160/500
    614/614 [==============================] - 0s 129us/sample - loss: 1.4349 - y1_output_loss: 0.3149 - y2_output_loss: 1.1545 - y1_output_root_mean_squared_error: 0.5514 - y2_output_root_mean_squared_error: 1.0634 - val_loss: 17.8158 - val_y1_output_loss: 5.2676 - val_y2_output_loss: 12.1189 - val_y1_output_root_mean_squared_error: 2.3274 - val_y2_output_root_mean_squared_error: 3.5212
    Epoch 161/500
    614/614 [==============================] - 0s 133us/sample - loss: 2.0466 - y1_output_loss: 0.5078 - y2_output_loss: 1.5259 - y1_output_root_mean_squared_error: 0.7145 - y2_output_root_mean_squared_error: 1.2394 - val_loss: 3.2763 - val_y1_output_loss: 0.6918 - val_y2_output_loss: 2.6635 - val_y1_output_root_mean_squared_error: 0.8160 - val_y2_output_root_mean_squared_error: 1.6157
    Epoch 162/500
    614/614 [==============================] - 0s 131us/sample - loss: 1.8339 - y1_output_loss: 0.3931 - y2_output_loss: 1.4286 - y1_output_root_mean_squared_error: 0.6294 - y2_output_root_mean_squared_error: 1.1991 - val_loss: 1.9028 - val_y1_output_loss: 0.3185 - val_y2_output_loss: 1.5536 - val_y1_output_root_mean_squared_error: 0.5622 - val_y2_output_root_mean_squared_error: 1.2597
    Epoch 163/500
    614/614 [==============================] - 0s 130us/sample - loss: 1.4584 - y1_output_loss: 0.3096 - y2_output_loss: 1.1434 - y1_output_root_mean_squared_error: 0.5566 - y2_output_root_mean_squared_error: 1.0717 - val_loss: 2.3129 - val_y1_output_loss: 0.3426 - val_y2_output_loss: 1.9264 - val_y1_output_root_mean_squared_error: 0.5846 - val_y2_output_root_mean_squared_error: 1.4040
    Epoch 164/500
    614/614 [==============================] - 0s 138us/sample - loss: 1.5564 - y1_output_loss: 0.2623 - y2_output_loss: 1.2807 - y1_output_root_mean_squared_error: 0.5143 - y2_output_root_mean_squared_error: 1.1366 - val_loss: 2.0835 - val_y1_output_loss: 0.3086 - val_y2_output_loss: 1.7398 - val_y1_output_root_mean_squared_error: 0.5532 - val_y2_output_root_mean_squared_error: 1.3332
    Epoch 165/500
    614/614 [==============================] - 0s 141us/sample - loss: 1.4494 - y1_output_loss: 0.2569 - y2_output_loss: 1.1882 - y1_output_root_mean_squared_error: 0.5089 - y2_output_root_mean_squared_error: 1.0911 - val_loss: 2.1943 - val_y1_output_loss: 0.3732 - val_y2_output_loss: 1.8105 - val_y1_output_root_mean_squared_error: 0.6029 - val_y2_output_root_mean_squared_error: 1.3530
    Epoch 166/500
    614/614 [==============================] - 0s 140us/sample - loss: 1.6654 - y1_output_loss: 0.3615 - y2_output_loss: 1.2888 - y1_output_root_mean_squared_error: 0.6037 - y2_output_root_mean_squared_error: 1.1406 - val_loss: 1.9214 - val_y1_output_loss: 0.3005 - val_y2_output_loss: 1.6091 - val_y1_output_root_mean_squared_error: 0.5481 - val_y2_output_root_mean_squared_error: 1.2732
    Epoch 167/500
    614/614 [==============================] - 0s 128us/sample - loss: 2.5690 - y1_output_loss: 0.6040 - y2_output_loss: 1.9433 - y1_output_root_mean_squared_error: 0.7807 - y2_output_root_mean_squared_error: 1.3998 - val_loss: 2.1552 - val_y1_output_loss: 0.3260 - val_y2_output_loss: 1.7962 - val_y1_output_root_mean_squared_error: 0.5686 - val_y2_output_root_mean_squared_error: 1.3535
    Epoch 168/500
    614/614 [==============================] - 0s 138us/sample - loss: 1.3252 - y1_output_loss: 0.2807 - y2_output_loss: 1.0519 - y1_output_root_mean_squared_error: 0.5263 - y2_output_root_mean_squared_error: 1.0238 - val_loss: 2.2654 - val_y1_output_loss: 0.4003 - val_y2_output_loss: 1.8589 - val_y1_output_root_mean_squared_error: 0.6306 - val_y2_output_root_mean_squared_error: 1.3666
    Epoch 169/500
    614/614 [==============================] - 0s 129us/sample - loss: 1.5162 - y1_output_loss: 0.2617 - y2_output_loss: 1.2489 - y1_output_root_mean_squared_error: 0.5132 - y2_output_root_mean_squared_error: 1.1193 - val_loss: 2.1480 - val_y1_output_loss: 0.3433 - val_y2_output_loss: 1.8100 - val_y1_output_root_mean_squared_error: 0.5760 - val_y2_output_root_mean_squared_error: 1.3477
    Epoch 170/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.6191 - y1_output_loss: 0.3068 - y2_output_loss: 1.3028 - y1_output_root_mean_squared_error: 0.5555 - y2_output_root_mean_squared_error: 1.1448 - val_loss: 2.2142 - val_y1_output_loss: 0.3917 - val_y2_output_loss: 1.7967 - val_y1_output_root_mean_squared_error: 0.6219 - val_y2_output_root_mean_squared_error: 1.3519
    Epoch 171/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.2163 - y1_output_loss: 0.2429 - y2_output_loss: 0.9711 - y1_output_root_mean_squared_error: 0.4900 - y2_output_root_mean_squared_error: 0.9880 - val_loss: 2.3488 - val_y1_output_loss: 0.4370 - val_y2_output_loss: 1.9251 - val_y1_output_root_mean_squared_error: 0.6564 - val_y2_output_root_mean_squared_error: 1.3849
    Epoch 172/500
    614/614 [==============================] - 0s 129us/sample - loss: 1.6585 - y1_output_loss: 0.3535 - y2_output_loss: 1.3039 - y1_output_root_mean_squared_error: 0.5887 - y2_output_root_mean_squared_error: 1.1454 - val_loss: 2.2045 - val_y1_output_loss: 0.3232 - val_y2_output_loss: 1.8400 - val_y1_output_root_mean_squared_error: 0.5657 - val_y2_output_root_mean_squared_error: 1.3728
    Epoch 173/500
    614/614 [==============================] - 0s 129us/sample - loss: 1.3625 - y1_output_loss: 0.2373 - y2_output_loss: 1.1155 - y1_output_root_mean_squared_error: 0.4886 - y2_output_root_mean_squared_error: 1.0601 - val_loss: 2.1779 - val_y1_output_loss: 0.3241 - val_y2_output_loss: 1.8185 - val_y1_output_root_mean_squared_error: 0.5714 - val_y2_output_root_mean_squared_error: 1.3607
    Epoch 174/500
    614/614 [==============================] - 0s 130us/sample - loss: 1.3294 - y1_output_loss: 0.2390 - y2_output_loss: 1.1012 - y1_output_root_mean_squared_error: 0.4849 - y2_output_root_mean_squared_error: 1.0461 - val_loss: 2.2113 - val_y1_output_loss: 0.4930 - val_y2_output_loss: 1.6968 - val_y1_output_root_mean_squared_error: 0.6957 - val_y2_output_root_mean_squared_error: 1.3142
    Epoch 175/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.6952 - y1_output_loss: 0.3306 - y2_output_loss: 1.3577 - y1_output_root_mean_squared_error: 0.5772 - y2_output_root_mean_squared_error: 1.1671 - val_loss: 2.4246 - val_y1_output_loss: 0.3024 - val_y2_output_loss: 2.1455 - val_y1_output_root_mean_squared_error: 0.5507 - val_y2_output_root_mean_squared_error: 1.4565
    Epoch 176/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.4641 - y1_output_loss: 0.2873 - y2_output_loss: 1.1662 - y1_output_root_mean_squared_error: 0.5369 - y2_output_root_mean_squared_error: 1.0844 - val_loss: 2.4049 - val_y1_output_loss: 0.5368 - val_y2_output_loss: 1.8796 - val_y1_output_root_mean_squared_error: 0.7223 - val_y2_output_root_mean_squared_error: 1.3723
    Epoch 177/500
    614/614 [==============================] - 0s 128us/sample - loss: 2.1496 - y1_output_loss: 0.4132 - y2_output_loss: 1.7210 - y1_output_root_mean_squared_error: 0.6446 - y2_output_root_mean_squared_error: 1.3168 - val_loss: 2.8180 - val_y1_output_loss: 0.6169 - val_y2_output_loss: 2.2136 - val_y1_output_root_mean_squared_error: 0.7797 - val_y2_output_root_mean_squared_error: 1.4866
    Epoch 178/500
    614/614 [==============================] - 0s 127us/sample - loss: 1.4223 - y1_output_loss: 0.3037 - y2_output_loss: 1.1289 - y1_output_root_mean_squared_error: 0.5500 - y2_output_root_mean_squared_error: 1.0582 - val_loss: 2.1397 - val_y1_output_loss: 0.3323 - val_y2_output_loss: 1.7632 - val_y1_output_root_mean_squared_error: 0.5796 - val_y2_output_root_mean_squared_error: 1.3431
    Epoch 179/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.2002 - y1_output_loss: 0.2554 - y2_output_loss: 0.9922 - y1_output_root_mean_squared_error: 0.5071 - y2_output_root_mean_squared_error: 0.9711 - val_loss: 5.0993 - val_y1_output_loss: 0.5634 - val_y2_output_loss: 4.5411 - val_y1_output_root_mean_squared_error: 0.7420 - val_y2_output_root_mean_squared_error: 2.1328
    Epoch 180/500
    614/614 [==============================] - 0s 129us/sample - loss: 1.8959 - y1_output_loss: 0.4004 - y2_output_loss: 1.4826 - y1_output_root_mean_squared_error: 0.6355 - y2_output_root_mean_squared_error: 1.2215 - val_loss: 2.6342 - val_y1_output_loss: 0.3741 - val_y2_output_loss: 2.2540 - val_y1_output_root_mean_squared_error: 0.6009 - val_y2_output_root_mean_squared_error: 1.5077
    Epoch 181/500
    614/614 [==============================] - 0s 137us/sample - loss: 1.1641 - y1_output_loss: 0.2121 - y2_output_loss: 0.9431 - y1_output_root_mean_squared_error: 0.4620 - y2_output_root_mean_squared_error: 0.9750 - val_loss: 2.3577 - val_y1_output_loss: 0.3820 - val_y2_output_loss: 1.9645 - val_y1_output_root_mean_squared_error: 0.6138 - val_y2_output_root_mean_squared_error: 1.4075
    Epoch 182/500
    614/614 [==============================] - 0s 134us/sample - loss: 1.3874 - y1_output_loss: 0.2611 - y2_output_loss: 1.1261 - y1_output_root_mean_squared_error: 0.5125 - y2_output_root_mean_squared_error: 1.0605 - val_loss: 3.9138 - val_y1_output_loss: 0.5955 - val_y2_output_loss: 3.4431 - val_y1_output_root_mean_squared_error: 0.7619 - val_y2_output_root_mean_squared_error: 1.8257
    Epoch 183/500
    614/614 [==============================] - 0s 127us/sample - loss: 1.2093 - y1_output_loss: 0.2183 - y2_output_loss: 0.9817 - y1_output_root_mean_squared_error: 0.4687 - y2_output_root_mean_squared_error: 0.9948 - val_loss: 1.8919 - val_y1_output_loss: 0.2970 - val_y2_output_loss: 1.5772 - val_y1_output_root_mean_squared_error: 0.5431 - val_y2_output_root_mean_squared_error: 1.2637
    Epoch 184/500
    614/614 [==============================] - 0s 130us/sample - loss: 1.6519 - y1_output_loss: 0.3735 - y2_output_loss: 1.2646 - y1_output_root_mean_squared_error: 0.6137 - y2_output_root_mean_squared_error: 1.1293 - val_loss: 2.0522 - val_y1_output_loss: 0.3653 - val_y2_output_loss: 1.6663 - val_y1_output_root_mean_squared_error: 0.6066 - val_y2_output_root_mean_squared_error: 1.2978
    Epoch 185/500
    614/614 [==============================] - 0s 141us/sample - loss: 1.2893 - y1_output_loss: 0.2407 - y2_output_loss: 1.0504 - y1_output_root_mean_squared_error: 0.4919 - y2_output_root_mean_squared_error: 1.0234 - val_loss: 3.6723 - val_y1_output_loss: 0.6873 - val_y2_output_loss: 3.0304 - val_y1_output_root_mean_squared_error: 0.8110 - val_y2_output_root_mean_squared_error: 1.7363
    Epoch 186/500
    614/614 [==============================] - 0s 131us/sample - loss: 1.7248 - y1_output_loss: 0.3704 - y2_output_loss: 1.3390 - y1_output_root_mean_squared_error: 0.6112 - y2_output_root_mean_squared_error: 1.1624 - val_loss: 2.1448 - val_y1_output_loss: 0.3511 - val_y2_output_loss: 1.7632 - val_y1_output_root_mean_squared_error: 0.5924 - val_y2_output_root_mean_squared_error: 1.3393
    Epoch 187/500
    614/614 [==============================] - 0s 131us/sample - loss: 1.1072 - y1_output_loss: 0.2201 - y2_output_loss: 0.8948 - y1_output_root_mean_squared_error: 0.4687 - y2_output_root_mean_squared_error: 0.9421 - val_loss: 2.2233 - val_y1_output_loss: 0.5061 - val_y2_output_loss: 1.7096 - val_y1_output_root_mean_squared_error: 0.7037 - val_y2_output_root_mean_squared_error: 1.3146
    Epoch 188/500
    614/614 [==============================] - 0s 133us/sample - loss: 1.2186 - y1_output_loss: 0.2306 - y2_output_loss: 1.0080 - y1_output_root_mean_squared_error: 0.4783 - y2_output_root_mean_squared_error: 0.9949 - val_loss: 2.3800 - val_y1_output_loss: 0.3884 - val_y2_output_loss: 1.9651 - val_y1_output_root_mean_squared_error: 0.6168 - val_y2_output_root_mean_squared_error: 1.4141
    Epoch 189/500
    614/614 [==============================] - 0s 137us/sample - loss: 1.1728 - y1_output_loss: 0.2022 - y2_output_loss: 0.9618 - y1_output_root_mean_squared_error: 0.4505 - y2_output_root_mean_squared_error: 0.9848 - val_loss: 1.9558 - val_y1_output_loss: 0.3237 - val_y2_output_loss: 1.6114 - val_y1_output_root_mean_squared_error: 0.5682 - val_y2_output_root_mean_squared_error: 1.2779
    Epoch 190/500
    614/614 [==============================] - 0s 140us/sample - loss: 1.3837 - y1_output_loss: 0.2374 - y2_output_loss: 1.1565 - y1_output_root_mean_squared_error: 0.4830 - y2_output_root_mean_squared_error: 1.0726 - val_loss: 2.8585 - val_y1_output_loss: 0.4526 - val_y2_output_loss: 2.3503 - val_y1_output_root_mean_squared_error: 0.6777 - val_y2_output_root_mean_squared_error: 1.5490
    Epoch 191/500
    614/614 [==============================] - 0s 140us/sample - loss: 1.2597 - y1_output_loss: 0.2173 - y2_output_loss: 1.0317 - y1_output_root_mean_squared_error: 0.4681 - y2_output_root_mean_squared_error: 1.0201 - val_loss: 2.7766 - val_y1_output_loss: 0.6099 - val_y2_output_loss: 2.1561 - val_y1_output_root_mean_squared_error: 0.7810 - val_y2_output_root_mean_squared_error: 1.4719
    Epoch 192/500
    614/614 [==============================] - 0s 132us/sample - loss: 1.3023 - y1_output_loss: 0.2701 - y2_output_loss: 1.0226 - y1_output_root_mean_squared_error: 0.5199 - y2_output_root_mean_squared_error: 1.0159 - val_loss: 1.9558 - val_y1_output_loss: 0.3322 - val_y2_output_loss: 1.6032 - val_y1_output_root_mean_squared_error: 0.5775 - val_y2_output_root_mean_squared_error: 1.2737
    Epoch 193/500
    614/614 [==============================] - 0s 130us/sample - loss: 1.3398 - y1_output_loss: 0.2897 - y2_output_loss: 1.0554 - y1_output_root_mean_squared_error: 0.5348 - y2_output_root_mean_squared_error: 1.0265 - val_loss: 6.0191 - val_y1_output_loss: 1.8854 - val_y2_output_loss: 4.0970 - val_y1_output_root_mean_squared_error: 1.3856 - val_y2_output_root_mean_squared_error: 2.0247
    Epoch 194/500
    614/614 [==============================] - 0s 132us/sample - loss: 1.3088 - y1_output_loss: 0.2909 - y2_output_loss: 1.0209 - y1_output_root_mean_squared_error: 0.5353 - y2_output_root_mean_squared_error: 1.0111 - val_loss: 3.6164 - val_y1_output_loss: 0.7030 - val_y2_output_loss: 2.9025 - val_y1_output_root_mean_squared_error: 0.8301 - val_y2_output_root_mean_squared_error: 1.7109
    Epoch 195/500
    614/614 [==============================] - 0s 137us/sample - loss: 1.4453 - y1_output_loss: 0.2751 - y2_output_loss: 1.1800 - y1_output_root_mean_squared_error: 0.5191 - y2_output_root_mean_squared_error: 1.0844 - val_loss: 4.0367 - val_y1_output_loss: 0.3735 - val_y2_output_loss: 3.6964 - val_y1_output_root_mean_squared_error: 0.6149 - val_y2_output_root_mean_squared_error: 1.9128
    Epoch 196/500
    614/614 [==============================] - 0s 137us/sample - loss: 1.1857 - y1_output_loss: 0.2280 - y2_output_loss: 0.9507 - y1_output_root_mean_squared_error: 0.4789 - y2_output_root_mean_squared_error: 0.9779 - val_loss: 1.8671 - val_y1_output_loss: 0.2969 - val_y2_output_loss: 1.5546 - val_y1_output_root_mean_squared_error: 0.5437 - val_y2_output_root_mean_squared_error: 1.2536
    Epoch 197/500
    614/614 [==============================] - 0s 141us/sample - loss: 1.1677 - y1_output_loss: 0.2209 - y2_output_loss: 0.9390 - y1_output_root_mean_squared_error: 0.4715 - y2_output_root_mean_squared_error: 0.9723 - val_loss: 2.0543 - val_y1_output_loss: 0.3267 - val_y2_output_loss: 1.7190 - val_y1_output_root_mean_squared_error: 0.5673 - val_y2_output_root_mean_squared_error: 1.3162
    Epoch 198/500
    614/614 [==============================] - 0s 141us/sample - loss: 1.3818 - y1_output_loss: 0.2326 - y2_output_loss: 1.1396 - y1_output_root_mean_squared_error: 0.4820 - y2_output_root_mean_squared_error: 1.0721 - val_loss: 2.1655 - val_y1_output_loss: 0.2822 - val_y2_output_loss: 1.8426 - val_y1_output_root_mean_squared_error: 0.5322 - val_y2_output_root_mean_squared_error: 1.3720
    Epoch 199/500
    614/614 [==============================] - 0s 142us/sample - loss: 1.2192 - y1_output_loss: 0.2122 - y2_output_loss: 1.0132 - y1_output_root_mean_squared_error: 0.4579 - y2_output_root_mean_squared_error: 1.0048 - val_loss: 2.6265 - val_y1_output_loss: 0.3670 - val_y2_output_loss: 2.1977 - val_y1_output_root_mean_squared_error: 0.6086 - val_y2_output_root_mean_squared_error: 1.5020
    Epoch 200/500
    614/614 [==============================] - 0s 135us/sample - loss: 1.3469 - y1_output_loss: 0.2508 - y2_output_loss: 1.0952 - y1_output_root_mean_squared_error: 0.5018 - y2_output_root_mean_squared_error: 1.0465 - val_loss: 3.1678 - val_y1_output_loss: 0.3729 - val_y2_output_loss: 2.8544 - val_y1_output_root_mean_squared_error: 0.6089 - val_y2_output_root_mean_squared_error: 1.6724
    Epoch 201/500
    614/614 [==============================] - 0s 139us/sample - loss: 1.1045 - y1_output_loss: 0.2084 - y2_output_loss: 0.8883 - y1_output_root_mean_squared_error: 0.4571 - y2_output_root_mean_squared_error: 0.9463 - val_loss: 1.9229 - val_y1_output_loss: 0.3299 - val_y2_output_loss: 1.5531 - val_y1_output_root_mean_squared_error: 0.5807 - val_y2_output_root_mean_squared_error: 1.2592
    Epoch 202/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.1747 - y1_output_loss: 0.2224 - y2_output_loss: 0.9523 - y1_output_root_mean_squared_error: 0.4716 - y2_output_root_mean_squared_error: 0.9758 - val_loss: 2.3271 - val_y1_output_loss: 0.3972 - val_y2_output_loss: 1.8740 - val_y1_output_root_mean_squared_error: 0.6375 - val_y2_output_root_mean_squared_error: 1.3859
    Epoch 203/500
    614/614 [==============================] - 0s 139us/sample - loss: 1.2407 - y1_output_loss: 0.2482 - y2_output_loss: 0.9862 - y1_output_root_mean_squared_error: 0.4991 - y2_output_root_mean_squared_error: 0.9958 - val_loss: 1.9095 - val_y1_output_loss: 0.3916 - val_y2_output_loss: 1.4958 - val_y1_output_root_mean_squared_error: 0.6275 - val_y2_output_root_mean_squared_error: 1.2311
    Epoch 204/500
    614/614 [==============================] - 0s 136us/sample - loss: 1.1138 - y1_output_loss: 0.2007 - y2_output_loss: 0.9107 - y1_output_root_mean_squared_error: 0.4492 - y2_output_root_mean_squared_error: 0.9550 - val_loss: 2.2980 - val_y1_output_loss: 0.3643 - val_y2_output_loss: 1.9109 - val_y1_output_root_mean_squared_error: 0.6013 - val_y2_output_root_mean_squared_error: 1.3916
    Epoch 205/500
    614/614 [==============================] - 0s 139us/sample - loss: 1.1207 - y1_output_loss: 0.2184 - y2_output_loss: 0.9001 - y1_output_root_mean_squared_error: 0.4685 - y2_output_root_mean_squared_error: 0.9493 - val_loss: 2.3972 - val_y1_output_loss: 0.3065 - val_y2_output_loss: 2.0459 - val_y1_output_root_mean_squared_error: 0.5577 - val_y2_output_root_mean_squared_error: 1.4443
    Epoch 206/500
    614/614 [==============================] - 0s 127us/sample - loss: 1.2925 - y1_output_loss: 0.2505 - y2_output_loss: 1.0620 - y1_output_root_mean_squared_error: 0.5001 - y2_output_root_mean_squared_error: 1.0210 - val_loss: 4.1310 - val_y1_output_loss: 0.6163 - val_y2_output_loss: 3.5856 - val_y1_output_root_mean_squared_error: 0.7895 - val_y2_output_root_mean_squared_error: 1.8729
    Epoch 207/500
    614/614 [==============================] - 0s 138us/sample - loss: 1.1913 - y1_output_loss: 0.2310 - y2_output_loss: 0.9514 - y1_output_root_mean_squared_error: 0.4830 - y2_output_root_mean_squared_error: 0.9788 - val_loss: 2.1774 - val_y1_output_loss: 0.3426 - val_y2_output_loss: 1.8474 - val_y1_output_root_mean_squared_error: 0.5819 - val_y2_output_root_mean_squared_error: 1.3560
    Epoch 208/500
    614/614 [==============================] - 0s 138us/sample - loss: 1.0031 - y1_output_loss: 0.1817 - y2_output_loss: 0.8138 - y1_output_root_mean_squared_error: 0.4266 - y2_output_root_mean_squared_error: 0.9061 - val_loss: 1.7559 - val_y1_output_loss: 0.3259 - val_y2_output_loss: 1.4132 - val_y1_output_root_mean_squared_error: 0.5712 - val_y2_output_root_mean_squared_error: 1.1957
    Epoch 209/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.9531 - y1_output_loss: 0.1907 - y2_output_loss: 0.7538 - y1_output_root_mean_squared_error: 0.4387 - y2_output_root_mean_squared_error: 0.8721 - val_loss: 1.6829 - val_y1_output_loss: 0.2860 - val_y2_output_loss: 1.3927 - val_y1_output_root_mean_squared_error: 0.5306 - val_y2_output_root_mean_squared_error: 1.1838
    Epoch 210/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.0354 - y1_output_loss: 0.1724 - y2_output_loss: 0.8604 - y1_output_root_mean_squared_error: 0.4116 - y2_output_root_mean_squared_error: 0.9306 - val_loss: 1.8774 - val_y1_output_loss: 0.3330 - val_y2_output_loss: 1.5417 - val_y1_output_root_mean_squared_error: 0.5744 - val_y2_output_root_mean_squared_error: 1.2439
    Epoch 211/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.1506 - y1_output_loss: 0.2311 - y2_output_loss: 0.9761 - y1_output_root_mean_squared_error: 0.4814 - y2_output_root_mean_squared_error: 0.9586 - val_loss: 5.7847 - val_y1_output_loss: 0.7565 - val_y2_output_loss: 5.0005 - val_y1_output_root_mean_squared_error: 0.8646 - val_y2_output_root_mean_squared_error: 2.2443
    Epoch 212/500
    614/614 [==============================] - 0s 127us/sample - loss: 1.5077 - y1_output_loss: 0.2963 - y2_output_loss: 1.2017 - y1_output_root_mean_squared_error: 0.5463 - y2_output_root_mean_squared_error: 1.0997 - val_loss: 1.9381 - val_y1_output_loss: 0.2667 - val_y2_output_loss: 1.6681 - val_y1_output_root_mean_squared_error: 0.5108 - val_y2_output_root_mean_squared_error: 1.2951
    Epoch 213/500
    614/614 [==============================] - 0s 127us/sample - loss: 1.5372 - y1_output_loss: 0.3118 - y2_output_loss: 1.2113 - y1_output_root_mean_squared_error: 0.5607 - y2_output_root_mean_squared_error: 1.1058 - val_loss: 1.8914 - val_y1_output_loss: 0.2746 - val_y2_output_loss: 1.5967 - val_y1_output_root_mean_squared_error: 0.5290 - val_y2_output_root_mean_squared_error: 1.2695
    Epoch 214/500
    614/614 [==============================] - 0s 136us/sample - loss: 1.2755 - y1_output_loss: 0.2886 - y2_output_loss: 1.0185 - y1_output_root_mean_squared_error: 0.5372 - y2_output_root_mean_squared_error: 0.9934 - val_loss: 3.5037 - val_y1_output_loss: 0.5577 - val_y2_output_loss: 3.0409 - val_y1_output_root_mean_squared_error: 0.7324 - val_y2_output_root_mean_squared_error: 1.7226
    Epoch 215/500
    614/614 [==============================] - 0s 138us/sample - loss: 1.0152 - y1_output_loss: 0.1684 - y2_output_loss: 0.8615 - y1_output_root_mean_squared_error: 0.4118 - y2_output_root_mean_squared_error: 0.9196 - val_loss: 2.8399 - val_y1_output_loss: 0.3469 - val_y2_output_loss: 2.4347 - val_y1_output_root_mean_squared_error: 0.5973 - val_y2_output_root_mean_squared_error: 1.5758
    Epoch 216/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.9876 - y1_output_loss: 0.1849 - y2_output_loss: 0.8131 - y1_output_root_mean_squared_error: 0.4298 - y2_output_root_mean_squared_error: 0.8960 - val_loss: 2.6939 - val_y1_output_loss: 0.3492 - val_y2_output_loss: 2.3175 - val_y1_output_root_mean_squared_error: 0.5976 - val_y2_output_root_mean_squared_error: 1.5287
    Epoch 217/500
    614/614 [==============================] - 0s 127us/sample - loss: 1.3030 - y1_output_loss: 0.2271 - y2_output_loss: 1.0712 - y1_output_root_mean_squared_error: 0.4778 - y2_output_root_mean_squared_error: 1.0367 - val_loss: 1.8467 - val_y1_output_loss: 0.2624 - val_y2_output_loss: 1.5530 - val_y1_output_root_mean_squared_error: 0.5146 - val_y2_output_root_mean_squared_error: 1.2577
    Epoch 218/500
    614/614 [==============================] - 0s 135us/sample - loss: 0.9172 - y1_output_loss: 0.1620 - y2_output_loss: 0.7656 - y1_output_root_mean_squared_error: 0.4021 - y2_output_root_mean_squared_error: 0.8692 - val_loss: 3.6414 - val_y1_output_loss: 1.0119 - val_y2_output_loss: 2.5254 - val_y1_output_root_mean_squared_error: 1.0210 - val_y2_output_root_mean_squared_error: 1.6121
    Epoch 219/500
    614/614 [==============================] - 0s 130us/sample - loss: 1.4936 - y1_output_loss: 0.3002 - y2_output_loss: 1.1960 - y1_output_root_mean_squared_error: 0.5470 - y2_output_root_mean_squared_error: 1.0929 - val_loss: 2.5874 - val_y1_output_loss: 0.5901 - val_y2_output_loss: 1.9694 - val_y1_output_root_mean_squared_error: 0.7620 - val_y2_output_root_mean_squared_error: 1.4166
    Epoch 220/500
    614/614 [==============================] - 0s 138us/sample - loss: 1.3108 - y1_output_loss: 0.2281 - y2_output_loss: 1.0714 - y1_output_root_mean_squared_error: 0.4791 - y2_output_root_mean_squared_error: 1.0399 - val_loss: 1.6858 - val_y1_output_loss: 0.3208 - val_y2_output_loss: 1.3416 - val_y1_output_root_mean_squared_error: 0.5695 - val_y2_output_root_mean_squared_error: 1.1668
    Epoch 221/500
    614/614 [==============================] - 0s 137us/sample - loss: 1.0038 - y1_output_loss: 0.1957 - y2_output_loss: 0.7995 - y1_output_root_mean_squared_error: 0.4440 - y2_output_root_mean_squared_error: 0.8981 - val_loss: 1.8213 - val_y1_output_loss: 0.3209 - val_y2_output_loss: 1.5071 - val_y1_output_root_mean_squared_error: 0.5644 - val_y2_output_root_mean_squared_error: 1.2259
    Epoch 222/500
    614/614 [==============================] - 0s 137us/sample - loss: 1.0309 - y1_output_loss: 0.2069 - y2_output_loss: 0.8196 - y1_output_root_mean_squared_error: 0.4549 - y2_output_root_mean_squared_error: 0.9077 - val_loss: 2.5169 - val_y1_output_loss: 0.3078 - val_y2_output_loss: 2.1687 - val_y1_output_root_mean_squared_error: 0.5584 - val_y2_output_root_mean_squared_error: 1.4850
    Epoch 223/500
    614/614 [==============================] - 0s 127us/sample - loss: 1.3663 - y1_output_loss: 0.2501 - y2_output_loss: 1.1314 - y1_output_root_mean_squared_error: 0.5017 - y2_output_root_mean_squared_error: 1.0557 - val_loss: 3.4296 - val_y1_output_loss: 0.4185 - val_y2_output_loss: 2.9987 - val_y1_output_root_mean_squared_error: 0.6455 - val_y2_output_root_mean_squared_error: 1.7358
    Epoch 224/500
    614/614 [==============================] - 0s 126us/sample - loss: 1.0585 - y1_output_loss: 0.2128 - y2_output_loss: 0.8551 - y1_output_root_mean_squared_error: 0.4612 - y2_output_root_mean_squared_error: 0.9197 - val_loss: 2.6303 - val_y1_output_loss: 0.5197 - val_y2_output_loss: 2.0650 - val_y1_output_root_mean_squared_error: 0.7244 - val_y2_output_root_mean_squared_error: 1.4511
    Epoch 225/500
    614/614 [==============================] - 0s 129us/sample - loss: 1.2045 - y1_output_loss: 0.2445 - y2_output_loss: 0.9617 - y1_output_root_mean_squared_error: 0.4963 - y2_output_root_mean_squared_error: 0.9789 - val_loss: 2.5784 - val_y1_output_loss: 0.4042 - val_y2_output_loss: 2.2012 - val_y1_output_root_mean_squared_error: 0.6281 - val_y2_output_root_mean_squared_error: 1.4778
    Epoch 226/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.2811 - y1_output_loss: 0.3060 - y2_output_loss: 0.9677 - y1_output_root_mean_squared_error: 0.5537 - y2_output_root_mean_squared_error: 0.9872 - val_loss: 1.7461 - val_y1_output_loss: 0.3465 - val_y2_output_loss: 1.3715 - val_y1_output_root_mean_squared_error: 0.5961 - val_y2_output_root_mean_squared_error: 1.1793
    Epoch 227/500
    614/614 [==============================] - 0s 138us/sample - loss: 2.6490 - y1_output_loss: 0.7530 - y2_output_loss: 1.8883 - y1_output_root_mean_squared_error: 0.8715 - y2_output_root_mean_squared_error: 1.3746 - val_loss: 2.1365 - val_y1_output_loss: 0.4985 - val_y2_output_loss: 1.6043 - val_y1_output_root_mean_squared_error: 0.7126 - val_y2_output_root_mean_squared_error: 1.2762
    Epoch 228/500
    614/614 [==============================] - 0s 141us/sample - loss: 0.9515 - y1_output_loss: 0.1912 - y2_output_loss: 0.7522 - y1_output_root_mean_squared_error: 0.4387 - y2_output_root_mean_squared_error: 0.8712 - val_loss: 1.6445 - val_y1_output_loss: 0.2833 - val_y2_output_loss: 1.3565 - val_y1_output_root_mean_squared_error: 0.5281 - val_y2_output_root_mean_squared_error: 1.1686
    Epoch 229/500
    614/614 [==============================] - 0s 139us/sample - loss: 0.9362 - y1_output_loss: 0.1882 - y2_output_loss: 0.7423 - y1_output_root_mean_squared_error: 0.4352 - y2_output_root_mean_squared_error: 0.8641 - val_loss: 1.7093 - val_y1_output_loss: 0.3021 - val_y2_output_loss: 1.4103 - val_y1_output_root_mean_squared_error: 0.5460 - val_y2_output_root_mean_squared_error: 1.1879
    Epoch 230/500
    614/614 [==============================] - 0s 136us/sample - loss: 0.8316 - y1_output_loss: 0.1669 - y2_output_loss: 0.6584 - y1_output_root_mean_squared_error: 0.4091 - y2_output_root_mean_squared_error: 0.8150 - val_loss: 3.2078 - val_y1_output_loss: 0.3641 - val_y2_output_loss: 2.9212 - val_y1_output_root_mean_squared_error: 0.5945 - val_y2_output_root_mean_squared_error: 1.6895
    Epoch 231/500
    614/614 [==============================] - 0s 131us/sample - loss: 1.2032 - y1_output_loss: 0.2214 - y2_output_loss: 0.9767 - y1_output_root_mean_squared_error: 0.4706 - y2_output_root_mean_squared_error: 0.9909 - val_loss: 1.9774 - val_y1_output_loss: 0.2863 - val_y2_output_loss: 1.6859 - val_y1_output_root_mean_squared_error: 0.5310 - val_y2_output_root_mean_squared_error: 1.3021
    Epoch 232/500
    614/614 [==============================] - 0s 138us/sample - loss: 1.0556 - y1_output_loss: 0.1844 - y2_output_loss: 0.8653 - y1_output_root_mean_squared_error: 0.4299 - y2_output_root_mean_squared_error: 0.9331 - val_loss: 2.0315 - val_y1_output_loss: 0.2941 - val_y2_output_loss: 1.7146 - val_y1_output_root_mean_squared_error: 0.5436 - val_y2_output_root_mean_squared_error: 1.3176
    Epoch 233/500
    614/614 [==============================] - 0s 140us/sample - loss: 0.9241 - y1_output_loss: 0.1850 - y2_output_loss: 0.7482 - y1_output_root_mean_squared_error: 0.4306 - y2_output_root_mean_squared_error: 0.8595 - val_loss: 3.4672 - val_y1_output_loss: 0.5643 - val_y2_output_loss: 2.9453 - val_y1_output_root_mean_squared_error: 0.7540 - val_y2_output_root_mean_squared_error: 1.7026
    Epoch 234/500
    614/614 [==============================] - 0s 138us/sample - loss: 1.1927 - y1_output_loss: 0.2321 - y2_output_loss: 0.9568 - y1_output_root_mean_squared_error: 0.4821 - y2_output_root_mean_squared_error: 0.9799 - val_loss: 2.1949 - val_y1_output_loss: 0.3882 - val_y2_output_loss: 1.7780 - val_y1_output_root_mean_squared_error: 0.6281 - val_y2_output_root_mean_squared_error: 1.3418
    Epoch 235/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.4456 - y1_output_loss: 0.2659 - y2_output_loss: 1.1730 - y1_output_root_mean_squared_error: 0.5155 - y2_output_root_mean_squared_error: 1.0862 - val_loss: 2.3837 - val_y1_output_loss: 0.5433 - val_y2_output_loss: 1.8907 - val_y1_output_root_mean_squared_error: 0.7153 - val_y2_output_root_mean_squared_error: 1.3682
    Epoch 236/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.8491 - y1_output_loss: 0.1857 - y2_output_loss: 0.6781 - y1_output_root_mean_squared_error: 0.4302 - y2_output_root_mean_squared_error: 0.8149 - val_loss: 2.1614 - val_y1_output_loss: 0.2542 - val_y2_output_loss: 1.8758 - val_y1_output_root_mean_squared_error: 0.5082 - val_y2_output_root_mean_squared_error: 1.3795
    Epoch 237/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.9620 - y1_output_loss: 0.1984 - y2_output_loss: 0.7824 - y1_output_root_mean_squared_error: 0.4463 - y2_output_root_mean_squared_error: 0.8734 - val_loss: 4.0788 - val_y1_output_loss: 0.4505 - val_y2_output_loss: 3.5797 - val_y1_output_root_mean_squared_error: 0.6748 - val_y2_output_root_mean_squared_error: 1.9035
    Epoch 238/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.9796 - y1_output_loss: 0.1937 - y2_output_loss: 0.7923 - y1_output_root_mean_squared_error: 0.4379 - y2_output_root_mean_squared_error: 0.8876 - val_loss: 2.4014 - val_y1_output_loss: 0.4011 - val_y2_output_loss: 2.0438 - val_y1_output_root_mean_squared_error: 0.6299 - val_y2_output_root_mean_squared_error: 1.4158
    Epoch 239/500
    614/614 [==============================] - 0s 128us/sample - loss: 1.0466 - y1_output_loss: 0.1976 - y2_output_loss: 0.8510 - y1_output_root_mean_squared_error: 0.4424 - y2_output_root_mean_squared_error: 0.9224 - val_loss: 1.7883 - val_y1_output_loss: 0.3037 - val_y2_output_loss: 1.4639 - val_y1_output_root_mean_squared_error: 0.5541 - val_y2_output_root_mean_squared_error: 1.2171
    Epoch 240/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.7153 - y1_output_loss: 0.1561 - y2_output_loss: 0.5574 - y1_output_root_mean_squared_error: 0.3949 - y2_output_root_mean_squared_error: 0.7479 - val_loss: 2.7334 - val_y1_output_loss: 0.3378 - val_y2_output_loss: 2.4025 - val_y1_output_root_mean_squared_error: 0.5847 - val_y2_output_root_mean_squared_error: 1.5465
    Epoch 241/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.9871 - y1_output_loss: 0.1845 - y2_output_loss: 0.7960 - y1_output_root_mean_squared_error: 0.4311 - y2_output_root_mean_squared_error: 0.8951 - val_loss: 2.3310 - val_y1_output_loss: 0.4365 - val_y2_output_loss: 1.9038 - val_y1_output_root_mean_squared_error: 0.6596 - val_y2_output_root_mean_squared_error: 1.3769
    Epoch 242/500
    614/614 [==============================] - 0s 130us/sample - loss: 1.1134 - y1_output_loss: 0.2359 - y2_output_loss: 0.8693 - y1_output_root_mean_squared_error: 0.4870 - y2_output_root_mean_squared_error: 0.9361 - val_loss: 1.8293 - val_y1_output_loss: 0.3372 - val_y2_output_loss: 1.4915 - val_y1_output_root_mean_squared_error: 0.5817 - val_y2_output_root_mean_squared_error: 1.2210
    Epoch 243/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.9027 - y1_output_loss: 0.1702 - y2_output_loss: 0.7326 - y1_output_root_mean_squared_error: 0.4122 - y2_output_root_mean_squared_error: 0.8560 - val_loss: 2.6233 - val_y1_output_loss: 0.2903 - val_y2_output_loss: 2.2967 - val_y1_output_root_mean_squared_error: 0.5438 - val_y2_output_root_mean_squared_error: 1.5257
    Epoch 244/500
    614/614 [==============================] - 0s 132us/sample - loss: 1.1612 - y1_output_loss: 0.2098 - y2_output_loss: 0.9470 - y1_output_root_mean_squared_error: 0.4581 - y2_output_root_mean_squared_error: 0.9754 - val_loss: 1.7937 - val_y1_output_loss: 0.2754 - val_y2_output_loss: 1.5210 - val_y1_output_root_mean_squared_error: 0.5218 - val_y2_output_root_mean_squared_error: 1.2335
    Epoch 245/500
    614/614 [==============================] - 0s 129us/sample - loss: 1.0123 - y1_output_loss: 0.1900 - y2_output_loss: 0.8139 - y1_output_root_mean_squared_error: 0.4377 - y2_output_root_mean_squared_error: 0.9059 - val_loss: 1.7229 - val_y1_output_loss: 0.2644 - val_y2_output_loss: 1.4710 - val_y1_output_root_mean_squared_error: 0.5061 - val_y2_output_root_mean_squared_error: 1.2111
    Epoch 246/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.7992 - y1_output_loss: 0.1728 - y2_output_loss: 0.6222 - y1_output_root_mean_squared_error: 0.4172 - y2_output_root_mean_squared_error: 0.7907 - val_loss: 1.7498 - val_y1_output_loss: 0.3203 - val_y2_output_loss: 1.4264 - val_y1_output_root_mean_squared_error: 0.5588 - val_y2_output_root_mean_squared_error: 1.1990
    Epoch 247/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.8325 - y1_output_loss: 0.1761 - y2_output_loss: 0.6563 - y1_output_root_mean_squared_error: 0.4187 - y2_output_root_mean_squared_error: 0.8106 - val_loss: 1.5323 - val_y1_output_loss: 0.2380 - val_y2_output_loss: 1.2804 - val_y1_output_root_mean_squared_error: 0.4933 - val_y2_output_root_mean_squared_error: 1.1353
    Epoch 248/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.9111 - y1_output_loss: 0.1779 - y2_output_loss: 0.7277 - y1_output_root_mean_squared_error: 0.4223 - y2_output_root_mean_squared_error: 0.8560 - val_loss: 1.7798 - val_y1_output_loss: 0.3686 - val_y2_output_loss: 1.3829 - val_y1_output_root_mean_squared_error: 0.6170 - val_y2_output_root_mean_squared_error: 1.1829
    Epoch 249/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.9156 - y1_output_loss: 0.1782 - y2_output_loss: 0.7394 - y1_output_root_mean_squared_error: 0.4214 - y2_output_root_mean_squared_error: 0.8591 - val_loss: 2.1083 - val_y1_output_loss: 0.2588 - val_y2_output_loss: 1.8981 - val_y1_output_root_mean_squared_error: 0.5088 - val_y2_output_root_mean_squared_error: 1.3599
    Epoch 250/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.8607 - y1_output_loss: 0.1866 - y2_output_loss: 0.6703 - y1_output_root_mean_squared_error: 0.4319 - y2_output_root_mean_squared_error: 0.8211 - val_loss: 1.6769 - val_y1_output_loss: 0.2772 - val_y2_output_loss: 1.3840 - val_y1_output_root_mean_squared_error: 0.5300 - val_y2_output_root_mean_squared_error: 1.1815
    Epoch 251/500
    614/614 [==============================] - 0s 130us/sample - loss: 1.2028 - y1_output_loss: 0.2174 - y2_output_loss: 0.9770 - y1_output_root_mean_squared_error: 0.4665 - y2_output_root_mean_squared_error: 0.9926 - val_loss: 1.9513 - val_y1_output_loss: 0.3537 - val_y2_output_loss: 1.6185 - val_y1_output_root_mean_squared_error: 0.5937 - val_y2_output_root_mean_squared_error: 1.2645
    Epoch 252/500
    614/614 [==============================] - 0s 145us/sample - loss: 0.9850 - y1_output_loss: 0.2150 - y2_output_loss: 0.7631 - y1_output_root_mean_squared_error: 0.4648 - y2_output_root_mean_squared_error: 0.8769 - val_loss: 1.6102 - val_y1_output_loss: 0.2640 - val_y2_output_loss: 1.3387 - val_y1_output_root_mean_squared_error: 0.5131 - val_y2_output_root_mean_squared_error: 1.1606
    Epoch 253/500
    614/614 [==============================] - 0s 142us/sample - loss: 1.0070 - y1_output_loss: 0.1856 - y2_output_loss: 0.8164 - y1_output_root_mean_squared_error: 0.4308 - y2_output_root_mean_squared_error: 0.9063 - val_loss: 1.9360 - val_y1_output_loss: 0.2497 - val_y2_output_loss: 1.6743 - val_y1_output_root_mean_squared_error: 0.4988 - val_y2_output_root_mean_squared_error: 1.2989
    Epoch 254/500
    614/614 [==============================] - 0s 135us/sample - loss: 0.9938 - y1_output_loss: 0.1793 - y2_output_loss: 0.8116 - y1_output_root_mean_squared_error: 0.4236 - y2_output_root_mean_squared_error: 0.9024 - val_loss: 1.7728 - val_y1_output_loss: 0.3002 - val_y2_output_loss: 1.4454 - val_y1_output_root_mean_squared_error: 0.5459 - val_y2_output_root_mean_squared_error: 1.2144
    Epoch 255/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.8851 - y1_output_loss: 0.1738 - y2_output_loss: 0.7149 - y1_output_root_mean_squared_error: 0.4177 - y2_output_root_mean_squared_error: 0.8430 - val_loss: 4.7019 - val_y1_output_loss: 0.4165 - val_y2_output_loss: 4.3608 - val_y1_output_root_mean_squared_error: 0.6430 - val_y2_output_root_mean_squared_error: 2.0709
    Epoch 256/500
    614/614 [==============================] - 0s 130us/sample - loss: 1.5765 - y1_output_loss: 0.3458 - y2_output_loss: 1.2182 - y1_output_root_mean_squared_error: 0.5904 - y2_output_root_mean_squared_error: 1.1081 - val_loss: 1.6176 - val_y1_output_loss: 0.2494 - val_y2_output_loss: 1.3527 - val_y1_output_root_mean_squared_error: 0.5013 - val_y2_output_root_mean_squared_error: 1.1689
    Epoch 257/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.7258 - y1_output_loss: 0.1670 - y2_output_loss: 0.5546 - y1_output_root_mean_squared_error: 0.4081 - y2_output_root_mean_squared_error: 0.7478 - val_loss: 1.4792 - val_y1_output_loss: 0.2433 - val_y2_output_loss: 1.2352 - val_y1_output_root_mean_squared_error: 0.4909 - val_y2_output_root_mean_squared_error: 1.1128
    Epoch 258/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.6766 - y1_output_loss: 0.1624 - y2_output_loss: 0.5296 - y1_output_root_mean_squared_error: 0.3895 - y2_output_root_mean_squared_error: 0.7245 - val_loss: 4.3364 - val_y1_output_loss: 1.2854 - val_y2_output_loss: 2.9756 - val_y1_output_root_mean_squared_error: 1.1477 - val_y2_output_root_mean_squared_error: 1.7376
    Epoch 259/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.9309 - y1_output_loss: 0.1767 - y2_output_loss: 0.7512 - y1_output_root_mean_squared_error: 0.4220 - y2_output_root_mean_squared_error: 0.8676 - val_loss: 1.4844 - val_y1_output_loss: 0.2790 - val_y2_output_loss: 1.2144 - val_y1_output_root_mean_squared_error: 0.5308 - val_y2_output_root_mean_squared_error: 1.0967
    Epoch 260/500
    614/614 [==============================] - 0s 142us/sample - loss: 0.6960 - y1_output_loss: 0.1387 - y2_output_loss: 0.5583 - y1_output_root_mean_squared_error: 0.3739 - y2_output_root_mean_squared_error: 0.7458 - val_loss: 1.9308 - val_y1_output_loss: 0.2568 - val_y2_output_loss: 1.6897 - val_y1_output_root_mean_squared_error: 0.5037 - val_y2_output_root_mean_squared_error: 1.2950
    Epoch 261/500
    614/614 [==============================] - 0s 142us/sample - loss: 0.7647 - y1_output_loss: 0.1423 - y2_output_loss: 0.6163 - y1_output_root_mean_squared_error: 0.3787 - y2_output_root_mean_squared_error: 0.7882 - val_loss: 1.8238 - val_y1_output_loss: 0.3000 - val_y2_output_loss: 1.5015 - val_y1_output_root_mean_squared_error: 0.5541 - val_y2_output_root_mean_squared_error: 1.2316
    Epoch 262/500
    614/614 [==============================] - 0s 142us/sample - loss: 0.7862 - y1_output_loss: 0.1565 - y2_output_loss: 0.6317 - y1_output_root_mean_squared_error: 0.3951 - y2_output_root_mean_squared_error: 0.7938 - val_loss: 2.2934 - val_y1_output_loss: 0.3119 - val_y2_output_loss: 1.9969 - val_y1_output_root_mean_squared_error: 0.5560 - val_y2_output_root_mean_squared_error: 1.4087
    Epoch 263/500
    614/614 [==============================] - 0s 141us/sample - loss: 0.8942 - y1_output_loss: 0.2043 - y2_output_loss: 0.6829 - y1_output_root_mean_squared_error: 0.4541 - y2_output_root_mean_squared_error: 0.8294 - val_loss: 1.4902 - val_y1_output_loss: 0.2460 - val_y2_output_loss: 1.2474 - val_y1_output_root_mean_squared_error: 0.4959 - val_y2_output_root_mean_squared_error: 1.1154
    Epoch 264/500
    614/614 [==============================] - 0s 142us/sample - loss: 0.8161 - y1_output_loss: 0.1831 - y2_output_loss: 0.6465 - y1_output_root_mean_squared_error: 0.4169 - y2_output_root_mean_squared_error: 0.8015 - val_loss: 3.2680 - val_y1_output_loss: 0.6990 - val_y2_output_loss: 2.5684 - val_y1_output_root_mean_squared_error: 0.8363 - val_y2_output_root_mean_squared_error: 1.6027
    Epoch 265/500
    614/614 [==============================] - 0s 141us/sample - loss: 1.0425 - y1_output_loss: 0.1959 - y2_output_loss: 0.8448 - y1_output_root_mean_squared_error: 0.4431 - y2_output_root_mean_squared_error: 0.9199 - val_loss: 1.8803 - val_y1_output_loss: 0.3685 - val_y2_output_loss: 1.5509 - val_y1_output_root_mean_squared_error: 0.6005 - val_y2_output_root_mean_squared_error: 1.2328
    Epoch 266/500
    614/614 [==============================] - 0s 147us/sample - loss: 1.0915 - y1_output_loss: 0.2276 - y2_output_loss: 0.8536 - y1_output_root_mean_squared_error: 0.4793 - y2_output_root_mean_squared_error: 0.9283 - val_loss: 1.3759 - val_y1_output_loss: 0.2387 - val_y2_output_loss: 1.1521 - val_y1_output_root_mean_squared_error: 0.4912 - val_y2_output_root_mean_squared_error: 1.0652
    Epoch 267/500
    614/614 [==============================] - 0s 142us/sample - loss: 0.8428 - y1_output_loss: 0.1667 - y2_output_loss: 0.6695 - y1_output_root_mean_squared_error: 0.4093 - y2_output_root_mean_squared_error: 0.8217 - val_loss: 1.5934 - val_y1_output_loss: 0.2745 - val_y2_output_loss: 1.3228 - val_y1_output_root_mean_squared_error: 0.5256 - val_y2_output_root_mean_squared_error: 1.1477
    Epoch 268/500
    614/614 [==============================] - 0s 135us/sample - loss: 1.0005 - y1_output_loss: 0.1755 - y2_output_loss: 0.8280 - y1_output_root_mean_squared_error: 0.4177 - y2_output_root_mean_squared_error: 0.9089 - val_loss: 2.9744 - val_y1_output_loss: 0.5018 - val_y2_output_loss: 2.4506 - val_y1_output_root_mean_squared_error: 0.7121 - val_y2_output_root_mean_squared_error: 1.5708
    Epoch 269/500
    614/614 [==============================] - 0s 133us/sample - loss: 0.8556 - y1_output_loss: 0.1885 - y2_output_loss: 0.6664 - y1_output_root_mean_squared_error: 0.4360 - y2_output_root_mean_squared_error: 0.8158 - val_loss: 1.9623 - val_y1_output_loss: 0.3209 - val_y2_output_loss: 1.6545 - val_y1_output_root_mean_squared_error: 0.5680 - val_y2_output_root_mean_squared_error: 1.2805
    Epoch 270/500
    614/614 [==============================] - 0s 141us/sample - loss: 0.8904 - y1_output_loss: 0.1932 - y2_output_loss: 0.6937 - y1_output_root_mean_squared_error: 0.4402 - y2_output_root_mean_squared_error: 0.8347 - val_loss: 1.4052 - val_y1_output_loss: 0.2345 - val_y2_output_loss: 1.1845 - val_y1_output_root_mean_squared_error: 0.4884 - val_y2_output_root_mean_squared_error: 1.0801
    Epoch 271/500
    614/614 [==============================] - 0s 147us/sample - loss: 0.7808 - y1_output_loss: 0.1711 - y2_output_loss: 0.6053 - y1_output_root_mean_squared_error: 0.4138 - y2_output_root_mean_squared_error: 0.7807 - val_loss: 1.6789 - val_y1_output_loss: 0.3156 - val_y2_output_loss: 1.3601 - val_y1_output_root_mean_squared_error: 0.5595 - val_y2_output_root_mean_squared_error: 1.1687
    Epoch 272/500
    614/614 [==============================] - 0s 133us/sample - loss: 0.8683 - y1_output_loss: 0.1620 - y2_output_loss: 0.7019 - y1_output_root_mean_squared_error: 0.4034 - y2_output_root_mean_squared_error: 0.8400 - val_loss: 1.4510 - val_y1_output_loss: 0.3060 - val_y2_output_loss: 1.1598 - val_y1_output_root_mean_squared_error: 0.5513 - val_y2_output_root_mean_squared_error: 1.0710
    Epoch 273/500
    614/614 [==============================] - 0s 131us/sample - loss: 1.4831 - y1_output_loss: 0.3404 - y2_output_loss: 1.1343 - y1_output_root_mean_squared_error: 0.5850 - y2_output_root_mean_squared_error: 1.0681 - val_loss: 1.5269 - val_y1_output_loss: 0.2700 - val_y2_output_loss: 1.2788 - val_y1_output_root_mean_squared_error: 0.5187 - val_y2_output_root_mean_squared_error: 1.1215
    Epoch 274/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.9091 - y1_output_loss: 0.1771 - y2_output_loss: 0.7236 - y1_output_root_mean_squared_error: 0.4227 - y2_output_root_mean_squared_error: 0.8546 - val_loss: 1.5515 - val_y1_output_loss: 0.2788 - val_y2_output_loss: 1.2707 - val_y1_output_root_mean_squared_error: 0.5297 - val_y2_output_root_mean_squared_error: 1.1274
    Epoch 275/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.9147 - y1_output_loss: 0.1807 - y2_output_loss: 0.7562 - y1_output_root_mean_squared_error: 0.4176 - y2_output_root_mean_squared_error: 0.8604 - val_loss: 6.4472 - val_y1_output_loss: 1.2733 - val_y2_output_loss: 5.2655 - val_y1_output_root_mean_squared_error: 1.1265 - val_y2_output_root_mean_squared_error: 2.2756
    Epoch 276/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.9874 - y1_output_loss: 0.1987 - y2_output_loss: 0.7813 - y1_output_root_mean_squared_error: 0.4470 - y2_output_root_mean_squared_error: 0.8874 - val_loss: 1.4328 - val_y1_output_loss: 0.2679 - val_y2_output_loss: 1.1618 - val_y1_output_root_mean_squared_error: 0.5195 - val_y2_output_root_mean_squared_error: 1.0784
    Epoch 277/500
    614/614 [==============================] - 0s 140us/sample - loss: 0.9096 - y1_output_loss: 0.2121 - y2_output_loss: 0.7652 - y1_output_root_mean_squared_error: 0.4349 - y2_output_root_mean_squared_error: 0.8488 - val_loss: 4.8138 - val_y1_output_loss: 1.5039 - val_y2_output_loss: 3.2221 - val_y1_output_root_mean_squared_error: 1.2381 - val_y2_output_root_mean_squared_error: 1.8113
    Epoch 278/500
    614/614 [==============================] - 0s 133us/sample - loss: 1.2443 - y1_output_loss: 0.2768 - y2_output_loss: 0.9581 - y1_output_root_mean_squared_error: 0.5285 - y2_output_root_mean_squared_error: 0.9823 - val_loss: 1.3738 - val_y1_output_loss: 0.2329 - val_y2_output_loss: 1.1445 - val_y1_output_root_mean_squared_error: 0.4823 - val_y2_output_root_mean_squared_error: 1.0683
    Epoch 279/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6732 - y1_output_loss: 0.1434 - y2_output_loss: 0.5270 - y1_output_root_mean_squared_error: 0.3801 - y2_output_root_mean_squared_error: 0.7272 - val_loss: 1.5163 - val_y1_output_loss: 0.2828 - val_y2_output_loss: 1.2192 - val_y1_output_root_mean_squared_error: 0.5372 - val_y2_output_root_mean_squared_error: 1.1080
    Epoch 280/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.7135 - y1_output_loss: 0.1500 - y2_output_loss: 0.5621 - y1_output_root_mean_squared_error: 0.3859 - y2_output_root_mean_squared_error: 0.7513 - val_loss: 2.1064 - val_y1_output_loss: 0.6483 - val_y2_output_loss: 1.4476 - val_y1_output_root_mean_squared_error: 0.8089 - val_y2_output_root_mean_squared_error: 1.2050
    Epoch 281/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.8326 - y1_output_loss: 0.1703 - y2_output_loss: 0.6629 - y1_output_root_mean_squared_error: 0.4091 - y2_output_root_mean_squared_error: 0.8156 - val_loss: 1.6284 - val_y1_output_loss: 0.3478 - val_y2_output_loss: 1.2798 - val_y1_output_root_mean_squared_error: 0.5907 - val_y2_output_root_mean_squared_error: 1.1311
    Epoch 282/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.7792 - y1_output_loss: 0.1546 - y2_output_loss: 0.6195 - y1_output_root_mean_squared_error: 0.3927 - y2_output_root_mean_squared_error: 0.7906 - val_loss: 1.4326 - val_y1_output_loss: 0.2642 - val_y2_output_loss: 1.1715 - val_y1_output_root_mean_squared_error: 0.5122 - val_y2_output_root_mean_squared_error: 1.0818
    Epoch 283/500
    614/614 [==============================] - 0s 142us/sample - loss: 0.7744 - y1_output_loss: 0.1607 - y2_output_loss: 0.6116 - y1_output_root_mean_squared_error: 0.3972 - y2_output_root_mean_squared_error: 0.7852 - val_loss: 1.4540 - val_y1_output_loss: 0.2355 - val_y2_output_loss: 1.2298 - val_y1_output_root_mean_squared_error: 0.4851 - val_y2_output_root_mean_squared_error: 1.1040
    Epoch 284/500
    614/614 [==============================] - 0s 133us/sample - loss: 0.7987 - y1_output_loss: 0.1854 - y2_output_loss: 0.6119 - y1_output_root_mean_squared_error: 0.4289 - y2_output_root_mean_squared_error: 0.7841 - val_loss: 1.4701 - val_y1_output_loss: 0.3041 - val_y2_output_loss: 1.1719 - val_y1_output_root_mean_squared_error: 0.5569 - val_y2_output_root_mean_squared_error: 1.0770
    Epoch 285/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6346 - y1_output_loss: 0.1414 - y2_output_loss: 0.4885 - y1_output_root_mean_squared_error: 0.3775 - y2_output_root_mean_squared_error: 0.7015 - val_loss: 1.4554 - val_y1_output_loss: 0.2520 - val_y2_output_loss: 1.2136 - val_y1_output_root_mean_squared_error: 0.5023 - val_y2_output_root_mean_squared_error: 1.0968
    Epoch 286/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.6930 - y1_output_loss: 0.1564 - y2_output_loss: 0.5512 - y1_output_root_mean_squared_error: 0.3819 - y2_output_root_mean_squared_error: 0.7397 - val_loss: 1.6572 - val_y1_output_loss: 0.4596 - val_y2_output_loss: 1.1850 - val_y1_output_root_mean_squared_error: 0.6847 - val_y2_output_root_mean_squared_error: 1.0901
    Epoch 287/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.9629 - y1_output_loss: 0.2106 - y2_output_loss: 0.7461 - y1_output_root_mean_squared_error: 0.4606 - y2_output_root_mean_squared_error: 0.8664 - val_loss: 1.3964 - val_y1_output_loss: 0.2450 - val_y2_output_loss: 1.1543 - val_y1_output_root_mean_squared_error: 0.4936 - val_y2_output_root_mean_squared_error: 1.0736
    Epoch 288/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.7134 - y1_output_loss: 0.1427 - y2_output_loss: 0.5673 - y1_output_root_mean_squared_error: 0.3780 - y2_output_root_mean_squared_error: 0.7553 - val_loss: 1.5144 - val_y1_output_loss: 0.2629 - val_y2_output_loss: 1.2425 - val_y1_output_root_mean_squared_error: 0.5126 - val_y2_output_root_mean_squared_error: 1.1187
    Epoch 289/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.6716 - y1_output_loss: 0.1370 - y2_output_loss: 0.5288 - y1_output_root_mean_squared_error: 0.3715 - y2_output_root_mean_squared_error: 0.7305 - val_loss: 1.4585 - val_y1_output_loss: 0.2939 - val_y2_output_loss: 1.1701 - val_y1_output_root_mean_squared_error: 0.5403 - val_y2_output_root_mean_squared_error: 1.0801
    Epoch 290/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6359 - y1_output_loss: 0.1466 - y2_output_loss: 0.4834 - y1_output_root_mean_squared_error: 0.3848 - y2_output_root_mean_squared_error: 0.6985 - val_loss: 1.3857 - val_y1_output_loss: 0.2384 - val_y2_output_loss: 1.1435 - val_y1_output_root_mean_squared_error: 0.4923 - val_y2_output_root_mean_squared_error: 1.0693
    Epoch 291/500
    614/614 [==============================] - 0s 131us/sample - loss: 1.4409 - y1_output_loss: 0.2659 - y2_output_loss: 1.1668 - y1_output_root_mean_squared_error: 0.5178 - y2_output_root_mean_squared_error: 1.0830 - val_loss: 1.4560 - val_y1_output_loss: 0.2925 - val_y2_output_loss: 1.1520 - val_y1_output_root_mean_squared_error: 0.5445 - val_y2_output_root_mean_squared_error: 1.0768
    Epoch 292/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6594 - y1_output_loss: 0.1492 - y2_output_loss: 0.5063 - y1_output_root_mean_squared_error: 0.3873 - y2_output_root_mean_squared_error: 0.7137 - val_loss: 1.5911 - val_y1_output_loss: 0.3277 - val_y2_output_loss: 1.2657 - val_y1_output_root_mean_squared_error: 0.5675 - val_y2_output_root_mean_squared_error: 1.1265
    Epoch 293/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6218 - y1_output_loss: 0.1339 - y2_output_loss: 0.4822 - y1_output_root_mean_squared_error: 0.3674 - y2_output_root_mean_squared_error: 0.6977 - val_loss: 1.3881 - val_y1_output_loss: 0.2219 - val_y2_output_loss: 1.1621 - val_y1_output_root_mean_squared_error: 0.4711 - val_y2_output_root_mean_squared_error: 1.0799
    Epoch 294/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6433 - y1_output_loss: 0.1432 - y2_output_loss: 0.5058 - y1_output_root_mean_squared_error: 0.3774 - y2_output_root_mean_squared_error: 0.7077 - val_loss: 2.4414 - val_y1_output_loss: 0.5457 - val_y2_output_loss: 1.9635 - val_y1_output_root_mean_squared_error: 0.7281 - val_y2_output_root_mean_squared_error: 1.3825
    Epoch 295/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.8195 - y1_output_loss: 0.1648 - y2_output_loss: 0.6586 - y1_output_root_mean_squared_error: 0.4064 - y2_output_root_mean_squared_error: 0.8089 - val_loss: 1.6706 - val_y1_output_loss: 0.3180 - val_y2_output_loss: 1.3627 - val_y1_output_root_mean_squared_error: 0.5663 - val_y2_output_root_mean_squared_error: 1.1619
    Epoch 296/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6153 - y1_output_loss: 0.1368 - y2_output_loss: 0.4771 - y1_output_root_mean_squared_error: 0.3682 - y2_output_root_mean_squared_error: 0.6927 - val_loss: 1.4198 - val_y1_output_loss: 0.3095 - val_y2_output_loss: 1.1067 - val_y1_output_root_mean_squared_error: 0.5569 - val_y2_output_root_mean_squared_error: 1.0534
    Epoch 297/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6019 - y1_output_loss: 0.1367 - y2_output_loss: 0.4617 - y1_output_root_mean_squared_error: 0.3709 - y2_output_root_mean_squared_error: 0.6814 - val_loss: 1.2900 - val_y1_output_loss: 0.2309 - val_y2_output_loss: 1.0493 - val_y1_output_root_mean_squared_error: 0.4819 - val_y2_output_root_mean_squared_error: 1.0284
    Epoch 298/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.9023 - y1_output_loss: 0.2112 - y2_output_loss: 0.6830 - y1_output_root_mean_squared_error: 0.4617 - y2_output_root_mean_squared_error: 0.8301 - val_loss: 1.4579 - val_y1_output_loss: 0.2725 - val_y2_output_loss: 1.1786 - val_y1_output_root_mean_squared_error: 0.5254 - val_y2_output_root_mean_squared_error: 1.0872
    Epoch 299/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.7606 - y1_output_loss: 0.1784 - y2_output_loss: 0.5771 - y1_output_root_mean_squared_error: 0.4233 - y2_output_root_mean_squared_error: 0.7625 - val_loss: 1.3619 - val_y1_output_loss: 0.2383 - val_y2_output_loss: 1.1227 - val_y1_output_root_mean_squared_error: 0.4867 - val_y2_output_root_mean_squared_error: 1.0607
    Epoch 300/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6280 - y1_output_loss: 0.1358 - y2_output_loss: 0.4879 - y1_output_root_mean_squared_error: 0.3691 - y2_output_root_mean_squared_error: 0.7012 - val_loss: 1.3622 - val_y1_output_loss: 0.2439 - val_y2_output_loss: 1.1411 - val_y1_output_root_mean_squared_error: 0.4930 - val_y2_output_root_mean_squared_error: 1.0579
    Epoch 301/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.7004 - y1_output_loss: 0.1487 - y2_output_loss: 0.5528 - y1_output_root_mean_squared_error: 0.3843 - y2_output_root_mean_squared_error: 0.7434 - val_loss: 1.5261 - val_y1_output_loss: 0.3025 - val_y2_output_loss: 1.2137 - val_y1_output_root_mean_squared_error: 0.5539 - val_y2_output_root_mean_squared_error: 1.1043
    Epoch 302/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.6556 - y1_output_loss: 0.1536 - y2_output_loss: 0.5044 - y1_output_root_mean_squared_error: 0.3929 - y2_output_root_mean_squared_error: 0.7080 - val_loss: 1.5520 - val_y1_output_loss: 0.2665 - val_y2_output_loss: 1.2872 - val_y1_output_root_mean_squared_error: 0.5165 - val_y2_output_root_mean_squared_error: 1.1337
    Epoch 303/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.9245 - y1_output_loss: 0.2224 - y2_output_loss: 0.7045 - y1_output_root_mean_squared_error: 0.4736 - y2_output_root_mean_squared_error: 0.8368 - val_loss: 2.1627 - val_y1_output_loss: 0.2653 - val_y2_output_loss: 1.8823 - val_y1_output_root_mean_squared_error: 0.5154 - val_y2_output_root_mean_squared_error: 1.3773
    Epoch 304/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.5641 - y1_output_loss: 0.1323 - y2_output_loss: 0.4304 - y1_output_root_mean_squared_error: 0.3632 - y2_output_root_mean_squared_error: 0.6574 - val_loss: 1.4303 - val_y1_output_loss: 0.2561 - val_y2_output_loss: 1.1707 - val_y1_output_root_mean_squared_error: 0.5082 - val_y2_output_root_mean_squared_error: 1.0826
    Epoch 305/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.8588 - y1_output_loss: 0.2081 - y2_output_loss: 0.6541 - y1_output_root_mean_squared_error: 0.4582 - y2_output_root_mean_squared_error: 0.8055 - val_loss: 2.2406 - val_y1_output_loss: 0.2942 - val_y2_output_loss: 1.9396 - val_y1_output_root_mean_squared_error: 0.5444 - val_y2_output_root_mean_squared_error: 1.3944
    Epoch 306/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.8282 - y1_output_loss: 0.1726 - y2_output_loss: 0.6493 - y1_output_root_mean_squared_error: 0.4166 - y2_output_root_mean_squared_error: 0.8091 - val_loss: 1.3783 - val_y1_output_loss: 0.2787 - val_y2_output_loss: 1.1163 - val_y1_output_root_mean_squared_error: 0.5238 - val_y2_output_root_mean_squared_error: 1.0507
    Epoch 307/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.5983 - y1_output_loss: 0.1363 - y2_output_loss: 0.4628 - y1_output_root_mean_squared_error: 0.3661 - y2_output_root_mean_squared_error: 0.6813 - val_loss: 1.4803 - val_y1_output_loss: 0.3591 - val_y2_output_loss: 1.1276 - val_y1_output_root_mean_squared_error: 0.5948 - val_y2_output_root_mean_squared_error: 1.0613
    Epoch 308/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.5658 - y1_output_loss: 0.1369 - y2_output_loss: 0.4341 - y1_output_root_mean_squared_error: 0.3699 - y2_output_root_mean_squared_error: 0.6549 - val_loss: 1.5776 - val_y1_output_loss: 0.4296 - val_y2_output_loss: 1.1476 - val_y1_output_root_mean_squared_error: 0.6563 - val_y2_output_root_mean_squared_error: 1.0709
    Epoch 309/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.7309 - y1_output_loss: 0.2080 - y2_output_loss: 0.5910 - y1_output_root_mean_squared_error: 0.4322 - y2_output_root_mean_squared_error: 0.7376 - val_loss: 11.8221 - val_y1_output_loss: 2.4388 - val_y2_output_loss: 9.1992 - val_y1_output_root_mean_squared_error: 1.5842 - val_y2_output_root_mean_squared_error: 3.0516
    Epoch 310/500
    614/614 [==============================] - 0s 130us/sample - loss: 1.4752 - y1_output_loss: 0.4113 - y2_output_loss: 1.0523 - y1_output_root_mean_squared_error: 0.6439 - y2_output_root_mean_squared_error: 1.0298 - val_loss: 1.3116 - val_y1_output_loss: 0.2278 - val_y2_output_loss: 1.0842 - val_y1_output_root_mean_squared_error: 0.4777 - val_y2_output_root_mean_squared_error: 1.0409
    Epoch 311/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.5213 - y1_output_loss: 0.1314 - y2_output_loss: 0.3866 - y1_output_root_mean_squared_error: 0.3631 - y2_output_root_mean_squared_error: 0.6241 - val_loss: 1.5809 - val_y1_output_loss: 0.2507 - val_y2_output_loss: 1.3191 - val_y1_output_root_mean_squared_error: 0.5053 - val_y2_output_root_mean_squared_error: 1.1513
    Epoch 312/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.6711 - y1_output_loss: 0.1398 - y2_output_loss: 0.5281 - y1_output_root_mean_squared_error: 0.3741 - y2_output_root_mean_squared_error: 0.7288 - val_loss: 1.4109 - val_y1_output_loss: 0.2902 - val_y2_output_loss: 1.1103 - val_y1_output_root_mean_squared_error: 0.5408 - val_y2_output_root_mean_squared_error: 1.0576
    Epoch 313/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.5734 - y1_output_loss: 0.1302 - y2_output_loss: 0.4405 - y1_output_root_mean_squared_error: 0.3610 - y2_output_root_mean_squared_error: 0.6656 - val_loss: 1.4110 - val_y1_output_loss: 0.2931 - val_y2_output_loss: 1.1038 - val_y1_output_root_mean_squared_error: 0.5446 - val_y2_output_root_mean_squared_error: 1.0557
    Epoch 314/500
    614/614 [==============================] - 0s 142us/sample - loss: 0.6529 - y1_output_loss: 0.1474 - y2_output_loss: 0.5116 - y1_output_root_mean_squared_error: 0.3797 - y2_output_root_mean_squared_error: 0.7132 - val_loss: 2.5652 - val_y1_output_loss: 0.5308 - val_y2_output_loss: 2.0411 - val_y1_output_root_mean_squared_error: 0.7281 - val_y2_output_root_mean_squared_error: 1.4266
    Epoch 315/500
    614/614 [==============================] - 0s 140us/sample - loss: 0.5520 - y1_output_loss: 0.1223 - y2_output_loss: 0.4253 - y1_output_root_mean_squared_error: 0.3509 - y2_output_root_mean_squared_error: 0.6549 - val_loss: 1.3877 - val_y1_output_loss: 0.2780 - val_y2_output_loss: 1.0970 - val_y1_output_root_mean_squared_error: 0.5272 - val_y2_output_root_mean_squared_error: 1.0535
    Epoch 316/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6083 - y1_output_loss: 0.1373 - y2_output_loss: 0.4665 - y1_output_root_mean_squared_error: 0.3715 - y2_output_root_mean_squared_error: 0.6858 - val_loss: 1.2583 - val_y1_output_loss: 0.2305 - val_y2_output_loss: 1.0212 - val_y1_output_root_mean_squared_error: 0.4820 - val_y2_output_root_mean_squared_error: 1.0129
    Epoch 317/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.5715 - y1_output_loss: 0.1244 - y2_output_loss: 0.4573 - y1_output_root_mean_squared_error: 0.3529 - y2_output_root_mean_squared_error: 0.6686 - val_loss: 4.2375 - val_y1_output_loss: 0.6016 - val_y2_output_loss: 3.6888 - val_y1_output_root_mean_squared_error: 0.7689 - val_y2_output_root_mean_squared_error: 1.9095
    Epoch 318/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.6090 - y1_output_loss: 0.1319 - y2_output_loss: 0.4758 - y1_output_root_mean_squared_error: 0.3638 - y2_output_root_mean_squared_error: 0.6904 - val_loss: 1.3628 - val_y1_output_loss: 0.2495 - val_y2_output_loss: 1.1211 - val_y1_output_root_mean_squared_error: 0.5016 - val_y2_output_root_mean_squared_error: 1.0541
    Epoch 319/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.7084 - y1_output_loss: 0.1469 - y2_output_loss: 0.5570 - y1_output_root_mean_squared_error: 0.3847 - y2_output_root_mean_squared_error: 0.7486 - val_loss: 1.4124 - val_y1_output_loss: 0.2943 - val_y2_output_loss: 1.1362 - val_y1_output_root_mean_squared_error: 0.5333 - val_y2_output_root_mean_squared_error: 1.0621
    Epoch 320/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.6453 - y1_output_loss: 0.1402 - y2_output_loss: 0.5071 - y1_output_root_mean_squared_error: 0.3758 - y2_output_root_mean_squared_error: 0.7100 - val_loss: 1.7547 - val_y1_output_loss: 0.3068 - val_y2_output_loss: 1.4486 - val_y1_output_root_mean_squared_error: 0.5543 - val_y2_output_root_mean_squared_error: 1.2031
    Epoch 321/500
    614/614 [==============================] - 0s 127us/sample - loss: 0.8317 - y1_output_loss: 0.2004 - y2_output_loss: 0.6327 - y1_output_root_mean_squared_error: 0.4482 - y2_output_root_mean_squared_error: 0.7942 - val_loss: 1.8005 - val_y1_output_loss: 0.3259 - val_y2_output_loss: 1.4825 - val_y1_output_root_mean_squared_error: 0.5676 - val_y2_output_root_mean_squared_error: 1.2159
    Epoch 322/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.5703 - y1_output_loss: 0.1278 - y2_output_loss: 0.4422 - y1_output_root_mean_squared_error: 0.3577 - y2_output_root_mean_squared_error: 0.6651 - val_loss: 1.3656 - val_y1_output_loss: 0.2463 - val_y2_output_loss: 1.1067 - val_y1_output_root_mean_squared_error: 0.5013 - val_y2_output_root_mean_squared_error: 1.0556
    Epoch 323/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.6024 - y1_output_loss: 0.1359 - y2_output_loss: 0.4638 - y1_output_root_mean_squared_error: 0.3683 - y2_output_root_mean_squared_error: 0.6832 - val_loss: 1.3428 - val_y1_output_loss: 0.2424 - val_y2_output_loss: 1.0883 - val_y1_output_root_mean_squared_error: 0.4970 - val_y2_output_root_mean_squared_error: 1.0468
    Epoch 324/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.5371 - y1_output_loss: 0.1334 - y2_output_loss: 0.4006 - y1_output_root_mean_squared_error: 0.3650 - y2_output_root_mean_squared_error: 0.6355 - val_loss: 1.2816 - val_y1_output_loss: 0.2663 - val_y2_output_loss: 1.0078 - val_y1_output_root_mean_squared_error: 0.5206 - val_y2_output_root_mean_squared_error: 1.0053
    Epoch 325/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.4613 - y1_output_loss: 0.1191 - y2_output_loss: 0.3442 - y1_output_root_mean_squared_error: 0.3434 - y2_output_root_mean_squared_error: 0.5859 - val_loss: 1.7976 - val_y1_output_loss: 0.4178 - val_y2_output_loss: 1.3548 - val_y1_output_root_mean_squared_error: 0.6542 - val_y2_output_root_mean_squared_error: 1.1703
    Epoch 326/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.5402 - y1_output_loss: 0.1304 - y2_output_loss: 0.4050 - y1_output_root_mean_squared_error: 0.3626 - y2_output_root_mean_squared_error: 0.6393 - val_loss: 1.3515 - val_y1_output_loss: 0.2612 - val_y2_output_loss: 1.1022 - val_y1_output_root_mean_squared_error: 0.5151 - val_y2_output_root_mean_squared_error: 1.0422
    Epoch 327/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6606 - y1_output_loss: 0.1391 - y2_output_loss: 0.5160 - y1_output_root_mean_squared_error: 0.3746 - y2_output_root_mean_squared_error: 0.7213 - val_loss: 1.3616 - val_y1_output_loss: 0.2395 - val_y2_output_loss: 1.1101 - val_y1_output_root_mean_squared_error: 0.4918 - val_y2_output_root_mean_squared_error: 1.0582
    Epoch 328/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.5927 - y1_output_loss: 0.1382 - y2_output_loss: 0.4521 - y1_output_root_mean_squared_error: 0.3694 - y2_output_root_mean_squared_error: 0.6754 - val_loss: 1.5347 - val_y1_output_loss: 0.3681 - val_y2_output_loss: 1.1492 - val_y1_output_root_mean_squared_error: 0.6142 - val_y2_output_root_mean_squared_error: 1.0759
    Epoch 329/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.8480 - y1_output_loss: 0.1641 - y2_output_loss: 0.6782 - y1_output_root_mean_squared_error: 0.4069 - y2_output_root_mean_squared_error: 0.8261 - val_loss: 1.2984 - val_y1_output_loss: 0.2482 - val_y2_output_loss: 1.0617 - val_y1_output_root_mean_squared_error: 0.4949 - val_y2_output_root_mean_squared_error: 1.0264
    Epoch 330/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.4995 - y1_output_loss: 0.1367 - y2_output_loss: 0.3603 - y1_output_root_mean_squared_error: 0.3711 - y2_output_root_mean_squared_error: 0.6015 - val_loss: 1.2023 - val_y1_output_loss: 0.2321 - val_y2_output_loss: 0.9573 - val_y1_output_root_mean_squared_error: 0.4878 - val_y2_output_root_mean_squared_error: 0.9820
    Epoch 331/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.4829 - y1_output_loss: 0.1274 - y2_output_loss: 0.3855 - y1_output_root_mean_squared_error: 0.3449 - y2_output_root_mean_squared_error: 0.6033 - val_loss: 2.5796 - val_y1_output_loss: 0.6024 - val_y2_output_loss: 1.9674 - val_y1_output_root_mean_squared_error: 0.7874 - val_y2_output_root_mean_squared_error: 1.3999
    Epoch 332/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.5756 - y1_output_loss: 0.1215 - y2_output_loss: 0.4538 - y1_output_root_mean_squared_error: 0.3486 - y2_output_root_mean_squared_error: 0.6739 - val_loss: 2.1868 - val_y1_output_loss: 0.3590 - val_y2_output_loss: 1.8311 - val_y1_output_root_mean_squared_error: 0.6041 - val_y2_output_root_mean_squared_error: 1.3498
    Epoch 333/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.6565 - y1_output_loss: 0.1343 - y2_output_loss: 0.5169 - y1_output_root_mean_squared_error: 0.3679 - y2_output_root_mean_squared_error: 0.7219 - val_loss: 1.1680 - val_y1_output_loss: 0.2349 - val_y2_output_loss: 0.9202 - val_y1_output_root_mean_squared_error: 0.4888 - val_y2_output_root_mean_squared_error: 0.9639
    Epoch 334/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6199 - y1_output_loss: 0.1362 - y2_output_loss: 0.4787 - y1_output_root_mean_squared_error: 0.3703 - y2_output_root_mean_squared_error: 0.6949 - val_loss: 1.5730 - val_y1_output_loss: 0.3833 - val_y2_output_loss: 1.1551 - val_y1_output_root_mean_squared_error: 0.6287 - val_y2_output_root_mean_squared_error: 1.0852
    Epoch 335/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.5939 - y1_output_loss: 0.1544 - y2_output_loss: 0.4428 - y1_output_root_mean_squared_error: 0.3926 - y2_output_root_mean_squared_error: 0.6632 - val_loss: 2.0103 - val_y1_output_loss: 0.3243 - val_y2_output_loss: 1.6960 - val_y1_output_root_mean_squared_error: 0.5651 - val_y2_output_root_mean_squared_error: 1.3004
    Epoch 336/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.6279 - y1_output_loss: 0.1540 - y2_output_loss: 0.4795 - y1_output_root_mean_squared_error: 0.3930 - y2_output_root_mean_squared_error: 0.6881 - val_loss: 1.2760 - val_y1_output_loss: 0.2553 - val_y2_output_loss: 1.0311 - val_y1_output_root_mean_squared_error: 0.5074 - val_y2_output_root_mean_squared_error: 1.0093
    Epoch 337/500
    614/614 [==============================] - 0s 133us/sample - loss: 0.4754 - y1_output_loss: 0.1170 - y2_output_loss: 0.3577 - y1_output_root_mean_squared_error: 0.3428 - y2_output_root_mean_squared_error: 0.5982 - val_loss: 1.2091 - val_y1_output_loss: 0.2293 - val_y2_output_loss: 0.9854 - val_y1_output_root_mean_squared_error: 0.4767 - val_y2_output_root_mean_squared_error: 0.9909
    Epoch 338/500
    614/614 [==============================] - 0s 133us/sample - loss: 0.4918 - y1_output_loss: 0.1168 - y2_output_loss: 0.3730 - y1_output_root_mean_squared_error: 0.3420 - y2_output_root_mean_squared_error: 0.6123 - val_loss: 1.1949 - val_y1_output_loss: 0.2448 - val_y2_output_loss: 0.9411 - val_y1_output_root_mean_squared_error: 0.4988 - val_y2_output_root_mean_squared_error: 0.9726
    Epoch 339/500
    614/614 [==============================] - 0s 134us/sample - loss: 0.4776 - y1_output_loss: 0.1253 - y2_output_loss: 0.3531 - y1_output_root_mean_squared_error: 0.3505 - y2_output_root_mean_squared_error: 0.5956 - val_loss: 1.3511 - val_y1_output_loss: 0.2291 - val_y2_output_loss: 1.1251 - val_y1_output_root_mean_squared_error: 0.4806 - val_y2_output_root_mean_squared_error: 1.0584
    Epoch 340/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.5038 - y1_output_loss: 0.1228 - y2_output_loss: 0.3781 - y1_output_root_mean_squared_error: 0.3512 - y2_output_root_mean_squared_error: 0.6168 - val_loss: 1.2969 - val_y1_output_loss: 0.2230 - val_y2_output_loss: 1.0566 - val_y1_output_root_mean_squared_error: 0.4730 - val_y2_output_root_mean_squared_error: 1.0359
    Epoch 341/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.6793 - y1_output_loss: 0.1680 - y2_output_loss: 0.5076 - y1_output_root_mean_squared_error: 0.4092 - y2_output_root_mean_squared_error: 0.7154 - val_loss: 1.1201 - val_y1_output_loss: 0.2345 - val_y2_output_loss: 0.8848 - val_y1_output_root_mean_squared_error: 0.4868 - val_y2_output_root_mean_squared_error: 0.9397
    Epoch 342/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.5329 - y1_output_loss: 0.1287 - y2_output_loss: 0.4009 - y1_output_root_mean_squared_error: 0.3597 - y2_output_root_mean_squared_error: 0.6353 - val_loss: 1.1850 - val_y1_output_loss: 0.2246 - val_y2_output_loss: 0.9631 - val_y1_output_root_mean_squared_error: 0.4740 - val_y2_output_root_mean_squared_error: 0.9800
    Epoch 343/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.5907 - y1_output_loss: 0.1435 - y2_output_loss: 0.4474 - y1_output_root_mean_squared_error: 0.3793 - y2_output_root_mean_squared_error: 0.6685 - val_loss: 1.3276 - val_y1_output_loss: 0.2429 - val_y2_output_loss: 1.0540 - val_y1_output_root_mean_squared_error: 0.4982 - val_y2_output_root_mean_squared_error: 1.0390
    Epoch 344/500
    614/614 [==============================] - 0s 129us/sample - loss: 1.5875 - y1_output_loss: 0.3387 - y2_output_loss: 1.2351 - y1_output_root_mean_squared_error: 0.5845 - y2_output_root_mean_squared_error: 1.1162 - val_loss: 1.3705 - val_y1_output_loss: 0.3463 - val_y2_output_loss: 1.0014 - val_y1_output_root_mean_squared_error: 0.5942 - val_y2_output_root_mean_squared_error: 1.0087
    Epoch 345/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.6907 - y1_output_loss: 0.1758 - y2_output_loss: 0.5090 - y1_output_root_mean_squared_error: 0.4207 - y2_output_root_mean_squared_error: 0.7167 - val_loss: 1.2088 - val_y1_output_loss: 0.2331 - val_y2_output_loss: 0.9772 - val_y1_output_root_mean_squared_error: 0.4848 - val_y2_output_root_mean_squared_error: 0.9868
    Epoch 346/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.4835 - y1_output_loss: 0.1071 - y2_output_loss: 0.3725 - y1_output_root_mean_squared_error: 0.3286 - y2_output_root_mean_squared_error: 0.6128 - val_loss: 1.1934 - val_y1_output_loss: 0.2421 - val_y2_output_loss: 0.9400 - val_y1_output_root_mean_squared_error: 0.4967 - val_y2_output_root_mean_squared_error: 0.9729
    Epoch 347/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.6103 - y1_output_loss: 0.1336 - y2_output_loss: 0.4892 - y1_output_root_mean_squared_error: 0.3624 - y2_output_root_mean_squared_error: 0.6921 - val_loss: 1.8223 - val_y1_output_loss: 0.3260 - val_y2_output_loss: 1.4947 - val_y1_output_root_mean_squared_error: 0.5761 - val_y2_output_root_mean_squared_error: 1.2208
    Epoch 348/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.5803 - y1_output_loss: 0.1352 - y2_output_loss: 0.4412 - y1_output_root_mean_squared_error: 0.3684 - y2_output_root_mean_squared_error: 0.6667 - val_loss: 1.3635 - val_y1_output_loss: 0.2223 - val_y2_output_loss: 1.1465 - val_y1_output_root_mean_squared_error: 0.4733 - val_y2_output_root_mean_squared_error: 1.0675
    Epoch 349/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.5815 - y1_output_loss: 0.1329 - y2_output_loss: 0.4449 - y1_output_root_mean_squared_error: 0.3656 - y2_output_root_mean_squared_error: 0.6692 - val_loss: 1.1870 - val_y1_output_loss: 0.2274 - val_y2_output_loss: 0.9524 - val_y1_output_root_mean_squared_error: 0.4792 - val_y2_output_root_mean_squared_error: 0.9785
    Epoch 350/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.5194 - y1_output_loss: 0.1271 - y2_output_loss: 0.3880 - y1_output_root_mean_squared_error: 0.3580 - y2_output_root_mean_squared_error: 0.6255 - val_loss: 1.2834 - val_y1_output_loss: 0.2315 - val_y2_output_loss: 1.0758 - val_y1_output_root_mean_squared_error: 0.4794 - val_y2_output_root_mean_squared_error: 1.0264
    Epoch 351/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.5628 - y1_output_loss: 0.1267 - y2_output_loss: 0.4393 - y1_output_root_mean_squared_error: 0.3488 - y2_output_root_mean_squared_error: 0.6642 - val_loss: 2.1366 - val_y1_output_loss: 0.7549 - val_y2_output_loss: 1.3447 - val_y1_output_root_mean_squared_error: 0.8763 - val_y2_output_root_mean_squared_error: 1.1699
    Epoch 352/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.4869 - y1_output_loss: 0.1351 - y2_output_loss: 0.3503 - y1_output_root_mean_squared_error: 0.3686 - y2_output_root_mean_squared_error: 0.5925 - val_loss: 1.3853 - val_y1_output_loss: 0.2288 - val_y2_output_loss: 1.1356 - val_y1_output_root_mean_squared_error: 0.4806 - val_y2_output_root_mean_squared_error: 1.0744
    Epoch 353/500
    614/614 [==============================] - 0s 127us/sample - loss: 0.6122 - y1_output_loss: 0.1308 - y2_output_loss: 0.4771 - y1_output_root_mean_squared_error: 0.3623 - y2_output_root_mean_squared_error: 0.6935 - val_loss: 1.3329 - val_y1_output_loss: 0.3030 - val_y2_output_loss: 1.0029 - val_y1_output_root_mean_squared_error: 0.5585 - val_y2_output_root_mean_squared_error: 1.0104
    Epoch 354/500
    614/614 [==============================] - 0s 127us/sample - loss: 0.6097 - y1_output_loss: 0.1758 - y2_output_loss: 0.4315 - y1_output_root_mean_squared_error: 0.4207 - y2_output_root_mean_squared_error: 0.6578 - val_loss: 1.3396 - val_y1_output_loss: 0.2108 - val_y2_output_loss: 1.1200 - val_y1_output_root_mean_squared_error: 0.4620 - val_y2_output_root_mean_squared_error: 1.0612
    Epoch 355/500
    614/614 [==============================] - 0s 133us/sample - loss: 0.5644 - y1_output_loss: 0.1422 - y2_output_loss: 0.4175 - y1_output_root_mean_squared_error: 0.3785 - y2_output_root_mean_squared_error: 0.6490 - val_loss: 1.1241 - val_y1_output_loss: 0.2283 - val_y2_output_loss: 0.8822 - val_y1_output_root_mean_squared_error: 0.4832 - val_y2_output_root_mean_squared_error: 0.9438
    Epoch 356/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.4629 - y1_output_loss: 0.1087 - y2_output_loss: 0.3585 - y1_output_root_mean_squared_error: 0.3308 - y2_output_root_mean_squared_error: 0.5945 - val_loss: 1.8995 - val_y1_output_loss: 0.2646 - val_y2_output_loss: 1.6078 - val_y1_output_root_mean_squared_error: 0.5206 - val_y2_output_root_mean_squared_error: 1.2761
    Epoch 357/500
    614/614 [==============================] - 0s 127us/sample - loss: 0.6421 - y1_output_loss: 0.1459 - y2_output_loss: 0.4921 - y1_output_root_mean_squared_error: 0.3812 - y2_output_root_mean_squared_error: 0.7048 - val_loss: 1.3004 - val_y1_output_loss: 0.2695 - val_y2_output_loss: 1.0394 - val_y1_output_root_mean_squared_error: 0.5195 - val_y2_output_root_mean_squared_error: 1.0151
    Epoch 358/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.4872 - y1_output_loss: 0.1216 - y2_output_loss: 0.3696 - y1_output_root_mean_squared_error: 0.3490 - y2_output_root_mean_squared_error: 0.6045 - val_loss: 1.4122 - val_y1_output_loss: 0.2889 - val_y2_output_loss: 1.1185 - val_y1_output_root_mean_squared_error: 0.5389 - val_y2_output_root_mean_squared_error: 1.0592
    Epoch 359/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.8968 - y1_output_loss: 0.1861 - y2_output_loss: 0.7031 - y1_output_root_mean_squared_error: 0.4327 - y2_output_root_mean_squared_error: 0.8424 - val_loss: 1.1743 - val_y1_output_loss: 0.2567 - val_y2_output_loss: 0.9130 - val_y1_output_root_mean_squared_error: 0.5104 - val_y2_output_root_mean_squared_error: 0.9560
    Epoch 360/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.5471 - y1_output_loss: 0.1209 - y2_output_loss: 0.4226 - y1_output_root_mean_squared_error: 0.3484 - y2_output_root_mean_squared_error: 0.6525 - val_loss: 1.3505 - val_y1_output_loss: 0.2709 - val_y2_output_loss: 1.0594 - val_y1_output_root_mean_squared_error: 0.5261 - val_y2_output_root_mean_squared_error: 1.0362
    Epoch 361/500
    614/614 [==============================] - 0s 136us/sample - loss: 0.6491 - y1_output_loss: 0.1389 - y2_output_loss: 0.5076 - y1_output_root_mean_squared_error: 0.3723 - y2_output_root_mean_squared_error: 0.7145 - val_loss: 2.3603 - val_y1_output_loss: 0.4847 - val_y2_output_loss: 1.8868 - val_y1_output_root_mean_squared_error: 0.6943 - val_y2_output_root_mean_squared_error: 1.3705
    Epoch 362/500
    614/614 [==============================] - 0s 133us/sample - loss: 0.5137 - y1_output_loss: 0.1252 - y2_output_loss: 0.3851 - y1_output_root_mean_squared_error: 0.3540 - y2_output_root_mean_squared_error: 0.6232 - val_loss: 1.2520 - val_y1_output_loss: 0.2552 - val_y2_output_loss: 0.9784 - val_y1_output_root_mean_squared_error: 0.5105 - val_y2_output_root_mean_squared_error: 0.9957
    Epoch 363/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.5605 - y1_output_loss: 0.1342 - y2_output_loss: 0.4246 - y1_output_root_mean_squared_error: 0.3672 - y2_output_root_mean_squared_error: 0.6524 - val_loss: 1.1787 - val_y1_output_loss: 0.2381 - val_y2_output_loss: 0.9284 - val_y1_output_root_mean_squared_error: 0.4913 - val_y2_output_root_mean_squared_error: 0.9681
    Epoch 364/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.4741 - y1_output_loss: 0.1222 - y2_output_loss: 0.3507 - y1_output_root_mean_squared_error: 0.3483 - y2_output_root_mean_squared_error: 0.5940 - val_loss: 1.5011 - val_y1_output_loss: 0.3011 - val_y2_output_loss: 1.1769 - val_y1_output_root_mean_squared_error: 0.5515 - val_y2_output_root_mean_squared_error: 1.0941
    Epoch 365/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.5295 - y1_output_loss: 0.1551 - y2_output_loss: 0.3717 - y1_output_root_mean_squared_error: 0.3945 - y2_output_root_mean_squared_error: 0.6115 - val_loss: 1.2060 - val_y1_output_loss: 0.2439 - val_y2_output_loss: 0.9677 - val_y1_output_root_mean_squared_error: 0.4945 - val_y2_output_root_mean_squared_error: 0.9806
    Epoch 366/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.5124 - y1_output_loss: 0.1189 - y2_output_loss: 0.3956 - y1_output_root_mean_squared_error: 0.3426 - y2_output_root_mean_squared_error: 0.6285 - val_loss: 1.7894 - val_y1_output_loss: 0.3121 - val_y2_output_loss: 1.4819 - val_y1_output_root_mean_squared_error: 0.5537 - val_y2_output_root_mean_squared_error: 1.2177
    Epoch 367/500
    614/614 [==============================] - 0s 136us/sample - loss: 0.5938 - y1_output_loss: 0.1390 - y2_output_loss: 0.4503 - y1_output_root_mean_squared_error: 0.3741 - y2_output_root_mean_squared_error: 0.6736 - val_loss: 1.1451 - val_y1_output_loss: 0.2460 - val_y2_output_loss: 0.8934 - val_y1_output_root_mean_squared_error: 0.5011 - val_y2_output_root_mean_squared_error: 0.9455
    Epoch 368/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.7523 - y1_output_loss: 0.1690 - y2_output_loss: 0.5790 - y1_output_root_mean_squared_error: 0.4120 - y2_output_root_mean_squared_error: 0.7632 - val_loss: 1.3477 - val_y1_output_loss: 0.2604 - val_y2_output_loss: 1.0789 - val_y1_output_root_mean_squared_error: 0.5123 - val_y2_output_root_mean_squared_error: 1.0418
    Epoch 369/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.4853 - y1_output_loss: 0.1334 - y2_output_loss: 0.3522 - y1_output_root_mean_squared_error: 0.3653 - y2_output_root_mean_squared_error: 0.5932 - val_loss: 1.1275 - val_y1_output_loss: 0.2671 - val_y2_output_loss: 0.8524 - val_y1_output_root_mean_squared_error: 0.5223 - val_y2_output_root_mean_squared_error: 0.9245
    Epoch 370/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.5805 - y1_output_loss: 0.1474 - y2_output_loss: 0.4284 - y1_output_root_mean_squared_error: 0.3851 - y2_output_root_mean_squared_error: 0.6574 - val_loss: 1.1236 - val_y1_output_loss: 0.2358 - val_y2_output_loss: 0.8871 - val_y1_output_root_mean_squared_error: 0.4899 - val_y2_output_root_mean_squared_error: 0.9400
    Epoch 371/500
    614/614 [==============================] - 0s 136us/sample - loss: 0.5049 - y1_output_loss: 0.1147 - y2_output_loss: 0.3861 - y1_output_root_mean_squared_error: 0.3396 - y2_output_root_mean_squared_error: 0.6241 - val_loss: 1.1989 - val_y1_output_loss: 0.2477 - val_y2_output_loss: 0.9403 - val_y1_output_root_mean_squared_error: 0.5005 - val_y2_output_root_mean_squared_error: 0.9738
    Epoch 372/500
    614/614 [==============================] - 0s 127us/sample - loss: 0.4510 - y1_output_loss: 0.1156 - y2_output_loss: 0.3314 - y1_output_root_mean_squared_error: 0.3414 - y2_output_root_mean_squared_error: 0.5784 - val_loss: 1.1313 - val_y1_output_loss: 0.2308 - val_y2_output_loss: 0.8906 - val_y1_output_root_mean_squared_error: 0.4827 - val_y2_output_root_mean_squared_error: 0.9478
    Epoch 373/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.4210 - y1_output_loss: 0.1264 - y2_output_loss: 0.2926 - y1_output_root_mean_squared_error: 0.3568 - y2_output_root_mean_squared_error: 0.5419 - val_loss: 1.3102 - val_y1_output_loss: 0.2607 - val_y2_output_loss: 1.0336 - val_y1_output_root_mean_squared_error: 0.5157 - val_y2_output_root_mean_squared_error: 1.0219
    Epoch 374/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.4852 - y1_output_loss: 0.1339 - y2_output_loss: 0.3529 - y1_output_root_mean_squared_error: 0.3618 - y2_output_root_mean_squared_error: 0.5953 - val_loss: 3.0508 - val_y1_output_loss: 0.8931 - val_y2_output_loss: 2.0992 - val_y1_output_root_mean_squared_error: 0.9605 - val_y2_output_root_mean_squared_error: 1.4589
    Epoch 375/500
    614/614 [==============================] - 0s 127us/sample - loss: 0.5366 - y1_output_loss: 0.1422 - y2_output_loss: 0.3903 - y1_output_root_mean_squared_error: 0.3782 - y2_output_root_mean_squared_error: 0.6273 - val_loss: 1.1034 - val_y1_output_loss: 0.2161 - val_y2_output_loss: 0.8810 - val_y1_output_root_mean_squared_error: 0.4660 - val_y2_output_root_mean_squared_error: 0.9414
    Epoch 376/500
    614/614 [==============================] - 0s 127us/sample - loss: 0.4717 - y1_output_loss: 0.1229 - y2_output_loss: 0.3452 - y1_output_root_mean_squared_error: 0.3522 - y2_output_root_mean_squared_error: 0.5897 - val_loss: 1.1834 - val_y1_output_loss: 0.2314 - val_y2_output_loss: 0.9469 - val_y1_output_root_mean_squared_error: 0.4813 - val_y2_output_root_mean_squared_error: 0.9756
    Epoch 377/500
    614/614 [==============================] - 0s 138us/sample - loss: 0.4618 - y1_output_loss: 0.1226 - y2_output_loss: 0.3371 - y1_output_root_mean_squared_error: 0.3508 - y2_output_root_mean_squared_error: 0.5821 - val_loss: 1.2700 - val_y1_output_loss: 0.2480 - val_y2_output_loss: 1.0054 - val_y1_output_root_mean_squared_error: 0.5017 - val_y2_output_root_mean_squared_error: 1.0091
    Epoch 378/500
    614/614 [==============================] - 0s 138us/sample - loss: 0.3922 - y1_output_loss: 0.1029 - y2_output_loss: 0.2900 - y1_output_root_mean_squared_error: 0.3218 - y2_output_root_mean_squared_error: 0.5372 - val_loss: 1.2826 - val_y1_output_loss: 0.2319 - val_y2_output_loss: 1.0524 - val_y1_output_root_mean_squared_error: 0.4831 - val_y2_output_root_mean_squared_error: 1.0243
    Epoch 379/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.4970 - y1_output_loss: 0.1364 - y2_output_loss: 0.3595 - y1_output_root_mean_squared_error: 0.3690 - y2_output_root_mean_squared_error: 0.6007 - val_loss: 1.1899 - val_y1_output_loss: 0.2396 - val_y2_output_loss: 0.9427 - val_y1_output_root_mean_squared_error: 0.4922 - val_y2_output_root_mean_squared_error: 0.9735
    Epoch 380/500
    614/614 [==============================] - 0s 135us/sample - loss: 0.4154 - y1_output_loss: 0.1000 - y2_output_loss: 0.3196 - y1_output_root_mean_squared_error: 0.3152 - y2_output_root_mean_squared_error: 0.5622 - val_loss: 2.8903 - val_y1_output_loss: 0.4253 - val_y2_output_loss: 2.4680 - val_y1_output_root_mean_squared_error: 0.6474 - val_y2_output_root_mean_squared_error: 1.5720
    Epoch 381/500
    614/614 [==============================] - 0s 135us/sample - loss: 0.4899 - y1_output_loss: 0.1166 - y2_output_loss: 0.3722 - y1_output_root_mean_squared_error: 0.3426 - y2_output_root_mean_squared_error: 0.6103 - val_loss: 2.1717 - val_y1_output_loss: 0.3481 - val_y2_output_loss: 1.8055 - val_y1_output_root_mean_squared_error: 0.5982 - val_y2_output_root_mean_squared_error: 1.3468
    Epoch 382/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.5781 - y1_output_loss: 0.1569 - y2_output_loss: 0.4226 - y1_output_root_mean_squared_error: 0.3954 - y2_output_root_mean_squared_error: 0.6495 - val_loss: 1.2733 - val_y1_output_loss: 0.2409 - val_y2_output_loss: 1.0366 - val_y1_output_root_mean_squared_error: 0.4924 - val_y2_output_root_mean_squared_error: 1.0153
    Epoch 383/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.4011 - y1_output_loss: 0.1018 - y2_output_loss: 0.2987 - y1_output_root_mean_squared_error: 0.3175 - y2_output_root_mean_squared_error: 0.5480 - val_loss: 1.1204 - val_y1_output_loss: 0.2146 - val_y2_output_loss: 0.8993 - val_y1_output_root_mean_squared_error: 0.4675 - val_y2_output_root_mean_squared_error: 0.9497
    Epoch 384/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.4584 - y1_output_loss: 0.1153 - y2_output_loss: 0.3416 - y1_output_root_mean_squared_error: 0.3383 - y2_output_root_mean_squared_error: 0.5865 - val_loss: 1.1464 - val_y1_output_loss: 0.2280 - val_y2_output_loss: 0.9215 - val_y1_output_root_mean_squared_error: 0.4792 - val_y2_output_root_mean_squared_error: 0.9575
    Epoch 385/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.4918 - y1_output_loss: 0.1194 - y2_output_loss: 0.3705 - y1_output_root_mean_squared_error: 0.3461 - y2_output_root_mean_squared_error: 0.6099 - val_loss: 1.1506 - val_y1_output_loss: 0.2157 - val_y2_output_loss: 0.9246 - val_y1_output_root_mean_squared_error: 0.4678 - val_y2_output_root_mean_squared_error: 0.9653
    Epoch 386/500
    614/614 [==============================] - 0s 139us/sample - loss: 0.4287 - y1_output_loss: 0.1128 - y2_output_loss: 0.3132 - y1_output_root_mean_squared_error: 0.3360 - y2_output_root_mean_squared_error: 0.5619 - val_loss: 1.0785 - val_y1_output_loss: 0.2402 - val_y2_output_loss: 0.8307 - val_y1_output_root_mean_squared_error: 0.4944 - val_y2_output_root_mean_squared_error: 0.9133
    Epoch 387/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.4645 - y1_output_loss: 0.1145 - y2_output_loss: 0.3484 - y1_output_root_mean_squared_error: 0.3381 - y2_output_root_mean_squared_error: 0.5918 - val_loss: 1.3764 - val_y1_output_loss: 0.3015 - val_y2_output_loss: 1.0672 - val_y1_output_root_mean_squared_error: 0.5504 - val_y2_output_root_mean_squared_error: 1.0360
    Epoch 388/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.4795 - y1_output_loss: 0.1086 - y2_output_loss: 0.3677 - y1_output_root_mean_squared_error: 0.3307 - y2_output_root_mean_squared_error: 0.6084 - val_loss: 1.1011 - val_y1_output_loss: 0.2403 - val_y2_output_loss: 0.8505 - val_y1_output_root_mean_squared_error: 0.4920 - val_y2_output_root_mean_squared_error: 0.9269
    Epoch 389/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.6228 - y1_output_loss: 0.1496 - y2_output_loss: 0.4695 - y1_output_root_mean_squared_error: 0.3881 - y2_output_root_mean_squared_error: 0.6872 - val_loss: 1.2640 - val_y1_output_loss: 0.2783 - val_y2_output_loss: 0.9638 - val_y1_output_root_mean_squared_error: 0.5343 - val_y2_output_root_mean_squared_error: 0.9892
    Epoch 390/500
    614/614 [==============================] - 0s 141us/sample - loss: 0.3830 - y1_output_loss: 0.1016 - y2_output_loss: 0.2852 - y1_output_root_mean_squared_error: 0.3185 - y2_output_root_mean_squared_error: 0.5306 - val_loss: 1.6136 - val_y1_output_loss: 0.2801 - val_y2_output_loss: 1.3281 - val_y1_output_root_mean_squared_error: 0.5343 - val_y2_output_root_mean_squared_error: 1.1524
    Epoch 391/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.4576 - y1_output_loss: 0.1154 - y2_output_loss: 0.3387 - y1_output_root_mean_squared_error: 0.3406 - y2_output_root_mean_squared_error: 0.5845 - val_loss: 1.1243 - val_y1_output_loss: 0.2248 - val_y2_output_loss: 0.9040 - val_y1_output_root_mean_squared_error: 0.4735 - val_y2_output_root_mean_squared_error: 0.9488
    Epoch 392/500
    614/614 [==============================] - 0s 141us/sample - loss: 0.3936 - y1_output_loss: 0.1040 - y2_output_loss: 0.2890 - y1_output_root_mean_squared_error: 0.3215 - y2_output_root_mean_squared_error: 0.5388 - val_loss: 2.0362 - val_y1_output_loss: 0.2538 - val_y2_output_loss: 1.7959 - val_y1_output_root_mean_squared_error: 0.5062 - val_y2_output_root_mean_squared_error: 1.3342
    Epoch 393/500
    614/614 [==============================] - 0s 133us/sample - loss: 0.6856 - y1_output_loss: 0.1552 - y2_output_loss: 0.5253 - y1_output_root_mean_squared_error: 0.3951 - y2_output_root_mean_squared_error: 0.7276 - val_loss: 1.1006 - val_y1_output_loss: 0.2219 - val_y2_output_loss: 0.8667 - val_y1_output_root_mean_squared_error: 0.4759 - val_y2_output_root_mean_squared_error: 0.9349
    Epoch 394/500
    614/614 [==============================] - 0s 136us/sample - loss: 0.4536 - y1_output_loss: 0.1165 - y2_output_loss: 0.3364 - y1_output_root_mean_squared_error: 0.3419 - y2_output_root_mean_squared_error: 0.5803 - val_loss: 1.0958 - val_y1_output_loss: 0.2303 - val_y2_output_loss: 0.8603 - val_y1_output_root_mean_squared_error: 0.4843 - val_y2_output_root_mean_squared_error: 0.9281
    Epoch 395/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.3851 - y1_output_loss: 0.1084 - y2_output_loss: 0.2762 - y1_output_root_mean_squared_error: 0.3287 - y2_output_root_mean_squared_error: 0.5264 - val_loss: 1.0939 - val_y1_output_loss: 0.2157 - val_y2_output_loss: 0.8727 - val_y1_output_root_mean_squared_error: 0.4665 - val_y2_output_root_mean_squared_error: 0.9361
    Epoch 396/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.4048 - y1_output_loss: 0.1151 - y2_output_loss: 0.2938 - y1_output_root_mean_squared_error: 0.3397 - y2_output_root_mean_squared_error: 0.5379 - val_loss: 1.7815 - val_y1_output_loss: 0.3762 - val_y2_output_loss: 1.3712 - val_y1_output_root_mean_squared_error: 0.6231 - val_y2_output_root_mean_squared_error: 1.1804
    Epoch 397/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.4391 - y1_output_loss: 0.1163 - y2_output_loss: 0.3206 - y1_output_root_mean_squared_error: 0.3424 - y2_output_root_mean_squared_error: 0.5674 - val_loss: 1.0736 - val_y1_output_loss: 0.2385 - val_y2_output_loss: 0.8295 - val_y1_output_root_mean_squared_error: 0.4910 - val_y2_output_root_mean_squared_error: 0.9124
    Epoch 398/500
    614/614 [==============================] - 0s 140us/sample - loss: 0.4609 - y1_output_loss: 0.1156 - y2_output_loss: 0.3428 - y1_output_root_mean_squared_error: 0.3411 - y2_output_root_mean_squared_error: 0.5870 - val_loss: 1.1509 - val_y1_output_loss: 0.2192 - val_y2_output_loss: 0.9161 - val_y1_output_root_mean_squared_error: 0.4734 - val_y2_output_root_mean_squared_error: 0.9627
    Epoch 399/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.3769 - y1_output_loss: 0.1068 - y2_output_loss: 0.2671 - y1_output_root_mean_squared_error: 0.3282 - y2_output_root_mean_squared_error: 0.5188 - val_loss: 1.0588 - val_y1_output_loss: 0.2069 - val_y2_output_loss: 0.8406 - val_y1_output_root_mean_squared_error: 0.4592 - val_y2_output_root_mean_squared_error: 0.9208
    Epoch 400/500
    614/614 [==============================] - 0s 141us/sample - loss: 0.3909 - y1_output_loss: 0.1036 - y2_output_loss: 0.2854 - y1_output_root_mean_squared_error: 0.3227 - y2_output_root_mean_squared_error: 0.5355 - val_loss: 1.3334 - val_y1_output_loss: 0.2920 - val_y2_output_loss: 1.0511 - val_y1_output_root_mean_squared_error: 0.5345 - val_y2_output_root_mean_squared_error: 1.0236
    Epoch 401/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.4302 - y1_output_loss: 0.1203 - y2_output_loss: 0.3070 - y1_output_root_mean_squared_error: 0.3480 - y2_output_root_mean_squared_error: 0.5559 - val_loss: 1.0726 - val_y1_output_loss: 0.2165 - val_y2_output_loss: 0.8544 - val_y1_output_root_mean_squared_error: 0.4669 - val_y2_output_root_mean_squared_error: 0.9245
    Epoch 402/500
    614/614 [==============================] - 0s 139us/sample - loss: 0.3761 - y1_output_loss: 0.1001 - y2_output_loss: 0.2788 - y1_output_root_mean_squared_error: 0.3161 - y2_output_root_mean_squared_error: 0.5255 - val_loss: 1.3729 - val_y1_output_loss: 0.2557 - val_y2_output_loss: 1.0973 - val_y1_output_root_mean_squared_error: 0.5135 - val_y2_output_root_mean_squared_error: 1.0532
    Epoch 403/500
    614/614 [==============================] - 0s 136us/sample - loss: 0.3845 - y1_output_loss: 0.1000 - y2_output_loss: 0.2822 - y1_output_root_mean_squared_error: 0.3163 - y2_output_root_mean_squared_error: 0.5334 - val_loss: 1.1771 - val_y1_output_loss: 0.2175 - val_y2_output_loss: 0.9512 - val_y1_output_root_mean_squared_error: 0.4719 - val_y2_output_root_mean_squared_error: 0.9769
    Epoch 404/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.3762 - y1_output_loss: 0.1049 - y2_output_loss: 0.2751 - y1_output_root_mean_squared_error: 0.3224 - y2_output_root_mean_squared_error: 0.5218 - val_loss: 1.5471 - val_y1_output_loss: 0.3584 - val_y2_output_loss: 1.2033 - val_y1_output_root_mean_squared_error: 0.5949 - val_y2_output_root_mean_squared_error: 1.0923
    Epoch 405/500
    614/614 [==============================] - 0s 141us/sample - loss: 0.4008 - y1_output_loss: 0.1119 - y2_output_loss: 0.2892 - y1_output_root_mean_squared_error: 0.3357 - y2_output_root_mean_squared_error: 0.5367 - val_loss: 1.6745 - val_y1_output_loss: 0.2643 - val_y2_output_loss: 1.4037 - val_y1_output_root_mean_squared_error: 0.5185 - val_y2_output_root_mean_squared_error: 1.1856
    Epoch 406/500
    614/614 [==============================] - 0s 138us/sample - loss: 0.4908 - y1_output_loss: 0.1341 - y2_output_loss: 0.3553 - y1_output_root_mean_squared_error: 0.3677 - y2_output_root_mean_squared_error: 0.5963 - val_loss: 1.1687 - val_y1_output_loss: 0.2224 - val_y2_output_loss: 0.9529 - val_y1_output_root_mean_squared_error: 0.4759 - val_y2_output_root_mean_squared_error: 0.9707
    Epoch 407/500
    614/614 [==============================] - 0s 138us/sample - loss: 0.3865 - y1_output_loss: 0.1032 - y2_output_loss: 0.2818 - y1_output_root_mean_squared_error: 0.3213 - y2_output_root_mean_squared_error: 0.5323 - val_loss: 1.2437 - val_y1_output_loss: 0.2998 - val_y2_output_loss: 0.9494 - val_y1_output_root_mean_squared_error: 0.5459 - val_y2_output_root_mean_squared_error: 0.9724
    Epoch 408/500
    614/614 [==============================] - 0s 139us/sample - loss: 0.4987 - y1_output_loss: 0.1238 - y2_output_loss: 0.3711 - y1_output_root_mean_squared_error: 0.3534 - y2_output_root_mean_squared_error: 0.6114 - val_loss: 1.1281 - val_y1_output_loss: 0.2291 - val_y2_output_loss: 0.8917 - val_y1_output_root_mean_squared_error: 0.4852 - val_y2_output_root_mean_squared_error: 0.9448
    Epoch 409/500
    614/614 [==============================] - 0s 138us/sample - loss: 0.3412 - y1_output_loss: 0.0968 - y2_output_loss: 0.2457 - y1_output_root_mean_squared_error: 0.3111 - y2_output_root_mean_squared_error: 0.4945 - val_loss: 1.1949 - val_y1_output_loss: 0.2172 - val_y2_output_loss: 0.9649 - val_y1_output_root_mean_squared_error: 0.4676 - val_y2_output_root_mean_squared_error: 0.9881
    Epoch 410/500
    614/614 [==============================] - 0s 134us/sample - loss: 0.3858 - y1_output_loss: 0.1030 - y2_output_loss: 0.2809 - y1_output_root_mean_squared_error: 0.3223 - y2_output_root_mean_squared_error: 0.5310 - val_loss: 1.0416 - val_y1_output_loss: 0.2119 - val_y2_output_loss: 0.8260 - val_y1_output_root_mean_squared_error: 0.4617 - val_y2_output_root_mean_squared_error: 0.9102
    Epoch 411/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.3690 - y1_output_loss: 0.1065 - y2_output_loss: 0.2674 - y1_output_root_mean_squared_error: 0.3213 - y2_output_root_mean_squared_error: 0.5155 - val_loss: 2.3209 - val_y1_output_loss: 0.5499 - val_y2_output_loss: 1.7525 - val_y1_output_root_mean_squared_error: 0.7487 - val_y2_output_root_mean_squared_error: 1.3268
    Epoch 412/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.4003 - y1_output_loss: 0.1092 - y2_output_loss: 0.2999 - y1_output_root_mean_squared_error: 0.3316 - y2_output_root_mean_squared_error: 0.5388 - val_loss: 2.0014 - val_y1_output_loss: 0.4248 - val_y2_output_loss: 1.5796 - val_y1_output_root_mean_squared_error: 0.6523 - val_y2_output_root_mean_squared_error: 1.2553
    Epoch 413/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.4899 - y1_output_loss: 0.1238 - y2_output_loss: 0.3634 - y1_output_root_mean_squared_error: 0.3522 - y2_output_root_mean_squared_error: 0.6048 - val_loss: 1.3207 - val_y1_output_loss: 0.2622 - val_y2_output_loss: 1.0913 - val_y1_output_root_mean_squared_error: 0.5065 - val_y2_output_root_mean_squared_error: 1.0316
    Epoch 414/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.3620 - y1_output_loss: 0.0965 - y2_output_loss: 0.2687 - y1_output_root_mean_squared_error: 0.3100 - y2_output_root_mean_squared_error: 0.5157 - val_loss: 1.1089 - val_y1_output_loss: 0.2392 - val_y2_output_loss: 0.8701 - val_y1_output_root_mean_squared_error: 0.4923 - val_y2_output_root_mean_squared_error: 0.9309
    Epoch 415/500
    614/614 [==============================] - 0s 138us/sample - loss: 0.3872 - y1_output_loss: 0.1023 - y2_output_loss: 0.2830 - y1_output_root_mean_squared_error: 0.3201 - y2_output_root_mean_squared_error: 0.5337 - val_loss: 1.0707 - val_y1_output_loss: 0.2129 - val_y2_output_loss: 0.8463 - val_y1_output_root_mean_squared_error: 0.4647 - val_y2_output_root_mean_squared_error: 0.9245
    Epoch 416/500
    614/614 [==============================] - 0s 140us/sample - loss: 0.3135 - y1_output_loss: 0.0912 - y2_output_loss: 0.2216 - y1_output_root_mean_squared_error: 0.3023 - y2_output_root_mean_squared_error: 0.4713 - val_loss: 1.0919 - val_y1_output_loss: 0.2163 - val_y2_output_loss: 0.8724 - val_y1_output_root_mean_squared_error: 0.4684 - val_y2_output_root_mean_squared_error: 0.9341
    Epoch 417/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.5134 - y1_output_loss: 0.1292 - y2_output_loss: 0.3810 - y1_output_root_mean_squared_error: 0.3603 - y2_output_root_mean_squared_error: 0.6193 - val_loss: 1.0079 - val_y1_output_loss: 0.2429 - val_y2_output_loss: 0.7771 - val_y1_output_root_mean_squared_error: 0.4960 - val_y2_output_root_mean_squared_error: 0.8729
    Epoch 418/500
    614/614 [==============================] - 0s 138us/sample - loss: 0.3891 - y1_output_loss: 0.1075 - y2_output_loss: 0.2833 - y1_output_root_mean_squared_error: 0.3251 - y2_output_root_mean_squared_error: 0.5324 - val_loss: 1.6511 - val_y1_output_loss: 0.3956 - val_y2_output_loss: 1.2566 - val_y1_output_root_mean_squared_error: 0.6281 - val_y2_output_root_mean_squared_error: 1.1210
    Epoch 419/500
    614/614 [==============================] - 0s 141us/sample - loss: 0.3715 - y1_output_loss: 0.1107 - y2_output_loss: 0.2594 - y1_output_root_mean_squared_error: 0.3319 - y2_output_root_mean_squared_error: 0.5112 - val_loss: 1.0644 - val_y1_output_loss: 0.2155 - val_y2_output_loss: 0.8401 - val_y1_output_root_mean_squared_error: 0.4674 - val_y2_output_root_mean_squared_error: 0.9198
    Epoch 420/500
    614/614 [==============================] - 0s 133us/sample - loss: 0.3998 - y1_output_loss: 0.1082 - y2_output_loss: 0.2880 - y1_output_root_mean_squared_error: 0.3302 - y2_output_root_mean_squared_error: 0.5392 - val_loss: 0.9742 - val_y1_output_loss: 0.2032 - val_y2_output_loss: 0.7748 - val_y1_output_root_mean_squared_error: 0.4534 - val_y2_output_root_mean_squared_error: 0.8767
    Epoch 421/500
    614/614 [==============================] - 0s 135us/sample - loss: 0.4357 - y1_output_loss: 0.1054 - y2_output_loss: 0.3345 - y1_output_root_mean_squared_error: 0.3247 - y2_output_root_mean_squared_error: 0.5747 - val_loss: 1.9909 - val_y1_output_loss: 0.3282 - val_y2_output_loss: 1.7093 - val_y1_output_root_mean_squared_error: 0.5716 - val_y2_output_root_mean_squared_error: 1.2900
    Epoch 422/500
    614/614 [==============================] - 0s 142us/sample - loss: 0.3626 - y1_output_loss: 0.1010 - y2_output_loss: 0.2619 - y1_output_root_mean_squared_error: 0.3183 - y2_output_root_mean_squared_error: 0.5111 - val_loss: 1.4665 - val_y1_output_loss: 0.2964 - val_y2_output_loss: 1.1915 - val_y1_output_root_mean_squared_error: 0.5431 - val_y2_output_root_mean_squared_error: 1.0824
    Epoch 423/500
    614/614 [==============================] - 0s 138us/sample - loss: 0.4186 - y1_output_loss: 0.1045 - y2_output_loss: 0.3127 - y1_output_root_mean_squared_error: 0.3241 - y2_output_root_mean_squared_error: 0.5600 - val_loss: 1.1880 - val_y1_output_loss: 0.2819 - val_y2_output_loss: 0.8847 - val_y1_output_root_mean_squared_error: 0.5396 - val_y2_output_root_mean_squared_error: 0.9470
    Epoch 424/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.3357 - y1_output_loss: 0.1008 - y2_output_loss: 0.2467 - y1_output_root_mean_squared_error: 0.3087 - y2_output_root_mean_squared_error: 0.4903 - val_loss: 5.2268 - val_y1_output_loss: 1.5125 - val_y2_output_loss: 3.7335 - val_y1_output_root_mean_squared_error: 1.2306 - val_y2_output_root_mean_squared_error: 1.9268
    Epoch 425/500
    614/614 [==============================] - 0s 140us/sample - loss: 0.6675 - y1_output_loss: 0.1880 - y2_output_loss: 0.4741 - y1_output_root_mean_squared_error: 0.4356 - y2_output_root_mean_squared_error: 0.6912 - val_loss: 1.0378 - val_y1_output_loss: 0.2042 - val_y2_output_loss: 0.8250 - val_y1_output_root_mean_squared_error: 0.4577 - val_y2_output_root_mean_squared_error: 0.9101
    Epoch 426/500
    614/614 [==============================] - 0s 138us/sample - loss: 0.3832 - y1_output_loss: 0.1024 - y2_output_loss: 0.2808 - y1_output_root_mean_squared_error: 0.3210 - y2_output_root_mean_squared_error: 0.5293 - val_loss: 1.5997 - val_y1_output_loss: 0.2957 - val_y2_output_loss: 1.2952 - val_y1_output_root_mean_squared_error: 0.5447 - val_y2_output_root_mean_squared_error: 1.1415
    Epoch 427/500
    614/614 [==============================] - 0s 140us/sample - loss: 0.3536 - y1_output_loss: 0.1041 - y2_output_loss: 0.2501 - y1_output_root_mean_squared_error: 0.3220 - y2_output_root_mean_squared_error: 0.4999 - val_loss: 1.1231 - val_y1_output_loss: 0.2607 - val_y2_output_loss: 0.8505 - val_y1_output_root_mean_squared_error: 0.5136 - val_y2_output_root_mean_squared_error: 0.9270
    Epoch 428/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.3365 - y1_output_loss: 0.0958 - y2_output_loss: 0.2394 - y1_output_root_mean_squared_error: 0.3104 - y2_output_root_mean_squared_error: 0.4900 - val_loss: 1.1163 - val_y1_output_loss: 0.2299 - val_y2_output_loss: 0.8839 - val_y1_output_root_mean_squared_error: 0.4811 - val_y2_output_root_mean_squared_error: 0.9406
    Epoch 429/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.3571 - y1_output_loss: 0.1027 - y2_output_loss: 0.2536 - y1_output_root_mean_squared_error: 0.3203 - y2_output_root_mean_squared_error: 0.5045 - val_loss: 1.0267 - val_y1_output_loss: 0.2142 - val_y2_output_loss: 0.8045 - val_y1_output_root_mean_squared_error: 0.4681 - val_y2_output_root_mean_squared_error: 0.8986
    Epoch 430/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.3900 - y1_output_loss: 0.1085 - y2_output_loss: 0.2791 - y1_output_root_mean_squared_error: 0.3300 - y2_output_root_mean_squared_error: 0.5302 - val_loss: 1.0186 - val_y1_output_loss: 0.2210 - val_y2_output_loss: 0.7869 - val_y1_output_root_mean_squared_error: 0.4736 - val_y2_output_root_mean_squared_error: 0.8912
    Epoch 431/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.3912 - y1_output_loss: 0.1109 - y2_output_loss: 0.2807 - y1_output_root_mean_squared_error: 0.3342 - y2_output_root_mean_squared_error: 0.5286 - val_loss: 1.1260 - val_y1_output_loss: 0.2236 - val_y2_output_loss: 0.9118 - val_y1_output_root_mean_squared_error: 0.4706 - val_y2_output_root_mean_squared_error: 0.9510
    Epoch 432/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.5681 - y1_output_loss: 0.1580 - y2_output_loss: 0.4169 - y1_output_root_mean_squared_error: 0.3964 - y2_output_root_mean_squared_error: 0.6410 - val_loss: 3.2776 - val_y1_output_loss: 0.7217 - val_y2_output_loss: 2.5459 - val_y1_output_root_mean_squared_error: 0.8546 - val_y2_output_root_mean_squared_error: 1.5960
    Epoch 433/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.4365 - y1_output_loss: 0.1182 - y2_output_loss: 0.3159 - y1_output_root_mean_squared_error: 0.3436 - y2_output_root_mean_squared_error: 0.5643 - val_loss: 1.2072 - val_y1_output_loss: 0.2426 - val_y2_output_loss: 0.9437 - val_y1_output_root_mean_squared_error: 0.4985 - val_y2_output_root_mean_squared_error: 0.9792
    Epoch 434/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.3938 - y1_output_loss: 0.1143 - y2_output_loss: 0.2779 - y1_output_root_mean_squared_error: 0.3374 - y2_output_root_mean_squared_error: 0.5292 - val_loss: 1.0393 - val_y1_output_loss: 0.2245 - val_y2_output_loss: 0.8128 - val_y1_output_root_mean_squared_error: 0.4757 - val_y2_output_root_mean_squared_error: 0.9017
    Epoch 435/500
    614/614 [==============================] - 0s 136us/sample - loss: 0.3942 - y1_output_loss: 0.0941 - y2_output_loss: 0.2972 - y1_output_root_mean_squared_error: 0.3074 - y2_output_root_mean_squared_error: 0.5474 - val_loss: 0.9771 - val_y1_output_loss: 0.2236 - val_y2_output_loss: 0.7501 - val_y1_output_root_mean_squared_error: 0.4788 - val_y2_output_root_mean_squared_error: 0.8648
    Epoch 436/500
    614/614 [==============================] - 0s 135us/sample - loss: 0.4674 - y1_output_loss: 0.1079 - y2_output_loss: 0.3570 - y1_output_root_mean_squared_error: 0.3290 - y2_output_root_mean_squared_error: 0.5993 - val_loss: 1.2242 - val_y1_output_loss: 0.2168 - val_y2_output_loss: 0.9946 - val_y1_output_root_mean_squared_error: 0.4676 - val_y2_output_root_mean_squared_error: 1.0028
    Epoch 437/500
    614/614 [==============================] - 0s 139us/sample - loss: 0.5225 - y1_output_loss: 0.1278 - y2_output_loss: 0.4022 - y1_output_root_mean_squared_error: 0.3554 - y2_output_root_mean_squared_error: 0.6294 - val_loss: 3.2577 - val_y1_output_loss: 0.6375 - val_y2_output_loss: 2.5888 - val_y1_output_root_mean_squared_error: 0.8070 - val_y2_output_root_mean_squared_error: 1.6144
    Epoch 438/500
    614/614 [==============================] - 0s 140us/sample - loss: 0.6260 - y1_output_loss: 0.1450 - y2_output_loss: 0.4802 - y1_output_root_mean_squared_error: 0.3824 - y2_output_root_mean_squared_error: 0.6927 - val_loss: 1.2051 - val_y1_output_loss: 0.2291 - val_y2_output_loss: 0.9653 - val_y1_output_root_mean_squared_error: 0.4811 - val_y2_output_root_mean_squared_error: 0.9867
    Epoch 439/500
    614/614 [==============================] - 0s 140us/sample - loss: 0.4041 - y1_output_loss: 0.1192 - y2_output_loss: 0.2924 - y1_output_root_mean_squared_error: 0.3446 - y2_output_root_mean_squared_error: 0.5342 - val_loss: 2.7623 - val_y1_output_loss: 0.3443 - val_y2_output_loss: 2.4629 - val_y1_output_root_mean_squared_error: 0.5848 - val_y2_output_root_mean_squared_error: 1.5557
    Epoch 440/500
    614/614 [==============================] - 0s 138us/sample - loss: 0.4493 - y1_output_loss: 0.1208 - y2_output_loss: 0.3251 - y1_output_root_mean_squared_error: 0.3490 - y2_output_root_mean_squared_error: 0.5723 - val_loss: 1.0317 - val_y1_output_loss: 0.2436 - val_y2_output_loss: 0.7913 - val_y1_output_root_mean_squared_error: 0.4986 - val_y2_output_root_mean_squared_error: 0.8849
    Epoch 441/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.3599 - y1_output_loss: 0.0989 - y2_output_loss: 0.2604 - y1_output_root_mean_squared_error: 0.3149 - y2_output_root_mean_squared_error: 0.5107 - val_loss: 0.9967 - val_y1_output_loss: 0.2057 - val_y2_output_loss: 0.7889 - val_y1_output_root_mean_squared_error: 0.4583 - val_y2_output_root_mean_squared_error: 0.8869
    Epoch 442/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.3816 - y1_output_loss: 0.1045 - y2_output_loss: 0.3085 - y1_output_root_mean_squared_error: 0.3242 - y2_output_root_mean_squared_error: 0.5258 - val_loss: 4.8275 - val_y1_output_loss: 0.3437 - val_y2_output_loss: 4.4864 - val_y1_output_root_mean_squared_error: 0.5935 - val_y2_output_root_mean_squared_error: 2.1155
    Epoch 443/500
    614/614 [==============================] - 0s 136us/sample - loss: 0.3845 - y1_output_loss: 0.0930 - y2_output_loss: 0.2898 - y1_output_root_mean_squared_error: 0.3060 - y2_output_root_mean_squared_error: 0.5393 - val_loss: 1.3170 - val_y1_output_loss: 0.2583 - val_y2_output_loss: 1.0465 - val_y1_output_root_mean_squared_error: 0.5120 - val_y2_output_root_mean_squared_error: 1.0270
    Epoch 444/500
    614/614 [==============================] - 0s 127us/sample - loss: 0.3474 - y1_output_loss: 0.1052 - y2_output_loss: 0.2406 - y1_output_root_mean_squared_error: 0.3243 - y2_output_root_mean_squared_error: 0.4922 - val_loss: 0.9297 - val_y1_output_loss: 0.2176 - val_y2_output_loss: 0.7126 - val_y1_output_root_mean_squared_error: 0.4715 - val_y2_output_root_mean_squared_error: 0.8411
    Epoch 445/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.3602 - y1_output_loss: 0.1013 - y2_output_loss: 0.2576 - y1_output_root_mean_squared_error: 0.3195 - y2_output_root_mean_squared_error: 0.5080 - val_loss: 1.2351 - val_y1_output_loss: 0.2164 - val_y2_output_loss: 1.0106 - val_y1_output_root_mean_squared_error: 0.4689 - val_y2_output_root_mean_squared_error: 1.0076
    Epoch 446/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.4020 - y1_output_loss: 0.1202 - y2_output_loss: 0.2969 - y1_output_root_mean_squared_error: 0.3447 - y2_output_root_mean_squared_error: 0.5321 - val_loss: 3.0885 - val_y1_output_loss: 0.4380 - val_y2_output_loss: 2.6193 - val_y1_output_root_mean_squared_error: 0.6724 - val_y2_output_root_mean_squared_error: 1.6237
    Epoch 447/500
    614/614 [==============================] - 0s 135us/sample - loss: 0.4355 - y1_output_loss: 0.1129 - y2_output_loss: 0.3196 - y1_output_root_mean_squared_error: 0.3364 - y2_output_root_mean_squared_error: 0.5677 - val_loss: 0.9747 - val_y1_output_loss: 0.2396 - val_y2_output_loss: 0.7255 - val_y1_output_root_mean_squared_error: 0.4953 - val_y2_output_root_mean_squared_error: 0.8540
    Epoch 448/500
    614/614 [==============================] - 0s 143us/sample - loss: 0.3420 - y1_output_loss: 0.0890 - y2_output_loss: 0.2593 - y1_output_root_mean_squared_error: 0.2995 - y2_output_root_mean_squared_error: 0.5022 - val_loss: 1.8310 - val_y1_output_loss: 0.2238 - val_y2_output_loss: 1.6470 - val_y1_output_root_mean_squared_error: 0.4772 - val_y2_output_root_mean_squared_error: 1.2662
    Epoch 449/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.4433 - y1_output_loss: 0.1075 - y2_output_loss: 0.3330 - y1_output_root_mean_squared_error: 0.3287 - y2_output_root_mean_squared_error: 0.5790 - val_loss: 1.0569 - val_y1_output_loss: 0.2234 - val_y2_output_loss: 0.8251 - val_y1_output_root_mean_squared_error: 0.4733 - val_y2_output_root_mean_squared_error: 0.9126
    Epoch 450/500
    614/614 [==============================] - 0s 135us/sample - loss: 0.4158 - y1_output_loss: 0.1031 - y2_output_loss: 0.3106 - y1_output_root_mean_squared_error: 0.3203 - y2_output_root_mean_squared_error: 0.5597 - val_loss: 1.1413 - val_y1_output_loss: 0.2904 - val_y2_output_loss: 0.8403 - val_y1_output_root_mean_squared_error: 0.5453 - val_y2_output_root_mean_squared_error: 0.9187
    Epoch 451/500
    614/614 [==============================] - 0s 140us/sample - loss: 0.3577 - y1_output_loss: 0.1030 - y2_output_loss: 0.2528 - y1_output_root_mean_squared_error: 0.3208 - y2_output_root_mean_squared_error: 0.5047 - val_loss: 1.0865 - val_y1_output_loss: 0.2207 - val_y2_output_loss: 0.8585 - val_y1_output_root_mean_squared_error: 0.4755 - val_y2_output_root_mean_squared_error: 0.9276
    Epoch 452/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.3938 - y1_output_loss: 0.1019 - y2_output_loss: 0.2918 - y1_output_root_mean_squared_error: 0.3205 - y2_output_root_mean_squared_error: 0.5396 - val_loss: 1.7294 - val_y1_output_loss: 0.2408 - val_y2_output_loss: 1.4705 - val_y1_output_root_mean_squared_error: 0.4984 - val_y2_output_root_mean_squared_error: 1.2169
    Epoch 453/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.3870 - y1_output_loss: 0.0967 - y2_output_loss: 0.2883 - y1_output_root_mean_squared_error: 0.3123 - y2_output_root_mean_squared_error: 0.5380 - val_loss: 1.0821 - val_y1_output_loss: 0.2244 - val_y2_output_loss: 0.8605 - val_y1_output_root_mean_squared_error: 0.4776 - val_y2_output_root_mean_squared_error: 0.9241
    Epoch 454/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.3806 - y1_output_loss: 0.1071 - y2_output_loss: 0.2750 - y1_output_root_mean_squared_error: 0.3271 - y2_output_root_mean_squared_error: 0.5231 - val_loss: 1.5119 - val_y1_output_loss: 0.2920 - val_y2_output_loss: 1.1895 - val_y1_output_root_mean_squared_error: 0.5492 - val_y2_output_root_mean_squared_error: 1.1002
    Epoch 455/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.4494 - y1_output_loss: 0.1140 - y2_output_loss: 0.3339 - y1_output_root_mean_squared_error: 0.3388 - y2_output_root_mean_squared_error: 0.5785 - val_loss: 1.1826 - val_y1_output_loss: 0.2682 - val_y2_output_loss: 0.9369 - val_y1_output_root_mean_squared_error: 0.5158 - val_y2_output_root_mean_squared_error: 0.9574
    Epoch 456/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.4038 - y1_output_loss: 0.0995 - y2_output_loss: 0.3010 - y1_output_root_mean_squared_error: 0.3167 - y2_output_root_mean_squared_error: 0.5509 - val_loss: 1.0021 - val_y1_output_loss: 0.2062 - val_y2_output_loss: 0.7895 - val_y1_output_root_mean_squared_error: 0.4576 - val_y2_output_root_mean_squared_error: 0.8903
    Epoch 457/500
    614/614 [==============================] - 0s 143us/sample - loss: 0.3724 - y1_output_loss: 0.1074 - y2_output_loss: 0.2647 - y1_output_root_mean_squared_error: 0.3278 - y2_output_root_mean_squared_error: 0.5147 - val_loss: 1.4256 - val_y1_output_loss: 0.2945 - val_y2_output_loss: 1.1147 - val_y1_output_root_mean_squared_error: 0.5506 - val_y2_output_root_mean_squared_error: 1.0594
    Epoch 458/500
    614/614 [==============================] - 0s 135us/sample - loss: 0.4185 - y1_output_loss: 0.1008 - y2_output_loss: 0.3146 - y1_output_root_mean_squared_error: 0.3185 - y2_output_root_mean_squared_error: 0.5631 - val_loss: 0.9514 - val_y1_output_loss: 0.2017 - val_y2_output_loss: 0.7487 - val_y1_output_root_mean_squared_error: 0.4514 - val_y2_output_root_mean_squared_error: 0.8647
    Epoch 459/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.4122 - y1_output_loss: 0.1150 - y2_output_loss: 0.2952 - y1_output_root_mean_squared_error: 0.3402 - y2_output_root_mean_squared_error: 0.5444 - val_loss: 0.9310 - val_y1_output_loss: 0.2094 - val_y2_output_loss: 0.7193 - val_y1_output_root_mean_squared_error: 0.4615 - val_y2_output_root_mean_squared_error: 0.8474
    Epoch 460/500
    614/614 [==============================] - 0s 139us/sample - loss: 0.3601 - y1_output_loss: 0.1030 - y2_output_loss: 0.2544 - y1_output_root_mean_squared_error: 0.3218 - y2_output_root_mean_squared_error: 0.5065 - val_loss: 0.9579 - val_y1_output_loss: 0.2025 - val_y2_output_loss: 0.7549 - val_y1_output_root_mean_squared_error: 0.4536 - val_y2_output_root_mean_squared_error: 0.8672
    Epoch 461/500
    614/614 [==============================] - 0s 149us/sample - loss: 0.3996 - y1_output_loss: 0.1234 - y2_output_loss: 0.2795 - y1_output_root_mean_squared_error: 0.3516 - y2_output_root_mean_squared_error: 0.5253 - val_loss: 1.6494 - val_y1_output_loss: 0.2979 - val_y2_output_loss: 1.3509 - val_y1_output_root_mean_squared_error: 0.5530 - val_y2_output_root_mean_squared_error: 1.1592
    Epoch 462/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.3541 - y1_output_loss: 0.1009 - y2_output_loss: 0.2510 - y1_output_root_mean_squared_error: 0.3188 - y2_output_root_mean_squared_error: 0.5024 - val_loss: 0.9593 - val_y1_output_loss: 0.2190 - val_y2_output_loss: 0.7386 - val_y1_output_root_mean_squared_error: 0.4700 - val_y2_output_root_mean_squared_error: 0.8593
    Epoch 463/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.4512 - y1_output_loss: 0.1314 - y2_output_loss: 0.3165 - y1_output_root_mean_squared_error: 0.3637 - y2_output_root_mean_squared_error: 0.5647 - val_loss: 1.0349 - val_y1_output_loss: 0.2541 - val_y2_output_loss: 0.7720 - val_y1_output_root_mean_squared_error: 0.5112 - val_y2_output_root_mean_squared_error: 0.8795
    Epoch 464/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.2920 - y1_output_loss: 0.0917 - y2_output_loss: 0.1986 - y1_output_root_mean_squared_error: 0.3041 - y2_output_root_mean_squared_error: 0.4466 - val_loss: 1.1706 - val_y1_output_loss: 0.2423 - val_y2_output_loss: 0.9175 - val_y1_output_root_mean_squared_error: 0.4998 - val_y2_output_root_mean_squared_error: 0.9595
    Epoch 465/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.3653 - y1_output_loss: 0.1003 - y2_output_loss: 0.2649 - y1_output_root_mean_squared_error: 0.3167 - y2_output_root_mean_squared_error: 0.5148 - val_loss: 0.9385 - val_y1_output_loss: 0.2119 - val_y2_output_loss: 0.7300 - val_y1_output_root_mean_squared_error: 0.4620 - val_y2_output_root_mean_squared_error: 0.8515
    Epoch 466/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.2908 - y1_output_loss: 0.0929 - y2_output_loss: 0.1979 - y1_output_root_mean_squared_error: 0.3051 - y2_output_root_mean_squared_error: 0.4446 - val_loss: 1.2952 - val_y1_output_loss: 0.2422 - val_y2_output_loss: 1.0423 - val_y1_output_root_mean_squared_error: 0.4976 - val_y2_output_root_mean_squared_error: 1.0235
    Epoch 467/500
    614/614 [==============================] - 0s 136us/sample - loss: 0.4902 - y1_output_loss: 0.1254 - y2_output_loss: 0.3634 - y1_output_root_mean_squared_error: 0.3531 - y2_output_root_mean_squared_error: 0.6046 - val_loss: 1.2235 - val_y1_output_loss: 0.2844 - val_y2_output_loss: 0.9322 - val_y1_output_root_mean_squared_error: 0.5386 - val_y2_output_root_mean_squared_error: 0.9662
    Epoch 468/500
    614/614 [==============================] - 0s 140us/sample - loss: 0.3641 - y1_output_loss: 0.0996 - y2_output_loss: 0.2622 - y1_output_root_mean_squared_error: 0.3159 - y2_output_root_mean_squared_error: 0.5141 - val_loss: 1.0004 - val_y1_output_loss: 0.2371 - val_y2_output_loss: 0.7544 - val_y1_output_root_mean_squared_error: 0.4949 - val_y2_output_root_mean_squared_error: 0.8692
    Epoch 469/500
    614/614 [==============================] - 0s 135us/sample - loss: 0.3499 - y1_output_loss: 0.1028 - y2_output_loss: 0.2467 - y1_output_root_mean_squared_error: 0.3204 - y2_output_root_mean_squared_error: 0.4973 - val_loss: 1.1239 - val_y1_output_loss: 0.2494 - val_y2_output_loss: 0.8778 - val_y1_output_root_mean_squared_error: 0.4996 - val_y2_output_root_mean_squared_error: 0.9351
    Epoch 470/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.3584 - y1_output_loss: 0.1000 - y2_output_loss: 0.2557 - y1_output_root_mean_squared_error: 0.3173 - y2_output_root_mean_squared_error: 0.5076 - val_loss: 0.9617 - val_y1_output_loss: 0.2105 - val_y2_output_loss: 0.7533 - val_y1_output_root_mean_squared_error: 0.4617 - val_y2_output_root_mean_squared_error: 0.8652
    Epoch 471/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.3543 - y1_output_loss: 0.0923 - y2_output_loss: 0.2623 - y1_output_root_mean_squared_error: 0.3016 - y2_output_root_mean_squared_error: 0.5132 - val_loss: 1.4158 - val_y1_output_loss: 0.3819 - val_y2_output_loss: 1.0248 - val_y1_output_root_mean_squared_error: 0.6227 - val_y2_output_root_mean_squared_error: 1.0139
    Epoch 472/500
    614/614 [==============================] - 0s 135us/sample - loss: 0.3430 - y1_output_loss: 0.0946 - y2_output_loss: 0.2469 - y1_output_root_mean_squared_error: 0.3088 - y2_output_root_mean_squared_error: 0.4977 - val_loss: 1.3370 - val_y1_output_loss: 0.3011 - val_y2_output_loss: 1.0385 - val_y1_output_root_mean_squared_error: 0.5511 - val_y2_output_root_mean_squared_error: 1.0165
    Epoch 473/500
    614/614 [==============================] - 0s 136us/sample - loss: 0.3656 - y1_output_loss: 0.1059 - y2_output_loss: 0.2575 - y1_output_root_mean_squared_error: 0.3252 - y2_output_root_mean_squared_error: 0.5098 - val_loss: 1.0341 - val_y1_output_loss: 0.2358 - val_y2_output_loss: 0.7918 - val_y1_output_root_mean_squared_error: 0.4910 - val_y2_output_root_mean_squared_error: 0.8905
    Epoch 474/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.3360 - y1_output_loss: 0.0992 - y2_output_loss: 0.2364 - y1_output_root_mean_squared_error: 0.3152 - y2_output_root_mean_squared_error: 0.4865 - val_loss: 1.2056 - val_y1_output_loss: 0.2417 - val_y2_output_loss: 0.9647 - val_y1_output_root_mean_squared_error: 0.4927 - val_y2_output_root_mean_squared_error: 0.9813
    Epoch 475/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.3457 - y1_output_loss: 0.0996 - y2_output_loss: 0.2442 - y1_output_root_mean_squared_error: 0.3167 - y2_output_root_mean_squared_error: 0.4955 - val_loss: 0.9481 - val_y1_output_loss: 0.2126 - val_y2_output_loss: 0.7343 - val_y1_output_root_mean_squared_error: 0.4642 - val_y2_output_root_mean_squared_error: 0.8560
    Epoch 476/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.3179 - y1_output_loss: 0.0913 - y2_output_loss: 0.2341 - y1_output_root_mean_squared_error: 0.3031 - y2_output_root_mean_squared_error: 0.4755 - val_loss: 1.9974 - val_y1_output_loss: 0.3689 - val_y2_output_loss: 1.6555 - val_y1_output_root_mean_squared_error: 0.6056 - val_y2_output_root_mean_squared_error: 1.2770
    Epoch 477/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.4755 - y1_output_loss: 0.1361 - y2_output_loss: 0.3360 - y1_output_root_mean_squared_error: 0.3698 - y2_output_root_mean_squared_error: 0.5820 - val_loss: 0.9239 - val_y1_output_loss: 0.2270 - val_y2_output_loss: 0.7090 - val_y1_output_root_mean_squared_error: 0.4768 - val_y2_output_root_mean_squared_error: 0.8346
    Epoch 478/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.3400 - y1_output_loss: 0.0997 - y2_output_loss: 0.2390 - y1_output_root_mean_squared_error: 0.3157 - y2_output_root_mean_squared_error: 0.4903 - val_loss: 1.0344 - val_y1_output_loss: 0.2704 - val_y2_output_loss: 0.7546 - val_y1_output_root_mean_squared_error: 0.5269 - val_y2_output_root_mean_squared_error: 0.8699
    Epoch 479/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.4045 - y1_output_loss: 0.1163 - y2_output_loss: 0.2890 - y1_output_root_mean_squared_error: 0.3392 - y2_output_root_mean_squared_error: 0.5381 - val_loss: 1.1920 - val_y1_output_loss: 0.3004 - val_y2_output_loss: 0.9016 - val_y1_output_root_mean_squared_error: 0.5423 - val_y2_output_root_mean_squared_error: 0.9476
    Epoch 480/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.3606 - y1_output_loss: 0.0936 - y2_output_loss: 0.2641 - y1_output_root_mean_squared_error: 0.3066 - y2_output_root_mean_squared_error: 0.5163 - val_loss: 0.9553 - val_y1_output_loss: 0.2217 - val_y2_output_loss: 0.7269 - val_y1_output_root_mean_squared_error: 0.4748 - val_y2_output_root_mean_squared_error: 0.8543
    Epoch 481/500
    614/614 [==============================] - 0s 132us/sample - loss: 0.3507 - y1_output_loss: 0.1083 - y2_output_loss: 0.2419 - y1_output_root_mean_squared_error: 0.3281 - y2_output_root_mean_squared_error: 0.4930 - val_loss: 1.0703 - val_y1_output_loss: 0.2515 - val_y2_output_loss: 0.8247 - val_y1_output_root_mean_squared_error: 0.5015 - val_y2_output_root_mean_squared_error: 0.9049
    Epoch 482/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.2908 - y1_output_loss: 0.0944 - y2_output_loss: 0.1977 - y1_output_root_mean_squared_error: 0.3080 - y2_output_root_mean_squared_error: 0.4427 - val_loss: 0.9882 - val_y1_output_loss: 0.2202 - val_y2_output_loss: 0.7715 - val_y1_output_root_mean_squared_error: 0.4711 - val_y2_output_root_mean_squared_error: 0.8754
    Epoch 483/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.3158 - y1_output_loss: 0.1005 - y2_output_loss: 0.2152 - y1_output_root_mean_squared_error: 0.3159 - y2_output_root_mean_squared_error: 0.4648 - val_loss: 0.9978 - val_y1_output_loss: 0.2255 - val_y2_output_loss: 0.7618 - val_y1_output_root_mean_squared_error: 0.4814 - val_y2_output_root_mean_squared_error: 0.8753
    Epoch 484/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.2778 - y1_output_loss: 0.0866 - y2_output_loss: 0.1906 - y1_output_root_mean_squared_error: 0.2944 - y2_output_root_mean_squared_error: 0.4371 - val_loss: 1.0369 - val_y1_output_loss: 0.2093 - val_y2_output_loss: 0.8298 - val_y1_output_root_mean_squared_error: 0.4598 - val_y2_output_root_mean_squared_error: 0.9086
    Epoch 485/500
    614/614 [==============================] - 0s 133us/sample - loss: 0.3496 - y1_output_loss: 0.0894 - y2_output_loss: 0.2614 - y1_output_root_mean_squared_error: 0.2995 - y2_output_root_mean_squared_error: 0.5098 - val_loss: 1.1315 - val_y1_output_loss: 0.2450 - val_y2_output_loss: 0.8911 - val_y1_output_root_mean_squared_error: 0.4983 - val_y2_output_root_mean_squared_error: 0.9398
    Epoch 486/500
    614/614 [==============================] - 0s 130us/sample - loss: 0.3351 - y1_output_loss: 0.1000 - y2_output_loss: 0.2336 - y1_output_root_mean_squared_error: 0.3160 - y2_output_root_mean_squared_error: 0.4850 - val_loss: 1.1084 - val_y1_output_loss: 0.2845 - val_y2_output_loss: 0.8141 - val_y1_output_root_mean_squared_error: 0.5400 - val_y2_output_root_mean_squared_error: 0.9038
    Epoch 487/500
    614/614 [==============================] - 0s 138us/sample - loss: 0.3908 - y1_output_loss: 0.1130 - y2_output_loss: 0.2746 - y1_output_root_mean_squared_error: 0.3371 - y2_output_root_mean_squared_error: 0.5265 - val_loss: 0.9860 - val_y1_output_loss: 0.2142 - val_y2_output_loss: 0.7668 - val_y1_output_root_mean_squared_error: 0.4688 - val_y2_output_root_mean_squared_error: 0.8753
    Epoch 488/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.3073 - y1_output_loss: 0.0919 - y2_output_loss: 0.2160 - y1_output_root_mean_squared_error: 0.3043 - y2_output_root_mean_squared_error: 0.4633 - val_loss: 0.9856 - val_y1_output_loss: 0.2237 - val_y2_output_loss: 0.7530 - val_y1_output_root_mean_squared_error: 0.4780 - val_y2_output_root_mean_squared_error: 0.8701
    Epoch 489/500
    614/614 [==============================] - 0s 137us/sample - loss: 0.3490 - y1_output_loss: 0.1053 - y2_output_loss: 0.2424 - y1_output_root_mean_squared_error: 0.3240 - y2_output_root_mean_squared_error: 0.4940 - val_loss: 0.9669 - val_y1_output_loss: 0.2205 - val_y2_output_loss: 0.7433 - val_y1_output_root_mean_squared_error: 0.4755 - val_y2_output_root_mean_squared_error: 0.8607
    Epoch 490/500
    614/614 [==============================] - 0s 136us/sample - loss: 0.3085 - y1_output_loss: 0.0914 - y2_output_loss: 0.2164 - y1_output_root_mean_squared_error: 0.3019 - y2_output_root_mean_squared_error: 0.4662 - val_loss: 1.0419 - val_y1_output_loss: 0.2267 - val_y2_output_loss: 0.8102 - val_y1_output_root_mean_squared_error: 0.4753 - val_y2_output_root_mean_squared_error: 0.9033
    Epoch 491/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.4418 - y1_output_loss: 0.1073 - y2_output_loss: 0.3318 - y1_output_root_mean_squared_error: 0.3285 - y2_output_root_mean_squared_error: 0.5778 - val_loss: 1.0703 - val_y1_output_loss: 0.2163 - val_y2_output_loss: 0.8500 - val_y1_output_root_mean_squared_error: 0.4691 - val_y2_output_root_mean_squared_error: 0.9221
    Epoch 492/500
    614/614 [==============================] - 0s 139us/sample - loss: 0.3671 - y1_output_loss: 0.1001 - y2_output_loss: 0.2708 - y1_output_root_mean_squared_error: 0.3165 - y2_output_root_mean_squared_error: 0.5167 - val_loss: 1.4523 - val_y1_output_loss: 0.2446 - val_y2_output_loss: 1.2062 - val_y1_output_root_mean_squared_error: 0.4959 - val_y2_output_root_mean_squared_error: 1.0984
    Epoch 493/500
    614/614 [==============================] - 0s 133us/sample - loss: 0.3098 - y1_output_loss: 0.0981 - y2_output_loss: 0.2101 - y1_output_root_mean_squared_error: 0.3133 - y2_output_root_mean_squared_error: 0.4600 - val_loss: 1.3396 - val_y1_output_loss: 0.2018 - val_y2_output_loss: 1.1433 - val_y1_output_root_mean_squared_error: 0.4535 - val_y2_output_root_mean_squared_error: 1.0649
    Epoch 494/500
    614/614 [==============================] - 0s 138us/sample - loss: 0.3793 - y1_output_loss: 0.0899 - y2_output_loss: 0.2960 - y1_output_root_mean_squared_error: 0.3003 - y2_output_root_mean_squared_error: 0.5377 - val_loss: 1.5428 - val_y1_output_loss: 0.3022 - val_y2_output_loss: 1.2344 - val_y1_output_root_mean_squared_error: 0.5511 - val_y2_output_root_mean_squared_error: 1.1131
    Epoch 495/500
    614/614 [==============================] - 0s 131us/sample - loss: 0.3093 - y1_output_loss: 0.0947 - y2_output_loss: 0.2123 - y1_output_root_mean_squared_error: 0.3085 - y2_output_root_mean_squared_error: 0.4628 - val_loss: 1.0124 - val_y1_output_loss: 0.2329 - val_y2_output_loss: 0.7718 - val_y1_output_root_mean_squared_error: 0.4883 - val_y2_output_root_mean_squared_error: 0.8798
    Epoch 496/500
    614/614 [==============================] - 0s 129us/sample - loss: 0.3796 - y1_output_loss: 0.1109 - y2_output_loss: 0.2917 - y1_output_root_mean_squared_error: 0.3251 - y2_output_root_mean_squared_error: 0.5234 - val_loss: 4.1684 - val_y1_output_loss: 0.9355 - val_y2_output_loss: 3.2757 - val_y1_output_root_mean_squared_error: 0.9677 - val_y2_output_root_mean_squared_error: 1.7978
    Epoch 497/500
    614/614 [==============================] - 0s 127us/sample - loss: 0.6473 - y1_output_loss: 0.1728 - y2_output_loss: 0.4762 - y1_output_root_mean_squared_error: 0.4152 - y2_output_root_mean_squared_error: 0.6892 - val_loss: 1.4599 - val_y1_output_loss: 0.2362 - val_y2_output_loss: 1.2027 - val_y1_output_root_mean_squared_error: 0.4949 - val_y2_output_root_mean_squared_error: 1.1023
    Epoch 498/500
    614/614 [==============================] - 0s 127us/sample - loss: 0.3066 - y1_output_loss: 0.0907 - y2_output_loss: 0.2225 - y1_output_root_mean_squared_error: 0.3006 - y2_output_root_mean_squared_error: 0.4650 - val_loss: 1.2814 - val_y1_output_loss: 0.3162 - val_y2_output_loss: 0.9717 - val_y1_output_root_mean_squared_error: 0.5631 - val_y2_output_root_mean_squared_error: 0.9820
    Epoch 499/500
    614/614 [==============================] - 0s 128us/sample - loss: 0.3391 - y1_output_loss: 0.0958 - y2_output_loss: 0.2481 - y1_output_root_mean_squared_error: 0.3081 - y2_output_root_mean_squared_error: 0.4941 - val_loss: 2.6223 - val_y1_output_loss: 0.3518 - val_y2_output_loss: 2.2997 - val_y1_output_root_mean_squared_error: 0.5913 - val_y2_output_root_mean_squared_error: 1.5075
    Epoch 500/500
    614/614 [==============================] - 0s 136us/sample - loss: 0.3151 - y1_output_loss: 0.0937 - y2_output_loss: 0.2216 - y1_output_root_mean_squared_error: 0.3063 - y2_output_root_mean_squared_error: 0.4704 - val_loss: 1.0319 - val_y1_output_loss: 0.2205 - val_y2_output_loss: 0.8139 - val_y1_output_root_mean_squared_error: 0.4708 - val_y2_output_root_mean_squared_error: 0.9001


## Evaluate the Model and Plot Metrics


```python
# Test the model and print loss and mse for both outputs
loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_test_X, y=test_Y)
print("Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(loss, Y1_loss, Y1_rmse, Y2_loss, Y2_rmse))
```

    154/154 [==============================] - 0s 37us/sample - loss: 1.0319 - y1_output_loss: 0.2209 - y2_output_loss: 0.8154 - y1_output_root_mean_squared_error: 0.4708 - y2_output_root_mean_squared_error: 0.9001
    Loss = 1.0318502450918223, Y1_loss = 0.22094647586345673, Y1_mse = 0.47081196308135986, Y2_loss = 0.8154182434082031, Y2_mse = 0.9001035094261169



```python
# Plot the loss and mse
Y_pred = model.predict(norm_test_X)
plot_diff(test_Y[0], Y_pred[0], title='Y1')
plot_diff(test_Y[1], Y_pred[1], title='Y2')
plot_metrics(metric_name='y1_output_root_mean_squared_error', title='Y1 RMSE', ylim=6)
plot_metrics(metric_name='y2_output_root_mean_squared_error', title='Y2 RMSE', ylim=7)
```


![png](output_15_0.png)



![png](output_15_1.png)



![png](output_15_2.png)



![png](output_15_3.png)

