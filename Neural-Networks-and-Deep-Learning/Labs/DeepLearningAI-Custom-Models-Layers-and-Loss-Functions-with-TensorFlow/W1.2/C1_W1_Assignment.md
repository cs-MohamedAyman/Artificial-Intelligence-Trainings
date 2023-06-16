# Week 1: Multiple Output Models using the Keras Functional API

Welcome to the first programming assignment of the course! Your task will be to use the Keras functional API to train a model to predict two outputs. For this lab, you will use the **[Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)** from the **UCI machine learning repository**. It has separate datasets for red wine and white wine.

Normally, the wines are classified into one of the quality ratings specified in the attributes. In this exercise, you will combine the two datasets to predict the wine quality and whether the wine is red or white solely from the attributes.

You will model wine quality estimations as a regression problem and wine type detection as a binary classification problem.

#### Please complete sections that are marked **(TODO)**

## Imports


```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

import utils
```

## Load Dataset


You will now load the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) which are **already saved** in your workspace (*Note: For successful grading, please **do not** modify the default string set to the `URI` variable below*).

### Pre-process the white wine dataset (TODO)
You will add a new column named `is_red` in your dataframe to indicate if the wine is white or red.
- In the white wine dataset, you will fill the column `is_red` with  zeros (0).


```python
## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.

# # URL of the white wine dataset
URI = './winequality-white.csv'

# # load the dataset from the URL
white_df = pd.read_csv(URI, sep=";")

# # fill the `is_red` column with zeros.
white_df["is_red"] = 0

# # keep only the first of duplicate items
white_df = white_df.drop_duplicates(keep='first')
```


```python
# You can click `File -> Open` in the menu above and open the `utils.py` file
# in case you want to inspect the unit tests being used for each graded function.

utils.test_white_df(white_df)

```

    [92m All public tests passed



```python
print(white_df.alcohol[0])
print(white_df.alcohol[100])

# EXPECTED OUTPUT
# 8.8
# 9.1
```

    8.8
    9.1


### Pre-process the red wine dataset (TODO)
- In the red wine dataset, you will fill in the column `is_red` with ones (1).


```python
## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.

# # URL of the red wine dataset
URI = './winequality-red.csv'

# # load the dataset from the URL
red_df = pd.read_csv(URI, sep=";")

# # fill the `is_red` column with ones.
red_df["is_red"] = 1

# # keep only the first of duplicate items
red_df = red_df.drop_duplicates(keep='first')
```


```python
utils.test_red_df(red_df)

```

    [92m All public tests passed



```python
print(red_df.alcohol[0])
print(red_df.alcohol[100])

# EXPECTED OUTPUT
# 9.4
# 10.2
```

    9.4
    10.2


### Concatenate the datasets

Next, concatenate the red and white wine dataframes.


```python
df = pd.concat([red_df, white_df], ignore_index=True)
```


```python
print(df.alcohol[0])
print(df.alcohol[100])

# EXPECTED OUTPUT
# 9.4
# 9.5
```

    9.4
    9.5


In a real-world scenario, you should shuffle the data. For this assignment however, **you are not** going to do that because the grader needs to test with deterministic data. If you want the code to do it **after** you've gotten your grade for this notebook, we left the commented line below for reference


```python
#df = df.iloc[np.random.permutation(len(df))]
```

This will chart the quality of the wines.


```python
df['quality'].hist(bins=20);
```


![png](output_17_0.png)


### Imbalanced data (TODO)
You can see from the plot above that the wine quality dataset is imbalanced.
- Since there are very few observations with quality equal to 3, 4, 8 and 9, you can drop these observations from your dataset.
- You can do this by removing data belonging to all classes except those > 4 and < 8.


```python
## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.

# get data with wine quality greater than 4 and less than 8
df = df[(df['quality'] > 4) & (df['quality'] < 8)]

# reset index and drop the old one
df = df.reset_index(drop=True)
```


```python
utils.test_df_drop(df)


```

    [92m All public tests passed



```python
print(df.alcohol[0])
print(df.alcohol[100])

# EXPECTED OUTPUT
# 9.4
# 10.9
```

    9.4
    10.9


You can plot again to see the new range of data and quality


```python
df['quality'].hist(bins=20);
```


![png](output_23_0.png)


### Train Test Split (TODO)

Next, you can split the datasets into training, test and validation datasets.
- The data frame should be split 80:20 into `train` and `test` sets.
- The resulting `train` should then be split 80:20 into `train` and `val` sets.
- The `train_test_split` parameter `test_size` takes a float value that ranges between 0. and 1, and represents the proportion of the dataset that is allocated to the test set.  The rest of the data is allocated to the training set.


```python
## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.

## Please do not change the random_state parameter. This is needed for grading.

# split df into 80:20 train and test sets
train, test = train_test_split(df, test_size=0.2, random_state = 1)

# split train into 80:20 train and val sets
train, val = train_test_split(train, test_size=0.2, random_state = 1)
```


```python
utils.test_data_sizes(train.size, test.size, val.size)


```

    [92m All public tests passed


Here's where you can explore the training stats. You can pop the labels 'is_red' and 'quality' from the data as these will be used as the labels



```python
train_stats = train.describe()
train_stats.pop('is_red')
train_stats.pop('quality')
train_stats = train_stats.transpose()
```

Explore the training stats!


```python
train_stats
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fixed acidity</th>
      <td>3155.0</td>
      <td>7.221616</td>
      <td>1.325297</td>
      <td>3.80000</td>
      <td>6.40000</td>
      <td>7.00000</td>
      <td>7.7000</td>
      <td>15.60000</td>
    </tr>
    <tr>
      <th>volatile acidity</th>
      <td>3155.0</td>
      <td>0.338929</td>
      <td>0.162476</td>
      <td>0.08000</td>
      <td>0.23000</td>
      <td>0.29000</td>
      <td>0.4000</td>
      <td>1.24000</td>
    </tr>
    <tr>
      <th>citric acid</th>
      <td>3155.0</td>
      <td>0.321569</td>
      <td>0.147970</td>
      <td>0.00000</td>
      <td>0.25000</td>
      <td>0.31000</td>
      <td>0.4000</td>
      <td>1.66000</td>
    </tr>
    <tr>
      <th>residual sugar</th>
      <td>3155.0</td>
      <td>5.155911</td>
      <td>4.639632</td>
      <td>0.60000</td>
      <td>1.80000</td>
      <td>2.80000</td>
      <td>7.6500</td>
      <td>65.80000</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>3155.0</td>
      <td>0.056976</td>
      <td>0.036802</td>
      <td>0.01200</td>
      <td>0.03800</td>
      <td>0.04700</td>
      <td>0.0660</td>
      <td>0.61100</td>
    </tr>
    <tr>
      <th>free sulfur dioxide</th>
      <td>3155.0</td>
      <td>30.388590</td>
      <td>17.236784</td>
      <td>1.00000</td>
      <td>17.00000</td>
      <td>28.00000</td>
      <td>41.0000</td>
      <td>131.00000</td>
    </tr>
    <tr>
      <th>total sulfur dioxide</th>
      <td>3155.0</td>
      <td>115.062282</td>
      <td>56.706617</td>
      <td>6.00000</td>
      <td>75.00000</td>
      <td>117.00000</td>
      <td>156.0000</td>
      <td>344.00000</td>
    </tr>
    <tr>
      <th>density</th>
      <td>3155.0</td>
      <td>0.994633</td>
      <td>0.003005</td>
      <td>0.98711</td>
      <td>0.99232</td>
      <td>0.99481</td>
      <td>0.9968</td>
      <td>1.03898</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>3155.0</td>
      <td>3.223201</td>
      <td>0.161272</td>
      <td>2.72000</td>
      <td>3.11000</td>
      <td>3.21000</td>
      <td>3.3300</td>
      <td>4.01000</td>
    </tr>
    <tr>
      <th>sulphates</th>
      <td>3155.0</td>
      <td>0.534051</td>
      <td>0.149149</td>
      <td>0.22000</td>
      <td>0.43000</td>
      <td>0.51000</td>
      <td>0.6000</td>
      <td>1.95000</td>
    </tr>
    <tr>
      <th>alcohol</th>
      <td>3155.0</td>
      <td>10.504466</td>
      <td>1.154654</td>
      <td>8.50000</td>
      <td>9.50000</td>
      <td>10.30000</td>
      <td>11.3000</td>
      <td>14.00000</td>
    </tr>
  </tbody>
</table>
</div>



### Get the labels (TODO)

The features and labels are currently in the same dataframe.
- You will want to store the label columns `is_red` and `quality` separately from the feature columns.
- The following function, `format_output`, gets these two columns from the dataframe (it's given to you).
- `format_output` also formats the data into numpy arrays.
- Please use the `format_output` and apply it to the `train`, `val` and `test` sets to get dataframes for the labels.


```python
def format_output(data):
    is_red = data.pop('is_red')
    is_red = np.array(is_red)
    quality = data.pop('quality')
    quality = np.array(quality)
    return (quality, is_red)
```


```python
## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.

# format the output of the train set
train_Y = format_output(train)

# format the output of the val set
val_Y = format_output(val)

# format the output of the test set
test_Y = format_output(test)
```


```python
utils.test_format_output(df, train_Y, val_Y, test_Y)
```

    [92m All public tests passed


Notice that after you get the labels, the `train`, `val` and `test` dataframes no longer contain the label columns, and contain just the feature columns.
- This is because you used `.pop` in the `format_output` function.


```python
train.head()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>225</th>
      <td>7.5</td>
      <td>0.65</td>
      <td>0.18</td>
      <td>7.0</td>
      <td>0.088</td>
      <td>27.0</td>
      <td>94.0</td>
      <td>0.99915</td>
      <td>3.38</td>
      <td>0.77</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>3557</th>
      <td>6.3</td>
      <td>0.27</td>
      <td>0.29</td>
      <td>12.2</td>
      <td>0.044</td>
      <td>59.0</td>
      <td>196.0</td>
      <td>0.99782</td>
      <td>3.14</td>
      <td>0.40</td>
      <td>8.8</td>
    </tr>
    <tr>
      <th>3825</th>
      <td>8.8</td>
      <td>0.27</td>
      <td>0.25</td>
      <td>5.0</td>
      <td>0.024</td>
      <td>52.0</td>
      <td>99.0</td>
      <td>0.99250</td>
      <td>2.87</td>
      <td>0.49</td>
      <td>11.4</td>
    </tr>
    <tr>
      <th>1740</th>
      <td>6.4</td>
      <td>0.45</td>
      <td>0.07</td>
      <td>1.1</td>
      <td>0.030</td>
      <td>10.0</td>
      <td>131.0</td>
      <td>0.99050</td>
      <td>2.97</td>
      <td>0.28</td>
      <td>10.8</td>
    </tr>
    <tr>
      <th>1221</th>
      <td>7.2</td>
      <td>0.53</td>
      <td>0.13</td>
      <td>2.0</td>
      <td>0.058</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>0.99573</td>
      <td>3.21</td>
      <td>0.68</td>
      <td>9.9</td>
    </tr>
  </tbody>
</table>
</div>



### Normalize the data (TODO)

Next, you can normalize the data, x, using the formula:
$$x_{norm} = \frac{x - \mu}{\sigma}$$
- The `norm` function is defined for you.
- Please apply the `norm` function to normalize the dataframes that contains the feature columns of `train`, `val` and `test` sets.


```python
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
```


```python
## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.

# normalize the train set
norm_train_X = norm(train)

# normalize the val set
norm_val_X = norm(val)

# normalize the test set
norm_test_X = norm(test)
```


```python
utils.test_norm(norm_train_X, norm_val_X, norm_test_X, train, val, test)

```

    [92m All public tests passed


## Define the Model (TODO)

Define the model using the functional API. The base model will be 2 `Dense` layers of 128 neurons each, and have the `'relu'` activation.
- Check out the documentation for [tf.keras.layers.Dense](https://tensorflow.org/api_docs/python/tf/keras/layers/Dense)


```python
## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.

def base_model(inputs):

    # connect a Dense layer with 128 neurons and a relu activation
    x = tf.keras.layers.Dense(128, activation='relu', name='base_dense_1')(inputs)

    # connect another Dense layer with 128 neurons and a relu activation
    x = tf.keras.layers.Dense(128, activation='relu', name='base_dense_2')(x)
    return x

```


```python
utils.test_base_model(base_model)
```

    [92m All public tests passed


# Define output layers of the model (TODO)

You will add output layers to the base model.
- The model will need two outputs.

One output layer will predict wine quality, which is a numeric value.
- Define a `Dense` layer with 1 neuron.
- Since this is a regression output, the activation can be left as its default value `None`.

The other output layer will predict the wine type, which is either red `1` or not red `0` (white).
- Define a `Dense` layer with 1 neuron.
- Since there are two possible categories, you can use a sigmoid activation for binary classification.

Define the `Model`
- Define the `Model` object, and set the following parameters:
  - `inputs`: pass in the inputs to the model as a list.
  - `outputs`: pass in a list of the outputs that you just defined: wine quality, then wine type.
  - **Note**: please list the wine quality before wine type in the outputs, as this will affect the calculated loss if you choose the other order.


```python
## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.

def final_model(inputs):

    # get the base model
    x = base_model(inputs)

    # connect the output Dense layer for regression
    wine_quality = Dense(units='1', name='wine_quality')(x)

    # connect the output Dense layer for classification. this will use a sigmoid activation.
    wine_type = Dense(units='1', activation='sigmoid', name='wine_type')(x)

    # define the model using the input and output layers
    model = Model(inputs=inputs, outputs=[wine_quality, wine_type])

    return model
```


```python
utils.test_final_model(final_model)
```

    [92m All public tests passed


## Compiling the Model

Next, compile the model. When setting the loss parameter of `model.compile`, you're setting the loss for each of the two outputs (wine quality and wine type).

To set more than one loss, use a dictionary of key-value pairs.
- You can look at the docs for the losses [here](https://tensorflow.org/api_docs/python/tf/keras/losses#functions).
    - **Note**: For the desired spelling, please look at the "Functions" section of the documentation and not the "classes" section on that same page.
- wine_type: Since you will be performing binary classification on wine type, you should use the binary crossentropy loss function for it.  Please pass this in as a string.
  - **Hint**, this should be all lowercase.  In the documentation, you'll see this under the "Functions" section, not the "Classes" section.
- wine_quality: since this is a regression output, use the mean squared error.  Please pass it in as a string, all lowercase.
  - **Hint**: You may notice that there are two aliases for mean squared error.  Please use the shorter name.


You will also set the metric for each of the two outputs.  Again, to set metrics for two or more outputs, use a dictionary with key value pairs.
- The metrics documentation is linked [here](https://tensorflow.org/api_docs/python/tf/keras/metrics).
- For the wine type, please set it to accuracy as a string, all lowercase.
- For wine quality, please use the root mean squared error.  Instead of a string, you'll set it to an instance of the class [RootMeanSquaredError](https://tensorflow.org/api_docs/python/tf/keras/metrics/RootMeanSquaredError), which belongs to the tf.keras.metrics module.

**Note**: If you see the error message
>Exception: wine quality loss function is incorrect.

- Please also check your other losses and metrics, as the error may be caused by the other three key-value pairs and not the wine quality loss.


```python
## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.

inputs = tf.keras.layers.Input(shape=(11,))
rms = tf.keras.optimizers.RMSprop(lr=0.0001)
model = final_model(inputs)

model.compile(optimizer=rms,
              loss = {'wine_type' : 'binary_crossentropy',
                      'wine_quality' : 'mean_squared_error'
                     },
              metrics = {'wine_type' : 'accuracy',
                         'wine_quality': tf.keras.metrics.RootMeanSquaredError()
                       }
             )
```


```python
utils.test_model_compile(model)
```

    [92m All public tests passed


## Training the Model (TODO)

Fit the model to the training inputs and outputs.
- Check the documentation for [model.fit](https://tensorflow.org/api_docs/python/tf/keras/Model#fit).
- Remember to use the normalized training set as inputs.
- For the validation data, please use the normalized validation set.

**Important: Please do not increase the number of epochs below. This is to avoid the grader from timing out. You can increase it once you have submitted your work.**


```python
## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.

history = model.fit(norm_train_X, train_Y,
                    epochs = 40, validation_data=(norm_val_X, val_Y))
```

    Train on 3155 samples, validate on 789 samples
    Epoch 1/40
    3155/3155 [==============================] - 1s 321us/sample - loss: 23.6090 - wine_quality_loss: 22.8271 - wine_type_loss: 0.7472 - wine_quality_root_mean_squared_error: 4.7814 - wine_type_accuracy: 0.3753 - val_loss: 16.2353 - val_wine_quality_loss: 15.5350 - val_wine_type_loss: 0.7099 - val_wine_quality_root_mean_squared_error: 3.9402 - val_wine_type_accuracy: 0.5501
    Epoch 2/40
    3155/3155 [==============================] - 0s 99us/sample - loss: 10.5400 - wine_quality_loss: 9.8449 - wine_type_loss: 0.6784 - wine_quality_root_mean_squared_error: 3.1403 - wine_type_accuracy: 0.6675 - val_loss: 5.9443 - val_wine_quality_loss: 5.3386 - val_wine_type_loss: 0.6434 - val_wine_quality_root_mean_squared_error: 2.3023 - val_wine_type_accuracy: 0.7313
    Epoch 3/40
    3155/3155 [==============================] - 0s 113us/sample - loss: 4.0737 - wine_quality_loss: 3.4717 - wine_type_loss: 0.5934 - wine_quality_root_mean_squared_error: 1.8654 - wine_type_accuracy: 0.7426 - val_loss: 2.8335 - val_wine_quality_loss: 2.3303 - val_wine_type_loss: 0.5436 - val_wine_quality_root_mean_squared_error: 1.5131 - val_wine_type_accuracy: 0.7351
    Epoch 4/40
    3155/3155 [==============================] - 0s 96us/sample - loss: 2.7428 - wine_quality_loss: 2.2565 - wine_type_loss: 0.4880 - wine_quality_root_mean_squared_error: 1.5014 - wine_type_accuracy: 0.7585 - val_loss: 2.3289 - val_wine_quality_loss: 1.9065 - val_wine_type_loss: 0.4459 - val_wine_quality_root_mean_squared_error: 1.3721 - val_wine_type_accuracy: 0.7719
    Epoch 5/40
    3155/3155 [==============================] - 0s 96us/sample - loss: 2.3118 - wine_quality_loss: 1.9097 - wine_type_loss: 0.3985 - wine_quality_root_mean_squared_error: 1.3830 - wine_type_accuracy: 0.8073 - val_loss: 2.0181 - val_wine_quality_loss: 1.6675 - val_wine_type_loss: 0.3641 - val_wine_quality_root_mean_squared_error: 1.2860 - val_wine_type_accuracy: 0.8479
    Epoch 6/40
    3155/3155 [==============================] - 0s 95us/sample - loss: 2.0315 - wine_quality_loss: 1.7048 - wine_type_loss: 0.3271 - wine_quality_root_mean_squared_error: 1.3055 - wine_type_accuracy: 0.8891 - val_loss: 1.7942 - val_wine_quality_loss: 1.5063 - val_wine_type_loss: 0.2963 - val_wine_quality_root_mean_squared_error: 1.2238 - val_wine_type_accuracy: 0.9265
    Epoch 7/40
    3155/3155 [==============================] - 0s 92us/sample - loss: 1.8045 - wine_quality_loss: 1.5384 - wine_type_loss: 0.2642 - wine_quality_root_mean_squared_error: 1.2410 - wine_type_accuracy: 0.9553 - val_loss: 1.6022 - val_wine_quality_loss: 1.3665 - val_wine_type_loss: 0.2407 - val_wine_quality_root_mean_squared_error: 1.1668 - val_wine_type_accuracy: 0.9620
    Epoch 8/40
    3155/3155 [==============================] - 0s 78us/sample - loss: 1.6337 - wine_quality_loss: 1.4182 - wine_type_loss: 0.2150 - wine_quality_root_mean_squared_error: 1.1909 - wine_type_accuracy: 0.9743 - val_loss: 1.4579 - val_wine_quality_loss: 1.2660 - val_wine_type_loss: 0.1941 - val_wine_quality_root_mean_squared_error: 1.1241 - val_wine_type_accuracy: 0.9785
    Epoch 9/40
    3155/3155 [==============================] - 0s 93us/sample - loss: 1.4747 - wine_quality_loss: 1.3019 - wine_type_loss: 0.1742 - wine_quality_root_mean_squared_error: 1.1403 - wine_type_accuracy: 0.9800 - val_loss: 1.3270 - val_wine_quality_loss: 1.1704 - val_wine_type_loss: 0.1582 - val_wine_quality_root_mean_squared_error: 1.0810 - val_wine_type_accuracy: 0.9835
    Epoch 10/40
    3155/3155 [==============================] - 0s 94us/sample - loss: 1.3533 - wine_quality_loss: 1.2077 - wine_type_loss: 0.1429 - wine_quality_root_mean_squared_error: 1.1001 - wine_type_accuracy: 0.9851 - val_loss: 1.2381 - val_wine_quality_loss: 1.1089 - val_wine_type_loss: 0.1291 - val_wine_quality_root_mean_squared_error: 1.0530 - val_wine_type_accuracy: 0.9835
    Epoch 11/40
    3155/3155 [==============================] - 0s 95us/sample - loss: 1.2455 - wine_quality_loss: 1.1275 - wine_type_loss: 0.1187 - wine_quality_root_mean_squared_error: 1.0615 - wine_type_accuracy: 0.9870 - val_loss: 1.1410 - val_wine_quality_loss: 1.0322 - val_wine_type_loss: 0.1086 - val_wine_quality_root_mean_squared_error: 1.0160 - val_wine_type_accuracy: 0.9848
    Epoch 12/40
    3155/3155 [==============================] - 0s 93us/sample - loss: 1.1541 - wine_quality_loss: 1.0536 - wine_type_loss: 0.1011 - wine_quality_root_mean_squared_error: 1.0261 - wine_type_accuracy: 0.9880 - val_loss: 1.0603 - val_wine_quality_loss: 0.9663 - val_wine_type_loss: 0.0934 - val_wine_quality_root_mean_squared_error: 0.9832 - val_wine_type_accuracy: 0.9873
    Epoch 13/40
    3155/3155 [==============================] - 0s 94us/sample - loss: 1.0735 - wine_quality_loss: 0.9868 - wine_type_loss: 0.0878 - wine_quality_root_mean_squared_error: 0.9928 - wine_type_accuracy: 0.9883 - val_loss: 0.9839 - val_wine_quality_loss: 0.9014 - val_wine_type_loss: 0.0816 - val_wine_quality_root_mean_squared_error: 0.9498 - val_wine_type_accuracy: 0.9873
    Epoch 14/40
    3155/3155 [==============================] - 0s 92us/sample - loss: 0.9977 - wine_quality_loss: 0.9194 - wine_type_loss: 0.0789 - wine_quality_root_mean_squared_error: 0.9593 - wine_type_accuracy: 0.9895 - val_loss: 0.9148 - val_wine_quality_loss: 0.8413 - val_wine_type_loss: 0.0725 - val_wine_quality_root_mean_squared_error: 0.9176 - val_wine_type_accuracy: 0.9861
    Epoch 15/40
    3155/3155 [==============================] - 0s 92us/sample - loss: 0.9389 - wine_quality_loss: 0.8687 - wine_type_loss: 0.0697 - wine_quality_root_mean_squared_error: 0.9323 - wine_type_accuracy: 0.9899 - val_loss: 0.8611 - val_wine_quality_loss: 0.7948 - val_wine_type_loss: 0.0658 - val_wine_quality_root_mean_squared_error: 0.8917 - val_wine_type_accuracy: 0.9873
    Epoch 16/40
    3155/3155 [==============================] - 0s 91us/sample - loss: 0.8804 - wine_quality_loss: 0.8171 - wine_type_loss: 0.0636 - wine_quality_root_mean_squared_error: 0.9038 - wine_type_accuracy: 0.9902 - val_loss: 0.8126 - val_wine_quality_loss: 0.7517 - val_wine_type_loss: 0.0599 - val_wine_quality_root_mean_squared_error: 0.8675 - val_wine_type_accuracy: 0.9873
    Epoch 17/40
    3155/3155 [==============================] - 0s 78us/sample - loss: 0.8336 - wine_quality_loss: 0.7768 - wine_type_loss: 0.0583 - wine_quality_root_mean_squared_error: 0.8804 - wine_type_accuracy: 0.9899 - val_loss: 0.7738 - val_wine_quality_loss: 0.7163 - val_wine_type_loss: 0.0559 - val_wine_quality_root_mean_squared_error: 0.8471 - val_wine_type_accuracy: 0.9886
    Epoch 18/40
    3155/3155 [==============================] - 0s 92us/sample - loss: 0.7854 - wine_quality_loss: 0.7310 - wine_type_loss: 0.0543 - wine_quality_root_mean_squared_error: 0.8550 - wine_type_accuracy: 0.9905 - val_loss: 0.7329 - val_wine_quality_loss: 0.6789 - val_wine_type_loss: 0.0524 - val_wine_quality_root_mean_squared_error: 0.8248 - val_wine_type_accuracy: 0.9886
    Epoch 19/40
    3155/3155 [==============================] - 0s 91us/sample - loss: 0.7416 - wine_quality_loss: 0.6888 - wine_type_loss: 0.0511 - wine_quality_root_mean_squared_error: 0.8310 - wine_type_accuracy: 0.9905 - val_loss: 0.7115 - val_wine_quality_loss: 0.6597 - val_wine_type_loss: 0.0496 - val_wine_quality_root_mean_squared_error: 0.8134 - val_wine_type_accuracy: 0.9886
    Epoch 20/40
    3155/3155 [==============================] - 0s 92us/sample - loss: 0.7053 - wine_quality_loss: 0.6586 - wine_type_loss: 0.0483 - wine_quality_root_mean_squared_error: 0.8105 - wine_type_accuracy: 0.9905 - val_loss: 0.6613 - val_wine_quality_loss: 0.6123 - val_wine_type_loss: 0.0472 - val_wine_quality_root_mean_squared_error: 0.7835 - val_wine_type_accuracy: 0.9911
    Epoch 21/40
    3155/3155 [==============================] - 0s 92us/sample - loss: 0.6704 - wine_quality_loss: 0.6257 - wine_type_loss: 0.0459 - wine_quality_root_mean_squared_error: 0.7902 - wine_type_accuracy: 0.9905 - val_loss: 0.6345 - val_wine_quality_loss: 0.5874 - val_wine_type_loss: 0.0452 - val_wine_quality_root_mean_squared_error: 0.7675 - val_wine_type_accuracy: 0.9924
    Epoch 22/40
    3155/3155 [==============================] - 0s 75us/sample - loss: 0.6399 - wine_quality_loss: 0.5975 - wine_type_loss: 0.0439 - wine_quality_root_mean_squared_error: 0.7719 - wine_type_accuracy: 0.9908 - val_loss: 0.5993 - val_wine_quality_loss: 0.5542 - val_wine_type_loss: 0.0437 - val_wine_quality_root_mean_squared_error: 0.7452 - val_wine_type_accuracy: 0.9924
    Epoch 23/40
    3155/3155 [==============================] - 0s 92us/sample - loss: 0.6115 - wine_quality_loss: 0.5689 - wine_type_loss: 0.0422 - wine_quality_root_mean_squared_error: 0.7544 - wine_type_accuracy: 0.9908 - val_loss: 0.5807 - val_wine_quality_loss: 0.5365 - val_wine_type_loss: 0.0423 - val_wine_quality_root_mean_squared_error: 0.7335 - val_wine_type_accuracy: 0.9924
    Epoch 24/40
    3155/3155 [==============================] - 0s 93us/sample - loss: 0.5850 - wine_quality_loss: 0.5433 - wine_type_loss: 0.0406 - wine_quality_root_mean_squared_error: 0.7377 - wine_type_accuracy: 0.9911 - val_loss: 0.5517 - val_wine_quality_loss: 0.5089 - val_wine_type_loss: 0.0413 - val_wine_quality_root_mean_squared_error: 0.7142 - val_wine_type_accuracy: 0.9924
    Epoch 25/40
    3155/3155 [==============================] - 0s 92us/sample - loss: 0.5609 - wine_quality_loss: 0.5222 - wine_type_loss: 0.0395 - wine_quality_root_mean_squared_error: 0.7221 - wine_type_accuracy: 0.9918 - val_loss: 0.5306 - val_wine_quality_loss: 0.4889 - val_wine_type_loss: 0.0401 - val_wine_quality_root_mean_squared_error: 0.7002 - val_wine_type_accuracy: 0.9924
    Epoch 26/40
    3155/3155 [==============================] - 0s 89us/sample - loss: 0.5384 - wine_quality_loss: 0.5003 - wine_type_loss: 0.0384 - wine_quality_root_mean_squared_error: 0.7072 - wine_type_accuracy: 0.9921 - val_loss: 0.5129 - val_wine_quality_loss: 0.4721 - val_wine_type_loss: 0.0392 - val_wine_quality_root_mean_squared_error: 0.6880 - val_wine_type_accuracy: 0.9924
    Epoch 27/40
    3155/3155 [==============================] - 0s 76us/sample - loss: 0.5158 - wine_quality_loss: 0.4786 - wine_type_loss: 0.0371 - wine_quality_root_mean_squared_error: 0.6918 - wine_type_accuracy: 0.9921 - val_loss: 0.4965 - val_wine_quality_loss: 0.4564 - val_wine_type_loss: 0.0383 - val_wine_quality_root_mean_squared_error: 0.6766 - val_wine_type_accuracy: 0.9924
    Epoch 28/40
    3155/3155 [==============================] - 0s 94us/sample - loss: 0.4976 - wine_quality_loss: 0.4609 - wine_type_loss: 0.0362 - wine_quality_root_mean_squared_error: 0.6792 - wine_type_accuracy: 0.9927 - val_loss: 0.4806 - val_wine_quality_loss: 0.4417 - val_wine_type_loss: 0.0374 - val_wine_quality_root_mean_squared_error: 0.6654 - val_wine_type_accuracy: 0.9924
    Epoch 29/40
    3155/3155 [==============================] - 0s 95us/sample - loss: 0.4815 - wine_quality_loss: 0.4457 - wine_type_loss: 0.0352 - wine_quality_root_mean_squared_error: 0.6680 - wine_type_accuracy: 0.9924 - val_loss: 0.4665 - val_wine_quality_loss: 0.4282 - val_wine_type_loss: 0.0368 - val_wine_quality_root_mean_squared_error: 0.6553 - val_wine_type_accuracy: 0.9924
    Epoch 30/40
    3155/3155 [==============================] - 0s 95us/sample - loss: 0.4659 - wine_quality_loss: 0.4323 - wine_type_loss: 0.0346 - wine_quality_root_mean_squared_error: 0.6567 - wine_type_accuracy: 0.9924 - val_loss: 0.4495 - val_wine_quality_loss: 0.4114 - val_wine_type_loss: 0.0362 - val_wine_quality_root_mean_squared_error: 0.6427 - val_wine_type_accuracy: 0.9924
    Epoch 31/40
    3155/3155 [==============================] - 0s 94us/sample - loss: 0.4508 - wine_quality_loss: 0.4172 - wine_type_loss: 0.0340 - wine_quality_root_mean_squared_error: 0.6456 - wine_type_accuracy: 0.9924 - val_loss: 0.4445 - val_wine_quality_loss: 0.4068 - val_wine_type_loss: 0.0357 - val_wine_quality_root_mean_squared_error: 0.6391 - val_wine_type_accuracy: 0.9924
    Epoch 32/40
    3155/3155 [==============================] - 0s 95us/sample - loss: 0.4383 - wine_quality_loss: 0.4062 - wine_type_loss: 0.0331 - wine_quality_root_mean_squared_error: 0.6364 - wine_type_accuracy: 0.9930 - val_loss: 0.4274 - val_wine_quality_loss: 0.3901 - val_wine_type_loss: 0.0354 - val_wine_quality_root_mean_squared_error: 0.6258 - val_wine_type_accuracy: 0.9937
    Epoch 33/40
    3155/3155 [==============================] - 0s 94us/sample - loss: 0.4278 - wine_quality_loss: 0.3952 - wine_type_loss: 0.0327 - wine_quality_root_mean_squared_error: 0.6285 - wine_type_accuracy: 0.9924 - val_loss: 0.4178 - val_wine_quality_loss: 0.3812 - val_wine_type_loss: 0.0349 - val_wine_quality_root_mean_squared_error: 0.6185 - val_wine_type_accuracy: 0.9924
    Epoch 34/40
    3155/3155 [==============================] - 0s 94us/sample - loss: 0.4173 - wine_quality_loss: 0.3849 - wine_type_loss: 0.0323 - wine_quality_root_mean_squared_error: 0.6205 - wine_type_accuracy: 0.9933 - val_loss: 0.4080 - val_wine_quality_loss: 0.3719 - val_wine_type_loss: 0.0345 - val_wine_quality_root_mean_squared_error: 0.6108 - val_wine_type_accuracy: 0.9949
    Epoch 35/40
    3155/3155 [==============================] - 0s 94us/sample - loss: 0.4085 - wine_quality_loss: 0.3774 - wine_type_loss: 0.0317 - wine_quality_root_mean_squared_error: 0.6138 - wine_type_accuracy: 0.9933 - val_loss: 0.3974 - val_wine_quality_loss: 0.3618 - val_wine_type_loss: 0.0340 - val_wine_quality_root_mean_squared_error: 0.6025 - val_wine_type_accuracy: 0.9949
    Epoch 36/40
    3155/3155 [==============================] - 0s 94us/sample - loss: 0.4014 - wine_quality_loss: 0.3704 - wine_type_loss: 0.0313 - wine_quality_root_mean_squared_error: 0.6083 - wine_type_accuracy: 0.9937 - val_loss: 0.3941 - val_wine_quality_loss: 0.3585 - val_wine_type_loss: 0.0337 - val_wine_quality_root_mean_squared_error: 0.6000 - val_wine_type_accuracy: 0.9949
    Epoch 37/40
    3155/3155 [==============================] - 0s 95us/sample - loss: 0.3932 - wine_quality_loss: 0.3621 - wine_type_loss: 0.0308 - wine_quality_root_mean_squared_error: 0.6019 - wine_type_accuracy: 0.9937 - val_loss: 0.3934 - val_wine_quality_loss: 0.3578 - val_wine_type_loss: 0.0335 - val_wine_quality_root_mean_squared_error: 0.5996 - val_wine_type_accuracy: 0.9949
    Epoch 38/40
    3155/3155 [==============================] - 0s 94us/sample - loss: 0.3863 - wine_quality_loss: 0.3556 - wine_type_loss: 0.0304 - wine_quality_root_mean_squared_error: 0.5965 - wine_type_accuracy: 0.9940 - val_loss: 0.3818 - val_wine_quality_loss: 0.3468 - val_wine_type_loss: 0.0333 - val_wine_quality_root_mean_squared_error: 0.5901 - val_wine_type_accuracy: 0.9949
    Epoch 39/40
    3155/3155 [==============================] - 0s 92us/sample - loss: 0.3807 - wine_quality_loss: 0.3501 - wine_type_loss: 0.0300 - wine_quality_root_mean_squared_error: 0.5921 - wine_type_accuracy: 0.9940 - val_loss: 0.3823 - val_wine_quality_loss: 0.3476 - val_wine_type_loss: 0.0331 - val_wine_quality_root_mean_squared_error: 0.5906 - val_wine_type_accuracy: 0.9949
    Epoch 40/40
    3155/3155 [==============================] - 0s 91us/sample - loss: 0.3755 - wine_quality_loss: 0.3456 - wine_type_loss: 0.0298 - wine_quality_root_mean_squared_error: 0.5879 - wine_type_accuracy: 0.9943 - val_loss: 0.3775 - val_wine_quality_loss: 0.3431 - val_wine_type_loss: 0.0328 - val_wine_quality_root_mean_squared_error: 0.5868 - val_wine_type_accuracy: 0.9949



```python
utils.test_history(history)
```

    [92m All public tests passed



```python
# Gather the training metrics
loss, wine_quality_loss, wine_type_loss, wine_quality_rmse, wine_type_accuracy = model.evaluate(x=norm_val_X, y=val_Y)

print()
print(f'loss: {loss}')
print(f'wine_quality_loss: {wine_quality_loss}')
print(f'wine_type_loss: {wine_type_loss}')
print(f'wine_quality_rmse: {wine_quality_rmse}')
print(f'wine_type_accuracy: {wine_type_accuracy}')

# EXPECTED VALUES
# ~ 0.30 - 0.38
# ~ 0.30 - 0.38
# ~ 0.018 - 0.036
# ~ 0.50 - 0.62
# ~ 0.97 - 1.0

# Example:
#0.3657050132751465
#0.3463745415210724
#0.019330406561493874
#0.5885359048843384
#0.9974651336669922
```

    789/789 [==============================] - 0s 21us/sample - loss: 0.3775 - wine_quality_loss: 0.3431 - wine_type_loss: 0.0328 - wine_quality_root_mean_squared_error: 0.5868 - wine_type_accuracy: 0.9949

    loss: 0.37753465900100835
    wine_quality_loss: 0.34310317039489746
    wine_type_loss: 0.03278212249279022
    wine_quality_rmse: 0.5868411660194397
    wine_type_accuracy: 0.9949302673339844


## Analyze the Model Performance

Note that the model has two outputs. The output at index 0 is quality and index 1 is wine type

So, round the quality predictions to the nearest integer.


```python
predictions = model.predict(norm_test_X)
quality_pred = predictions[0]
type_pred = predictions[1]
```


```python
print(quality_pred[0])

# EXPECTED OUTPUT
# 5.4 - 6.0
```

    [5.6106915]



```python
print(type_pred[0])
print(type_pred[944])

# EXPECTED OUTPUT
# A number close to zero
# A number close to or equal to 1
```

    [0.00260193]
    [0.99981433]


### Plot Utilities

We define a few utilities to visualize the model performance.


```python
def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name],color='blue',label=metric_name)
    plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)

```


```python
def plot_confusion_matrix(y_true, y_pred, title='', labels=[0,1]):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="black" if cm[i, j] > thresh else "white")
    plt.show()
```


```python
def plot_diff(y_true, y_pred, title = '' ):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-100, 100], [-100, 100])
    return plt
```

### Plots for Metrics


```python
plot_metrics('wine_quality_root_mean_squared_error', 'RMSE', ylim=2)
```


![png](output_63_0.png)



```python
plot_metrics('wine_type_loss', 'Wine Type Loss', ylim=0.2)
```


![png](output_64_0.png)


### Plots for Confusion Matrix

Plot the confusion matrices for wine type. You can see that the model performs well for prediction of wine type from the confusion matrix and the loss metrics.


```python
plot_confusion_matrix(test_Y[1], np.round(type_pred), title='Wine Type', labels = [0, 1])
```


![png](output_66_0.png)



```python
scatter_plot = plot_diff(test_Y[0], quality_pred, title='Type')
```


![png](output_67_0.png)

