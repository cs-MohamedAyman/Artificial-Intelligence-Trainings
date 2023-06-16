<a href="https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W2/ungraded_labs/C4_W2_Lab_1_features_and_labels.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Ungraded Lab: Preparing Time Series Features and Labels

In this lab, you will prepare time series data into features and labels that you can use to train a model. This is mainly achieved by a *windowing* technique where in you group consecutive measurement values into one feature and the next measurement will be the label. For example, in hourly measurements, you can use values taken at hours 1 to 11 to predict the value at hour 12. The next sections will show how you can implement this in Tensorflow.

Let's begin!

## Imports

Tensorflow will be your lone import in this module and you'll be using methods mainly from the [tf.data API](https://tensorflow.org/guide/data), particularly the [tf.data.Dataset](https://tensorflow.org/api_docs/python/tf/data/Dataset) class. This contains many useful methods to arrange sequences of data and you'll see that shortly.


```python
import tensorflow as tf
```

## Create a Simple Dataset

For this exercise, you will just use a sequence of numbers as your dataset so you can clearly see the effect of each command. For example, the cell below uses the [range()](https://tensorflow.org/api_docs/python/tf/data/Dataset#range) method to generate a dataset containing numbers 0 to 9.


```python
# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Preview the result
for val in dataset:
   print(val.numpy())
```

You will see this command several times in the next sections.

## Windowing the data

As mentioned earlier, you want to group consecutive elements of your data and use that to predict a future value. This is called windowing and you can use that with the [window()](https://tensorflow.org/api_docs/python/tf/data/Dataset#window) method as shown below. Here, you will take 5 elements per window (i.e. `size` parameter) and you will move this window 1 element at a time (i.e. `shift` parameter). One caveat to using this method is that each window returned is a [Dataset](https://tensorflow.org/guide/data#dataset_structure) in itself. This is a Python iterable and, as of the current version (TF 2.8), it won't show the elements if you use the `print()` method on it. It will just show a description of the data structure (e.g. `<_VariantDataset shapes: (), types: tf.int64>`).


```python
# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data
dataset = dataset.window(size=5, shift=1)

# Print the result
for window_dataset in dataset:
  print(window_dataset)
```

If you want to see the elements, you will have to iterate over each iterable. This can be done by modifying the print statement above with a nested for-loop or list comprehension. The code below shows the list comprehension while in the lecture video, you saw the for-loop.


```python
# Print the result
for window_dataset in dataset:
  print([item.numpy() for item in window_dataset])
```

Now that you can see the elements of each window, you'll notice that the resulting sets are not sized evenly because there are no more elements after the number `9`. You can use the `drop_remainder` flag to make sure that only 5-element windows are retained.


```python
# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data but only take those with the specified size
dataset = dataset.window(size=5, shift=1, drop_remainder=True)

# Print the result
for window_dataset in dataset:
  print([item.numpy() for item in window_dataset])
```

## Flatten the Windows

In training the model later, you will want to prepare the windows to be [tensors](https://tensorflow.org/guide/tensor) instead of the `Dataset` structure. You can do that by feeding a mapping function to the [flat_map()](https://tensorflow.org/api_docs/python/tf/data/Dataset#flat_map) method. This function will be applied to each window and the results will be [flattened into a single dataset](https://tensorflow.org/api_docs/python/tf/data/Dataset#flatten_a_dataset_of_windows_2). To illustrate, the code below will put all elements of a window into a single batch then flatten the result.


```python
# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data but only take those with the specified size
dataset = dataset.window(5, shift=1, drop_remainder=True)

# Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

# Print the results
for window in dataset:
  print(window.numpy())
```

## Group into features and labels

Next, you will want to mark the labels in each window. For this exercise, you will do that by splitting the last element of each window from the first four. This is done with the [map()](https://tensorflow.org/api_docs/python/tf/data/Dataset#map) method containing a lambda function that defines the window slicing.


```python
# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data but only take those with the specified size
dataset = dataset.window(5, shift=1, drop_remainder=True)

# Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

# Create tuples with features (first four elements of the window) and labels (last element)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

# Print the results
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())
  print()
```

## Shuffle the data

It is good practice to shuffle your dataset to reduce *sequence bias* while training your model. This refers to the neural network overfitting to the order of inputs and consequently, it will not perform well when it does not see that particular order when testing. You don't want the sequence of training inputs to impact the network this way so it's good to shuffle them up.

You can simply use the [shuffle()](https://tensorflow.org/api_docs/python/tf/data/Dataset#shuffle) method to do this. The `buffer_size` parameter is required for that and as mentioned in the doc, you should put a number equal or greater than the total number of elements for better shuffling. We can see from the previous cells that the total number of windows in the dataset is `6` so we can choose this number or higher.


```python
# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data but only take those with the specified size
dataset = dataset.window(5, shift=1, drop_remainder=True)

# Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

# Create tuples with features (first four elements of the window) and labels (last element)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

# Shuffle the windows
dataset = dataset.shuffle(buffer_size=10)

# Print the results
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())
  print()

```

## Create batches for training

Lastly, you will want to group your windows into batches. You can do that with the [batch()](https://tensorflow.org/api_docs/python/tf/data/Dataset#batch) method as shown below. Simply specify the batch size and it will return a batched dataset with that number of windows. As a rule of thumb, it is also good to specify a [prefetch()](https://tensorflow.org/api_docs/python/tf/data/Dataset#prefetch) step. This optimizes the execution time when the model is already training. By specifying a prefetch `buffer_size` of `1` as shown below, Tensorflow will prepare the next one batch in advance (i.e. putting it in a buffer) while the current batch is being consumed by the model. You can read more about it [here](https://towardsdatascience.com/optimising-your-input-pipeline-performance-with-tf-data-part-1-32e52a30cac4#Prefetching).


```python
# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data but only take those with the specified size
dataset = dataset.window(5, shift=1, drop_remainder=True)

# Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

# Create tuples with features (first four elements of the window) and labels (last element)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

# Shuffle the windows
dataset = dataset.shuffle(buffer_size=10)

# Create batches of windows
dataset = dataset.batch(2).prefetch(1)

# Print the results
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())
  print()

```

## Wrap Up

This short exercise showed you how to chain different methods of the `tf.data.Dataset` class to prepare a sequence into shuffled and batched window datasets. You will be using this same concept in the next exercises when you apply it to synthetic data and use the result to train a neural network. On to the next!
