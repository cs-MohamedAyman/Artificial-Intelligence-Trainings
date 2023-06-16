#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W2/ungraded_labs/C1_W2_Lab_2_callbacks.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Ungraded Lab: Using Callbacks to Control Training
#
# In this lab, you will use the [Callbacks API](https://keras.io/api/callbacks/) to stop training when a specified metric is met. This is a useful feature so you won't need to complete all epochs when this threshold is reached. For example, if you set 1000 epochs and your desired accuracy is already reached at epoch 200, then the training will automatically stop. Let's see how this is implemented in the next sections.
#

# ## Load and Normalize the Fashion MNIST dataset
#
# Like the previous lab, you will use the Fashion MNIST dataset again for this exercise. And also as mentioned before, you will normalize the pixel values to help optimize the training.

# In[ ]:


import tensorflow as tf

# Instantiate the dataset API
fmnist = tf.keras.datasets.fashion_mnist

# Load the dataset
(x_train, y_train),(x_test, y_test) = fmnist.load_data()

# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0


# ## Creating a Callback class
#
# You can create a callback by defining a class that inherits the [tf.keras.callbacks.Callback](https://tensorflow.org/api_docs/python/tf/keras/callbacks/Callback) base class. From there, you can define available methods to set where the callback will be executed. For instance below, you will use the [on_epoch_end()](https://tensorflow.org/api_docs/python/tf/keras/callbacks/Callback#on_epoch_end) method to check the loss at each training epoch.

# In[ ]:


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    '''
    Halts the training after reaching 60 percent accuracy

    Args:
      epoch (integer) - index of epoch (required but unused in the function definition below)
      logs (dict) - metric results from the training epoch
    '''

    # Check accuracy
    if(logs.get('loss') < 0.4):

      # Stop if threshold is met
      print("\nLoss is lower than 0.4 so cancelling training!")
      self.model.stop_training = True

# Instantiate class
callbacks = myCallback()


# ## Define and compile the model
#
# Next, you will define and compile the model. The architecture will be similar to the one you built in the previous lab. Afterwards, you will set the optimizer, loss, and metrics that you will use for training.

# In[ ]:


# Define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ### Train the model
#
# Now you are ready to train the model. To set the callback, simply set the `callbacks` parameter to the `myCallback` instance you declared before. Run the cell below and observe what happens.

# In[ ]:


# Train the model with a callback
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])


# You will notice that the training does not need to complete all 10 epochs. By having a callback at each end of the epoch, it is able to check the training parameters and compare if it meets the threshold you set in the function definition. In this case, it will simply stop when the loss falls below `0.40` after the current epoch.
#
# *Optional Challenge: Modify the code to make the training stop when the accuracy metric exceeds 60%.*
#
# That concludes this simple exercise on callbacks!
