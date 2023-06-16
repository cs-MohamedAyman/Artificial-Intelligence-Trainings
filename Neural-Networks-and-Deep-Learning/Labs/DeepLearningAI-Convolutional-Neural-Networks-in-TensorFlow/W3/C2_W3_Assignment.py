#!/usr/bin/env python
# coding: utf-8

# # Week 3: Transfer Learning
#
# Welcome to this assignment! This week, you are going to use a technique called `Transfer Learning` in which you utilize an already trained network to help you solve a similar problem to the one it was originally trained to solve.
#
# Let's get started!

# In[ ]:


import os
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# ## Dataset
#
# For this assignment, you will use the `Horse or Human dataset`, which contains images of horses and humans.
#
# Download the `training` and `validation` sets by running the cell below:

# In[ ]:


# Get the Horse or Human training dataset
get_ipython().system('wget -q -P /content/ https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip')

# Get the Horse or Human validation dataset
get_ipython().system('wget -q -P /content/ https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip')

test_local_zip = './horse-or-human.zip'
zip_ref = zipfile.ZipFile(test_local_zip, 'r')
zip_ref.extractall('/tmp/training')

val_local_zip = './validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(val_local_zip, 'r')
zip_ref.extractall('/tmp/validation')

zip_ref.close()


# This dataset already has an structure that is compatible with Keras' `flow_from_directory` so you don't need to move the images into subdirectories as you did in the previous assignments. However, it is still a good idea to save the paths of the images so you can use them later on:

# In[ ]:


# Define the training and validation base directories
train_dir = '/tmp/training'
validation_dir = '/tmp/validation'

# Directory with training horse pictures
train_horses_dir = os.path.join(train_dir, 'horses')
# Directory with training humans pictures
train_humans_dir = os.path.join(train_dir, 'humans')
# Directory with validation horse pictures
validation_horses_dir = os.path.join(validation_dir, 'horses')
# Directory with validation human pictures
validation_humans_dir = os.path.join(validation_dir, 'humans')

# Check the number of images for each class and set
print(f"There are {len(os.listdir(train_horses_dir))} images of horses for training.\n")
print(f"There are {len(os.listdir(train_humans_dir))} images of humans for training.\n")
print(f"There are {len(os.listdir(validation_horses_dir))} images of horses for validation.\n")
print(f"There are {len(os.listdir(validation_humans_dir))} images of humans for validation.\n")


# Now take a look at a sample image of each one of the classes:

# In[ ]:


print("Sample horse image:")
plt.imshow(load_img(f"{os.path.join(train_horses_dir, os.listdir(train_horses_dir)[0])}"))
plt.show()

print("\nSample human image:")
plt.imshow(load_img(f"{os.path.join(train_humans_dir, os.listdir(train_humans_dir)[0])}"))
plt.show()


# `matplotlib` makes it easy to see that these images have a resolution of 300x300 and are colored, but you can double check this by using the code below:

# In[ ]:


# Load the first example of a horse
sample_image  = load_img(f"{os.path.join(train_horses_dir, os.listdir(train_horses_dir)[0])}")

# Convert the image into its numpy array representation
sample_array = img_to_array(sample_image)

print(f"Each image has shape: {sample_array.shape}")


# As expected, the sample image has a resolution of 300x300 and the last dimension is used for each one of the RGB channels to represent color.

# ## Training and Validation Generators
#
# Now that you know the images you are dealing with, it is time for you to code the generators that will fed these images to your Network. For this, complete the `train_val_generators` function below:
#
# **Important Note:** The images have a resolution of 300x300 but the `flow_from_directory` method you will use allows you to set a target resolution. In this case, **set a `target_size` of (150, 150)**. This will heavily lower the number of trainable parameters in your final network, yielding much quicker training times without compromising the accuracy!

# In[ ]:


# GRADED FUNCTION: train_val_generators
def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
  """
  Creates the training and validation data generators

  Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images

  Returns:
    train_generator, validation_generator: tuple containing the generators
  """
  ### START CODE HERE

  # Instantiate the ImageDataGenerator class
  # Don't forget to normalize pixel values and set arguments to augment the images
  train_datagen = None

  # Pass in the appropriate arguments to the flow_from_directory method
  train_generator = train_datagen.flow_from_directory(directory=None,
                                                      batch_size=32,
                                                      class_mode=None,
                                                      target_size=(None, None))

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  # Remember that validation data should not be augmented
  validation_datagen = None

  # Pass in the appropriate arguments to the flow_from_directory method
  validation_generator = validation_datagen.flow_from_directory(directory=None,
                                                                batch_size=32,
                                                                class_mode=None,
                                                                target_size=(None, None))
  ### END CODE HERE
  return train_generator, validation_generator


# In[ ]:


# Test your generators
train_generator, validation_generator = train_val_generators(train_dir, validation_dir)


# **Expected Output:**
# ```
# Found 1027 images belonging to 2 classes.
# Found 256 images belonging to 2 classes.
# ```

# ## Transfer learning - Create the pre-trained model
#
# Download the `inception V3` weights into the `/tmp/` directory:

# In[ ]:


# Download the inception v3 weights
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')


# Now load the `InceptionV3` model and save the path to the weights you just downloaded:

# In[ ]:


# Import the inception model
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


# Complete the `create_pre_trained_model` function below. You should specify the correct `input_shape` for the model (remember that you set a new resolution for the images instead of the native 300x300) and make all of the layers non-trainable:

# In[ ]:


# GRADED FUNCTION: create_pre_trained_model
def create_pre_trained_model(local_weights_file):
  """
  Initializes an InceptionV3 model.

  Args:
    local_weights_file (string): path pointing to a pretrained weights H5 file

  Returns:
    pre_trained_model: the initialized InceptionV3 model
  """
  ### START CODE HERE
  pre_trained_model = InceptionV3(input_shape = (None, None, None),
                                  include_top = False,
                                  weights = None)

  pre_trained_model.load_weights(local_weights_file)

  # Make all the layers in the pre-trained model non-trainable
  for None in None:
    None = None

  ### END CODE HERE

  return pre_trained_model



# Check that everything went well by comparing the last few rows of the model summary to the expected output:

# In[ ]:


pre_trained_model = create_pre_trained_model(local_weights_file)

# Print the model summary
pre_trained_model.summary()


# **Expected Output:**
# ```
# batch_normalization_v1_281 (Bat (None, 3, 3, 192)    576         conv2d_281[0][0]
# __________________________________________________________________________________________________
# activation_273 (Activation)     (None, 3, 3, 320)    0           batch_normalization_v1_273[0][0]
# __________________________________________________________________________________________________
# mixed9_1 (Concatenate)          (None, 3, 3, 768)    0           activation_275[0][0]
#                                                                 activation_276[0][0]
# __________________________________________________________________________________________________
# concatenate_5 (Concatenate)     (None, 3, 3, 768)    0           activation_279[0][0]
#                                                                 activation_280[0][0]
# __________________________________________________________________________________________________
# activation_281 (Activation)     (None, 3, 3, 192)    0           batch_normalization_v1_281[0][0]
# __________________________________________________________________________________________________
# mixed10 (Concatenate)           (None, 3, 3, 2048)   0           activation_273[0][0]
#                                                                 mixed9_1[0][0]
#                                                                 concatenate_5[0][0]
#                                                                 activation_281[0][0]
# ==================================================================================================
# Total params: 21,802,784
# Trainable params: 0
# Non-trainable params: 21,802,784
#
#
# ```

# To check that all the layers in the model were set to be non-trainable, you can also run the cell below:

# In[ ]:


total_params = pre_trained_model.count_params()
num_trainable_params = sum([w.shape.num_elements() for w in pre_trained_model.trainable_weights])

print(f"There are {total_params:,} total parameters in this model.")
print(f"There are {num_trainable_params:,} trainable parameters in this model.")


# **Expected Output:**
# ```
# There are 21,802,784 total parameters in this model.
# There are 0 trainable parameters in this model.
# ```

# ## Creating callbacks for later
#
# You have already worked with callbacks in the first course of this specialization so the callback to stop training once an accuracy of 99.9% is reached, is provided for you:

# In[ ]:


# Define a Callback class that stops training once accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True


# ## Pipelining the pre-trained model with your own
#
# Now that the pre-trained model is ready, you need to "glue" it to your own model to solve the task at hand.
#
# For this you will need the last output of the pre-trained model, since this will be the input for your own. Complete the `output_of_last_layer` function below.
#
# **Note:** For grading purposes use the `mixed7` layer as the last layer of the pre-trained model. However, after submitting feel free to come back here and play around with this.

# In[ ]:


# GRADED FUNCTION: output_of_last_layer
def output_of_last_layer(pre_trained_model):
  """
  Gets the last layer output of a model

  Args:
    pre_trained_model (tf.keras Model): model to get the last layer output from

  Returns:
    last_output: output of the model's last layer
  """
  ### START CODE HERE
  last_desired_layer = None
  print('last layer output shape: ', last_desired_layer.output_shape)
  last_output = None
  print('last layer output: ', last_output)
  ### END CODE HERE

  return last_output


# Check that everything works as expected:

# In[ ]:


last_output = output_of_last_layer(pre_trained_model)


# **Expected Output (if `mixed7` layer was used):**
# ```
# last layer output shape:  (None, 7, 7, 768)
# last layer output:  KerasTensor(type_spec=TensorSpec(shape=(None, 7, 7, 768), dtype=tf.float32, name=None), name='mixed7/concat:0', description="created by layer 'mixed7'")
# ```

# Now you will create the final model by adding some additional layers on top of the pre-trained model.
#
# Complete the `create_final_model` function below. You will need to use Tensorflow's [Functional API](https://tensorflow.org/guide/keras/functional) for this since the pretrained model has been created using it.
#
# Let's double check this first:

# In[ ]:


# Print the type of the pre-trained model
print(f"The pretrained model has type: {type(pre_trained_model)}")


# To create the final model, you will use Keras' Model class by defining the appropriate inputs and outputs as described in the first way to instantiate a Model in the [docs](https://tensorflow.org/api_docs/python/tf/keras/Model).
#
# Note that you can get the input from any existing model by using its `input` attribute and by using the Funcional API you can use the last layer directly as output when creating the final model.

# In[ ]:


# GRADED FUNCTION: create_final_model
def create_final_model(pre_trained_model, last_output):
  """
  Appends a custom model to a pre-trained model

  Args:
    pre_trained_model (tf.keras Model): model that will accept the train/test inputs
    last_output (tensor): last layer output of the pre-trained model

  Returns:
    model: the combined model
  """
  # Flatten the output layer to 1 dimension
  x = layers.Flatten()(last_output)

  ### START CODE HERE

  # Add a fully connected layer with 1024 hidden units and ReLU activation
  x = None
  # Add a dropout rate of 0.2
  x = None
  # Add a final sigmoid layer for classification
  x = None

  # Create the complete model by using the Model class
  model = Model(inputs=None, outputs=None)

  # Compile the model
  model.compile(optimizer = RMSprop(learning_rate=0.0001),
                loss = None,
                metrics = [None])

  ### END CODE HERE

  return model


# In[ ]:


# Save your model in a variable
model = create_final_model(pre_trained_model, last_output)

# Inspect parameters
total_params = model.count_params()
num_trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])

print(f"There are {total_params:,} total parameters in this model.")
print(f"There are {num_trainable_params:,} trainable parameters in this model.")


# **Expected Output:**
# ```
# There are 47,512,481 total parameters in this model.
# There are 38,537,217 trainable parameters in this model.
# ```

# Wow, that is a lot of parameters!
#
# After submitting your assignment later, try re-running this notebook but use the original resolution of 300x300, you will be surprised to see how many more parameters are for that case.
#
# Now train the model:

# In[ ]:


# Run this and see how many epochs it should take before the callback
# fires, and stops training at 99.9% accuracy
# (It should take a few epochs)
callbacks = myCallback()
history = model.fit(train_generator,
                    validation_data = validation_generator,
                    epochs = 100,
                    verbose = 2,
                    callbacks=callbacks)


# The training should have stopped after less than 10 epochs and it should have reached an accuracy over 99,9% (firing the callback). This happened so quickly because of the pre-trained model you used, which already contained information to classify humans from horses. Really cool!
#
# Now take a quick look at the training and validation accuracies for each epoch of training:

# In[ ]:


# Plot the training and validation accuracies for each epoch

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


# You will need to submit this notebook for grading. To download it, click on the `File` tab in the upper left corner of the screen then click on `Download` -> `Download .ipynb`. You can name it anything you want as long as it is a valid `.ipynb` (jupyter notebook) file.

# **Congratulations on finishing this week's assignment!**
#
# You have successfully implemented a convolutional neural network that leverages a pre-trained network to help you solve the problem of classifying humans from horses.
#
# **Keep it up!**
