```python
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad.
Create a convolutional neural network that trains to 100% accuracy on these images,  which cancels training upon hitting training accuracy of >.999

Hint -- it will work best with 3 convolutional layers.


```python
import tensorflow as tf
import os
import zipfile


DESIRED_ACCURACY = 0.999

!wget --no-check-certificate \
    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
    -O "/tmp/happy-or-sad.zip"

zip_ref = zipfile.ZipFile("/tmp/happy-or-sad.zip", 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

class myCallback(# Your Code):
  # Your Code

callbacks = myCallback()
```


```python
# This Code Block should Define and Compile the Model
model = tf.keras.models.Sequential([
# Your Code Here
])

from tensorflow.keras.optimizers import RMSprop

model.compile(# Your Code Here #)
```


```python
# This code block should create an instance of an ImageDataGenerator called train_datagen
# And a train_generator by calling train_datagen.flow_from_directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = # Your Code Here

train_generator = train_datagen.flow_from_directory(
        # Your Code Here)

# Expected output: 'Found 80 images belonging to 2 classes'
```


```python
# This code block should call model.fit and train for
# a number of epochs.
history = model.fit(
      # Your Code Here)

# Expected output: "Reached 99.9% accuracy so cancelling training!""
```
