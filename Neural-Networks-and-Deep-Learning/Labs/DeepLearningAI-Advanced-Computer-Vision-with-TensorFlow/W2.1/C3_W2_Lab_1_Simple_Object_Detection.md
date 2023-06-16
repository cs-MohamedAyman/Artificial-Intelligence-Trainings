# Simple Object Detection in Tensorflow

This lab will walk you through how to use object detection models available in [Tensorflow Hub](https://tensorflow.org/hub). In the following sections, you will:

* explore the Tensorflow Hub for object detection models
* load the models in your workspace
* preprocess an image for inference
* run inference on the models and inspect the output

Let's get started!


```python
# Install this package to use Colab's GPU for training
!apt install --allow-change-held-packages libcudnn8=8.4.1.50-1+cuda11.6
```

## Imports


```python
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from PIL import ImageOps
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
```

### Download the model from Tensorflow Hub

Tensorflow Hub is a repository of trained machine learning models which you can reuse in your own projects.
- You can see the domains covered [here](https://tfhub.dev/) and its subcategories.
- For this lab, you will want to look at the [image object detection subcategory](https://tfhub.dev/s?module-type=image-object-detection).
- You can select a model to see more information about it and copy the URL so you can download it to your workspace.
- We selected a [inception resnet version 2](https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1)
- You can also modify this following cell to choose the other model that we selected, [ssd mobilenet version 2](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2)


```python
# you can switch the commented lines here to pick the other model

# inception resnet version 2
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

# You can choose ssd mobilenet version 2 instead and compare the results
#module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
```

#### Load the model

Next, you'll load the model specified by the `module_handle`.
- This will take a few minutes to load the model.


```python
model = hub.load(module_handle)
```

#### Choose the default signature

Some models in the Tensorflow hub can be used for different tasks. So each model's documentation should show what *signature* to use when running the model.
- If you want to see if a model has more than one signature then you can do something like `print(hub.load(module_handle).signatures.keys())`. In your case, the models you will be using only have the `default` signature so you don't have to worry about other types.


```python
# take a look at the available signatures for this particular model
model.signatures.keys()
```

Please choose the 'default' signature for your object detector.
- For object detection models, its 'default' signature will accept a batch of image tensors and output a dictionary describing the objects detected, which is what you'll want here.


```python
detector = model.signatures['default']
```

### download_and_resize_image

This function downloads an image specified by a given "url", pre-processes it, and then saves it to disk.


```python
def download_and_resize_image(url, new_width=256, new_height=256):
    '''
    Fetches an image online, resizes it and saves it locally.

    Args:
        url (string) -- link to the image
        new_width (int) -- size in pixels used for resizing the width of the image
        new_height (int) -- size in pixels used for resizing the length of the image

    Returns:
        (string) -- path to the saved image
    '''


    # create a temporary file ending with ".jpg"
    _, filename = tempfile.mkstemp(suffix=".jpg")

    # opens the given URL
    response = urlopen(url)

    # reads the image fetched from the URL
    image_data = response.read()

    # puts the image data in memory buffer
    image_data = BytesIO(image_data)

    # opens the image
    pil_image = Image.open(image_data)

    # resizes the image. will crop if aspect ratio is different.
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)

    # converts to the RGB colorspace
    pil_image_rgb = pil_image.convert("RGB")

    # saves the image to the temporary file created earlier
    pil_image_rgb.save(filename, format="JPEG", quality=90)

    print("Image downloaded to %s." % filename)

    return filename
```

### Download and preprocess an image

Now, using `download_and_resize_image` you can get a sample image online and save it locally.
- We've provided a URL for you, but feel free to choose another image to run through the object detector.
- You can use the original width and height of the image but feel free to modify it and see what results you get.


```python
# You can choose a different URL that points to an image of your choice
image_url = "https://upload.wikimedia.org/wikipedia/commons/f/fb/20130807_dublin014.JPG"

# download the image and use the original height and width
downloaded_image_path = download_and_resize_image(image_url, 3872, 2592)
```

### run_detector

This function will take in the object detection model `detector` and the path to a sample image, then use this model to detect objects and display its predicted class categories and detection boxes.
- run_detector uses `load_image` to convert the image into a tensor.


```python
def load_img(path):
    '''
    Loads a JPEG image and converts it to a tensor.

    Args:
        path (string) -- path to a locally saved JPEG image

    Returns:
        (tensor) -- an image tensor
    '''

    # read the file
    img = tf.io.read_file(path)

    # convert to a tensor
    img = tf.image.decode_jpeg(img, channels=3)

    return img


def run_detector(detector, path):
    '''
    Runs inference on a local file using an object detection model.

    Args:
        detector (model) -- an object detection model loaded from TF Hub
        path (string) -- path to an image saved locally
    '''

    # load an image tensor from a local file path
    img = load_img(path)

    # add a batch dimension in front of the tensor
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

    # run inference using the model
    result = detector(converted_img)

    # save the results in a dictionary
    result = {key:value.numpy() for key,value in result.items()}

    # print results
    print("Found %d objects." % len(result["detection_scores"]))

    print(result["detection_scores"])
    print(result["detection_class_entities"])
    print(result["detection_boxes"])

```

### Run inference on the image

You can run your detector by calling the `run_detector` function. This will print the number of objects found followed by three lists:

* The detection scores of each object found (i.e. how confident the model is),
* The classes of each object found,
* The bounding boxes of each object

You will see how to overlay this information on the original image in the next sections and in this week's assignment!


```python
# runs the object detection model and prints information about the objects found
run_detector(detector, downloaded_image_path)
```
