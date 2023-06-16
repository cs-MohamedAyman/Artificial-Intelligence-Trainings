<a href="https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-3-public/blob/main/Course%202%20-%20Custom%20Training%20loops%2C%20Gradients%20and%20Distributed%20Training/Week%204%20-%20Distribution%20Strategy/C2_W4_Lab_4_one-device-strategy.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# One Device Strategy

In this ungraded lab, you'll learn how to set up a [One Device Strategy](https://tensorflow.org/api_docs/python/tf/distribute/OneDeviceStrategy). This is typically used to deliberately test your code on a single device. This can be used before switching to a different strategy that distributes across multiple devices. Please click on the **Open in Colab** badge above so you can download the datasets and use a GPU-enabled lab environment.

## Imports


```python
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()
```

## Define the Distribution Strategy

You can list available devices in your machine and specify a device type. This allows you to verify the device name to pass in `tf.distribute.OneDeviceStrategy()`.


```python
# choose a device type such as CPU or GPU
devices = tf.config.list_physical_devices('CPU')
print(devices[0])

# You'll see that the name will look something like "/physical_device:GPU:0"
# Just take the GPU:0 part and use that as the name
gpu_name = "GPU:0"

# define the strategy and pass in the device name
one_strategy = tf.distribute.OneDeviceStrategy(device=gpu_name)
```

    PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')


## Parameters

We'll define a few global variables for setting up the model and dataset.


```python
pixels = 224
MODULE_HANDLE = 'https://tfhub.dev/tensorflow/resnet_50/feature_vector/1'
IMAGE_SIZE = (pixels, pixels)
BATCH_SIZE = 32

print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))
```

    Using https://tfhub.dev/tensorflow/resnet_50/feature_vector/1 with input size (224, 224)


## Download and Prepare the Dataset

We will use the [Cats vs Dogs](https://tensorflow.org/datasets/catalog/cats_vs_dogs) dataset and we will fetch it via TFDS.


```python
splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']

(train_examples, validation_examples, test_examples), info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True, split=splits)

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
```

    [1mDownloading and preparing dataset cats_vs_dogs/4.0.0 (download: 786.68 MiB, generated: Unknown size, total: 786.68 MiB) to /home/jovyan/tensorflow_datasets/cats_vs_dogs/4.0.0...[0m



    ---------------------------------------------------------------------------

    Error                                     Traceback (most recent call last)

    /opt/conda/lib/python3.7/site-packages/urllib3/contrib/pyopenssl.py in wrap_socket(self, sock, server_side, do_handshake_on_connect, suppress_ragged_eofs, server_hostname)
        487             try:
    --> 488                 cnx.do_handshake()
        489             except OpenSSL.SSL.WantReadError:


    /opt/conda/lib/python3.7/site-packages/OpenSSL/SSL.py in do_handshake(self)
       1933         result = _lib.SSL_do_handshake(self._ssl)
    -> 1934         self._raise_ssl_error(self._ssl, result)
       1935


    /opt/conda/lib/python3.7/site-packages/OpenSSL/SSL.py in _raise_ssl_error(self, ssl, result)
       1670         else:
    -> 1671             _raise_current_error()
       1672


    /opt/conda/lib/python3.7/site-packages/OpenSSL/_util.py in exception_from_error_queue(exception_type)
         53
    ---> 54     raise exception_type(errors)
         55


    Error: [('SSL routines', 'tls_process_server_certificate', 'certificate verify failed')]


    During handling of the above exception, another exception occurred:


    SSLError                                  Traceback (most recent call last)

    /opt/conda/lib/python3.7/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        676                 headers=headers,
    --> 677                 chunked=chunked,
        678             )


    /opt/conda/lib/python3.7/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        380         try:
    --> 381             self._validate_conn(conn)
        382         except (SocketTimeout, BaseSSLError) as e:


    /opt/conda/lib/python3.7/site-packages/urllib3/connectionpool.py in _validate_conn(self, conn)
        975         if not getattr(conn, "sock", None):  # AppEngine might not have  `.sock`
    --> 976             conn.connect()
        977


    /opt/conda/lib/python3.7/site-packages/urllib3/connection.py in connect(self)
        369             server_hostname=server_hostname,
    --> 370             ssl_context=context,
        371         )


    /opt/conda/lib/python3.7/site-packages/urllib3/util/ssl_.py in ssl_wrap_socket(sock, keyfile, certfile, cert_reqs, ca_certs, server_hostname, ssl_version, ciphers, ssl_context, ca_cert_dir, key_password, ca_cert_data)
        376         if HAS_SNI and server_hostname is not None:
    --> 377             return context.wrap_socket(sock, server_hostname=server_hostname)
        378


    /opt/conda/lib/python3.7/site-packages/urllib3/contrib/pyopenssl.py in wrap_socket(self, sock, server_side, do_handshake_on_connect, suppress_ragged_eofs, server_hostname)
        493             except OpenSSL.SSL.Error as e:
    --> 494                 raise ssl.SSLError("bad handshake: %r" % e)
        495             break


    SSLError: ("bad handshake: Error([('SSL routines', 'tls_process_server_certificate', 'certificate verify failed')])",)


    During handling of the above exception, another exception occurred:


    MaxRetryError                             Traceback (most recent call last)

    /opt/conda/lib/python3.7/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        448                     retries=self.max_retries,
    --> 449                     timeout=timeout
        450                 )


    /opt/conda/lib/python3.7/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        724             retries = retries.increment(
    --> 725                 method, url, error=e, _pool=self, _stacktrace=sys.exc_info()[2]
        726             )


    /opt/conda/lib/python3.7/site-packages/urllib3/util/retry.py in increment(self, method, url, response, error, _pool, _stacktrace)
        438         if new_retry.is_exhausted():
    --> 439             raise MaxRetryError(_pool, url, error or ResponseError(cause))
        440


    MaxRetryError: HTTPSConnectionPool(host='download.microsoft.com', port=443): Max retries exceeded with url: /download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip (Caused by SSLError(SSLError("bad handshake: Error([('SSL routines', 'tls_process_server_certificate', 'certificate verify failed')])")))


    During handling of the above exception, another exception occurred:


    SSLError                                  Traceback (most recent call last)

    <ipython-input-5-2029b6c09494> in <module>
          1 splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']
          2
    ----> 3 (train_examples, validation_examples, test_examples), info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True, split=splits)
          4
          5 num_examples = info.splits['train'].num_examples


    /opt/conda/lib/python3.7/site-packages/tensorflow_datasets/core/api_utils.py in disallow_positional_args_dec(fn, instance, args, kwargs)
         67     _check_no_positional(fn, args, ismethod, allowed=allowed)
         68     _check_required(fn, kwargs)
    ---> 69     return fn(*args, **kwargs)
         70
         71   return disallow_positional_args_dec(wrapped)  # pylint: disable=no-value-for-parameter


    /opt/conda/lib/python3.7/site-packages/tensorflow_datasets/core/registered.py in load(name, split, data_dir, batch_size, shuffle_files, download, as_supervised, decoders, read_config, with_info, builder_kwargs, download_and_prepare_kwargs, as_dataset_kwargs, try_gcs)
        369   if download:
        370     download_and_prepare_kwargs = download_and_prepare_kwargs or {}
    --> 371     dbuilder.download_and_prepare(**download_and_prepare_kwargs)
        372
        373   if as_dataset_kwargs is None:


    /opt/conda/lib/python3.7/site-packages/tensorflow_datasets/core/api_utils.py in disallow_positional_args_dec(fn, instance, args, kwargs)
         67     _check_no_positional(fn, args, ismethod, allowed=allowed)
         68     _check_required(fn, kwargs)
    ---> 69     return fn(*args, **kwargs)
         70
         71   return disallow_positional_args_dec(wrapped)  # pylint: disable=no-value-for-parameter


    /opt/conda/lib/python3.7/site-packages/tensorflow_datasets/core/dataset_builder.py in download_and_prepare(self, download_dir, download_config)
        374           self._download_and_prepare(
        375               dl_manager=dl_manager,
    --> 376               download_config=download_config)
        377
        378           # NOTE: If modifying the lines below to put additional information in


    /opt/conda/lib/python3.7/site-packages/tensorflow_datasets/core/dataset_builder.py in _download_and_prepare(self, dl_manager, download_config)
       1017     super(GeneratorBasedBuilder, self)._download_and_prepare(
       1018         dl_manager=dl_manager,
    -> 1019         max_examples_per_split=download_config.max_examples_per_split,
       1020     )
       1021


    /opt/conda/lib/python3.7/site-packages/tensorflow_datasets/core/dataset_builder.py in _download_and_prepare(self, dl_manager, **prepare_split_kwargs)
        937         prepare_split_kwargs)
        938     for split_generator in self._split_generators(
    --> 939         dl_manager, **split_generators_kwargs):
        940       if str(split_generator.split_info.name).lower() == "all":
        941         raise ValueError(


    /opt/conda/lib/python3.7/site-packages/tensorflow_datasets/image_classification/cats_vs_dogs.py in _split_generators(self, dl_manager)
         73
         74   def _split_generators(self, dl_manager):
    ---> 75     path = dl_manager.download(_URL)
         76
         77     # There is no predefined train/val/test split for this dataset.


    /opt/conda/lib/python3.7/site-packages/tensorflow_datasets/core/download/download_manager.py in download(self, url_or_urls)
        544     # Add progress bar to follow the download state
        545     with self._downloader.tqdm():
    --> 546       return _map_promise(self._download, url_or_urls)
        547
        548   def iter_archive(self, resource):


    /opt/conda/lib/python3.7/site-packages/tensorflow_datasets/core/download/download_manager.py in _map_promise(map_fn, all_inputs)
        639   """Map the function into each element and resolve the promise."""
        640   all_promises = tf.nest.map_structure(map_fn, all_inputs)  # Apply the function
    --> 641   res = tf.nest.map_structure(_wait_on_promise, all_promises)
        642   return res


    /opt/conda/lib/python3.7/site-packages/tensorflow/python/util/nest.py in map_structure(func, *structure, **kwargs)
        633
        634   return pack_sequence_as(
    --> 635       structure[0], [func(*x) for x in entries],
        636       expand_composites=expand_composites)
        637


    /opt/conda/lib/python3.7/site-packages/tensorflow/python/util/nest.py in <listcomp>(.0)
        633
        634   return pack_sequence_as(
    --> 635       structure[0], [func(*x) for x in entries],
        636       expand_composites=expand_composites)
        637


    /opt/conda/lib/python3.7/site-packages/tensorflow_datasets/core/download/download_manager.py in _wait_on_promise(p)
        633
        634 def _wait_on_promise(p):
    --> 635   return p.get()
        636
        637


    /opt/conda/lib/python3.7/site-packages/promise/promise.py in get(self, timeout)
        510         target = self._target()
        511         self._wait(timeout or DEFAULT_TIMEOUT)
    --> 512         return self._target_settled_value(_raise=True)
        513
        514     def _target_settled_value(self, _raise=False):


    /opt/conda/lib/python3.7/site-packages/promise/promise.py in _target_settled_value(self, _raise)
        514     def _target_settled_value(self, _raise=False):
        515         # type: (bool) -> Any
    --> 516         return self._target()._settled_value(_raise)
        517
        518     _value = _reason = _target_settled_value


    /opt/conda/lib/python3.7/site-packages/promise/promise.py in _settled_value(self, _raise)
        224             if _raise:
        225                 raise_val = self._fulfillment_handler0
    --> 226                 reraise(type(raise_val), raise_val, self._traceback)
        227             return self._fulfillment_handler0
        228


    /opt/conda/lib/python3.7/site-packages/six.py in reraise(tp, value, tb)
        701             if value.__traceback__ is not tb:
        702                 raise value.with_traceback(tb)
    --> 703             raise value
        704         finally:
        705             value = None


    /opt/conda/lib/python3.7/site-packages/promise/promise.py in handle_future_result(future)
        842         # type: (Any) -> None
        843         try:
    --> 844             resolve(future.result())
        845         except Exception as e:
        846             tb = exc_info()[2]


    /opt/conda/lib/python3.7/concurrent/futures/_base.py in result(self, timeout)
        426                 raise CancelledError()
        427             elif self._state == FINISHED:
    --> 428                 return self.__get_result()
        429
        430             self._condition.wait(timeout)


    /opt/conda/lib/python3.7/concurrent/futures/_base.py in __get_result(self)
        382     def __get_result(self):
        383         if self._exception:
    --> 384             raise self._exception
        385         else:
        386             return self._result


    /opt/conda/lib/python3.7/concurrent/futures/thread.py in run(self)
         55
         56         try:
    ---> 57             result = self.fn(*self.args, **self.kwargs)
         58         except BaseException as exc:
         59             self.future.set_exception(exc)


    /opt/conda/lib/python3.7/site-packages/tensorflow_datasets/core/download/downloader.py in _sync_download(self, url, destination_path)
        143       pass
        144
    --> 145     with _open_url(url) as (response, iter_content):
        146       fname = _get_filename(response)
        147       path = os.path.join(destination_path, fname)


    /opt/conda/lib/python3.7/contextlib.py in __enter__(self)
        110         del self.args, self.kwds, self.func
        111         try:
    --> 112             return next(self.gen)
        113         except StopIteration:
        114             raise RuntimeError("generator didn't yield") from None


    /opt/conda/lib/python3.7/site-packages/tensorflow_datasets/core/download/downloader.py in _open_with_requests(url)
        189     if _DRIVE_URL.match(url):
        190       url = _get_drive_url(url, session)
    --> 191     with session.get(url, stream=True) as response:
        192       _assert_status(response)
        193       yield (response, response.iter_content(chunk_size=io.DEFAULT_BUFFER_SIZE))


    /opt/conda/lib/python3.7/site-packages/requests/sessions.py in get(self, url, **kwargs)
        541
        542         kwargs.setdefault('allow_redirects', True)
    --> 543         return self.request('GET', url, **kwargs)
        544
        545     def options(self, url, **kwargs):


    /opt/conda/lib/python3.7/site-packages/requests/sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        528         }
        529         send_kwargs.update(settings)
    --> 530         resp = self.send(prep, **send_kwargs)
        531
        532         return resp


    /opt/conda/lib/python3.7/site-packages/requests/sessions.py in send(self, request, **kwargs)
        641
        642         # Send the request
    --> 643         r = adapter.send(request, **kwargs)
        644
        645         # Total elapsed time of the request (approximately)


    /opt/conda/lib/python3.7/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        512             if isinstance(e.reason, _SSLError):
        513                 # This branch is for urllib3 v1.22 and later.
    --> 514                 raise SSLError(e, request=request)
        515
        516             raise ConnectionError(e, request=request)


    SSLError: HTTPSConnectionPool(host='download.microsoft.com', port=443): Max retries exceeded with url: /download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip (Caused by SSLError(SSLError("bad handshake: Error([('SSL routines', 'tls_process_server_certificate', 'certificate verify failed')])")))



```python
# resize the image and normalize pixel values
def format_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    return  image, label
```


```python
# prepare batches
train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test_examples.map(format_image).batch(1)
```


```python
# check if the batches have the correct size and the images have the correct shape
for image_batch, label_batch in train_batches.take(1):
    pass

print(image_batch.shape)
```

## Define and Configure the Model

As with other strategies, setting up the model requires minimal code changes. Let's first define a utility function to build and compile the model.


```python
# tells if we want to freeze the layer weights of our feature extractor during training
do_fine_tuning = False
```


```python
def build_and_compile_model():
    print("Building model with", MODULE_HANDLE)

    # configures the feature extractor fetched from TF Hub
    feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                   input_shape=IMAGE_SIZE + (3,),
                                   trainable=do_fine_tuning)

    # define the model
    model = tf.keras.Sequential([
      feature_extractor,
      # append a dense with softmax for the number of classes
      tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # display summary
    model.summary()

    # configure the optimizer, loss and metrics
    optimizer = tf.keras.optimizers.SGD(lr=0.002, momentum=0.9) if do_fine_tuning else 'adam'
    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model
```

You can now call the function under the strategy scope. This places variables and computations on the device you specified earlier.


```python
# build and compile under the strategy scope
with one_strategy.scope():
    model = build_and_compile_model()
```

`model.fit()` can be run as usual.


```python
EPOCHS = 5
hist = model.fit(train_batches,
                 epochs=EPOCHS,
                 validation_data=validation_batches)
```

Once everything is working correctly, you can switch to a different device or a different strategy that distributes to multiple devices.
