# Keras
## Preprocessing
### Download file
```python
    tensorflow.keras.utils.get_file('filename', 'url', untar=True)
    # untar: specify whether the file should be decompressed
    # WARNING got certificate verfication problems
```

### Load data from files
``` python
    file_list = ['files1.csv', 'file2.csv', .....]
    tf.data.Dataset.list_files(file_list, seed=42, shuffle=True)
```

### Load image from directory
``` python
    keras.preprocessing.image_dataset_from_dicrectory(
        'data_dir',
        validation_split='<percent>',
        subset='<name_supset>',
        seed=42,
        image_size=(width, height),
        batch_size=batch_size
    )
    # subset = 'training' | 'validation'
```
### Load text file from directory
```python
    keras.preprocessing.text_dataset_from_directory(
        'data_dir',
        validation_split='<percent>',
        subset='<name_supset>',
        seed=42,
        batch_size=batch_size
    )
    # subset = 'training' | 'validation'

```

### File processing
```python
    # Read file
    tf.io.read_file("filename") # -> Return string tensor
    tf.io.write_file("filename", "contents")  # content is string tensor
```

### Image processing
```python
    # Turn string tensor to unit8
    img = tf.io.decode_jpeg("contents", channels=3)   # content is string tensor
    img =tf.io.decode_png("contents", channels=3, dtype=tf.unit8)

    # Turn 3D tensor [width, height, channels] to string tensor
    tf_str = tf.io.encode_jpeg(img)   # content is string tensor
    tf_str = tf.io.encode_png(img)

```
 For more [tf.image](https://www.tensorflow.org/api_docs/python/tf/image)

### CSV processing
```python
    import pandas as pd
    import numpy as np
    data = pd.read_csv('filename.csv')
    label = data.pop('label_colunm')
    feature = data

    # create input layer
    inputs = {}
    for name, column in feature.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

        # 1 column 1 input with shape(1,)
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    # separate numeric and string
    numeric_input = {name : input for name, input in inputs.items() if input.dtype==tf.float32}

    # process numeric input
    # process string input

    # create concat layer
    concat = tf.keras.layers.Concateante()(inputs)

```

### Text processing
```python
    VOCAL_SIZE = 10000

    # birary mode [1.0, 0.0, 0.0,..., 0.0, 0.0]
    binary_layer = keras.processing.TextVectorization(
        max_tokens=VOCAL_SIZE,
        output_mode='binary'
    )

    # int mode [24.0, 531.0, 4.0, 0.0,..., 65.0, 1.0]
    MAX_SEQUENCE_LENGTH=250
    int_layer = keras.processing.TextVectorization(
        max_tokens=VOCAL_SIZE,
        output_model='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH
    )

    binary_layer.apapt('<train_ds>')    # -> to create index
    int_layer.apapt('<train_ds>')       #

    keras.preprocessing.TextLineDataset('<text_file>')  # -> Read file and seperate each line of text is one instance

    tokenizer = tf.text.UnicodeUnicodeScriptTokenizer   # -> Split word by space, punctuation character
    tokenizer.tokenize('<text_line>')


    # Create one category table
    vocab = ['a', 'b', 'c', 'hi', 'hello']
    vocab_size = len(vocab)
    num_oov_buckets = 2
    values = range(1 + num_oov_buckets, vocab_size + 1 + num_oov_buckets)   # 0 for padding
    init = tf.lookup.KeyValueTensorInitializer(keys=vocal, values=values, key_dtype=tf.string, value_dtype=tf.int64)
    vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)

    # create text processing layer
    preprocess_layer = keras.preprocessing.TextVectorization(
        max_tokens=vocab_size,
        standardize=tf_text.case_fold_utf8,
        split=tokenizer.tokenize,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH)
    preprocess_layer.set_vocabulary(vocab)


```

---

# Tensorflow
## Executing
### Eager execution and Function
```python
    tf.executing_eagerly()  # -> True

    #   Turn on eager exec for tf.Function
    tf.config.run_functions_eagerly(True)

    @tf.function
    def assign(x):
        x.assign_add(1.0)

    a = tf.Variable(1.0)
    assign(a)

    assign.get_concrete_function(tf.Variable(3.0)).graph.as_graph_def() # -> get the graph

    assign.pretty_printed_concrete_signatures()
    # assign(x)
    #     Args:
    #         x: VariableSpec(shape=(), dtype=tf.float32, name='x')
    #     Returns:
    #         NoneTensorSpec()



```

## Tensor
### Attribute
```python
    arr = tf.constant([[1,2,3,4], [5,6,7,8]])

    #   dtype
    arr.dtype # -> tf.int32

    #   ndim
    arr.ndim  # -> 2

    #   size
    tf.size(arr).numpy() # -> 8

```

### Slicing & Indexing
```python
    d = tf.constant(np.arange(30).reshape(2, 3, -1))
    # <tf.Tensor: shape=(2, 3, 5), dtype=int64, numpy=
    # array([[[ 0,  1,  2,  3,  4],
    #         [ 5,  6,  7,  8,  9],
    #         [10, 11, 12, 13, 14]],

    #        [[15, 16, 17, 18, 19],
    #         [20, 21, 22, 23, 24],
    #         [25, 26, 27, 28, 29]]])>

    d[:, :, 4]
    # <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
    # array([[ 4,  9, 14],
    #        [19, 24, 29]])>

```

### Reshape
```python
    #   Reshape
    a = tf.constant([[1,2,3,4]])    # -> <tf.Tensor: shape=(1, 4), dtype=int32, numpy=array([[1, 2, 3, 4]], dtype=int32)>
    tf.reshape(a, [2, -1])
    # <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    # array([[1, 2],
    #        [3, 4]], dtype=int32)>

    #   Flatten
    tf.reshape(a, [-1]) # -> <tf.Tensor: shape=(4,), dtype=int32, numpy=array([1, 2, 3, 4], dtype=int32)>

    #   Transpose
    tf.transpose(a, [1, 0]) # -> swap axis 0 and 1

```
### Cast type
```python
    b = tf.constant([1., 2., 3.])
    # <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1., 2., 3.], dtype=float32)>

    tf.cast(b, tf.int8)
    # <tf.Tensor: shape=(3,), dtype=int8, numpy=array([1, 2, 3], dtype=int8)>
```

### Range tensor
```python
    ragged = tf.ragged.constant([[1,2,3,4], [5,6], [7]])  # -> <tf.RaggedTensor [[1, 2, 3, 4], [5, 6], [7]]>

    ragged.shape # -> (3, None)


    tf.RaggedTensor.from_value_rowids( values=[3, 1, 4, 1, 5, 9, 2], value_rowids=[0, 0, 0, 0, 2, 2, 3])
    # <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9], [2]]>

    tf.RaggedTensor.from_row_lengths( values=[3, 1, 4, 1, 5, 9, 2], row_lengths=[4, 0, 2, 1])
    # <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9], [2]]>

    ragged = tf.ragged.constant([[1,2,3,4], [5,6], [7]])
    ragged.shape    # -> TensorShape([3, None])
    ragged.bounding_shape() # -> <tf.Tensor: shape=(2,), dtype=int64, numpy=array([3, 4])>

    ragged.to_tensor(default_value=0, shape=[3, None])
    # array([[1, 2, 3, 4],
    #       [5, 6, 0, 0],
    #       [7, 0, 0, 0]], dtype=int32)>

    ragged.to_list()    # -> [[1, 2, 3, 4], [5, 6], [7]]

    ragged.numpy()
    # array([array([1, 2, 3, 4], dtype=int32),
    #        array([5, 6], dtype=int32),
    #        array([7], dtype=int32)], dtype=object)

```

---

### Automatic differentiation
```python
    x1 = tf.Variable(2.0)
    x2 = tf.Variable(1.0)
    with tf.GradientTape() as t:

        loss = x1 * x1 + x1 * x2

    grad = t.gradient(loss, [x1, x2])
    # [<tf.Tensor: shape=(), dtype=float32, numpy=5.0>,
    #  <tf.Tensor: shape=(), dtype=float32, numpy=2.0>]


    c = tf.constant(1.0)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(c)
        loss = c * c + 3 * c


    grad = tape.gradient(loss, c)
    # <tf.Tensor: shape=(), dtype=float32, numpy=5.0>

    x = tf.constant(2.0)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(a)
        y = x * x
        z = y * y + y

    grad1 = tape.gradient(z, y)
    # <tf.Tensor: shape=(), dtype=float32, numpy=9.0>

    grad2 = tape.gradient(y, x)
    # <tf.Tensor: shape=(), dtype=float32, numpy=36.0>


    x = tf.Variable([2., 2.])
    y = tf.Variable(3.)
    with tf.GradientTape() as tape:
      z = y**2

    tape.gradient(z, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # tf.Tensor([0. 0.], shape=(2,), dtype=float32)

```

## Module
```python
    from tensorflow import keras
    model = keras.Sequential(
        keras.layers.Dense(8, input_shape=(4,))
    )

    model.variables
    # [<tf.Variable 'dense_1/kernel:0' shape=(4, 8) dtype=float32, numpy=
    # array([[ 0.5602028 , -0.45202714,  0.4786535 ,  0.32618052, -0.26827532,
    #         -0.22784147,  0.57154936,  0.37307972],
    #        [-0.5993536 ,  0.3489011 ,  0.4463678 , -0.30167195, -0.52523047,
    #         -0.3345585 , -0.49846232, -0.38535234],
    #        [ 0.17935842,  0.49219376,  0.23006076,  0.41090518,  0.33342808,
    #          0.6061254 , -0.32397506,  0.647996  ],
    #        [-0.05402601, -0.5455225 , -0.37996194,  0.09829068, -0.49169677,
    #          0.05708152,  0.14714223,  0.14184624]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(8,) dtype=float32, numpy=array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)>]

    model.trainable_variables


    mode.save('filename')   # Save to file

    tf.keras.models.loaf_model('filename')  # Load model from file

```

---
# Custom model
## Custom layer
### Layer operation
```python
    # Basic layer
    layer = tf.keras.layers.Dense(10)

    # To use layer just call it
    layer(tf.arange(10).reshape(1, -1))
    # <tf.Tensor: shape=(1, 10), dtype=float32, numpy=
    # array([[ 9.4073105,  1.5531795,  8.866978 , -6.4064355, -4.562314 ,
    #          1.3550291,  7.9261227, -4.917206 ,  2.6355855,  0.9796776]],
    #       dtype=float32)>

    layer.variables     # -> variables of layer kernel and bias
    layer.kernel, layer.bias

```
### Create custom layer
```python
    class Customlayer(tf.keras.layers.Layer):
        def __init__(self, num_outputs, **kwargs):
            super(Customlayer, self).__init__()
            self.num_outputs = num_outputs

        # Will call when first get input to build layer
        def build(self, input_shape):
            self.kernel = self.add_weight("kernel",
                                          shape=[int(input_shape[-1]), self.num_outputs],
                                          initializer="random_normal",
                                          trainable=True)
            self.bias = self.add_weight("bias",
                                        shape=num_outputs,
                                        initializer="zeros",
                                        trainable=True)

        # Call for the output
        def call(self, inputs):
            return tf.matmul(inputs, self.kernel) + self.bias

    layer = Customlayer(10)

```

## Custom block
```python
    class ResnetIdentityBlock(tf.keras.Model):
        def __init__(self, kernel_size, filters):
            super(ResnetIdentityBlock, self).__init__(name='')
            filters1, filters2, filters3 = filters

            self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
            self.bn2a = tf.keras.layers.BatchNormalization()

            self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
            self.bn2b = tf.keras.layers.BatchNormalization()

            self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
            self.bn2c = tf.keras.layers.BatchNormalization()

        def call(self, input_tensor, training=False):
            x = self.conv2a(input_tensor)
            x = self.bn2a(x, training=training)
            x = tf.nn.relu(x)

            x = self.conv2b(x)
            x = self.bn2b(x, training=training)
            x = tf.nn.relu(x)

            x = self.conv2c(x)
            x = self.bn2c(x, training=training)

            x += input_tensor
            return tf.nn.relu(x)


    block = ResnetIdentityBlock(3, [1, 2, 3])

    block.layers
    # [<tensorflow.python.keras.layers.convolutional.Conv2D at 0x1666400d0>,
    #  <tensorflow.python.keras.layers.normalization_v2.BatchNormalization at 0x166701b80>,
    #  <tensorflow.python.keras.layers.convolutional.Conv2D at 0x1666f9460>,
    #  <tensorflow.python.keras.layers.normalization_v2.BatchNormalization at 0x1666f98e0>,
    #  <tensorflow.python.keras.layers.convolutional.Conv2D at 0x1666f9c10>,
    #  <tensorflow.python.keras.layers.normalization_v2.BatchNormalization at 0x1666f9df0>]

    block.summary()
    Model: "resnet_identity_block_1"
    # _________________________________________________________________
    # Layer (type)                 Output Shape              Param #
    # =================================================================
    # conv2d_3 (Conv2D)            multiple                  4
    # _________________________________________________________________
    # batch_normalization_3 (Batch multiple                  4
    # _________________________________________________________________
    # conv2d_4 (Conv2D)            multiple                  4
    # _________________________________________________________________
    # batch_normalization_4 (Batch multiple                  8
    # _________________________________________________________________
    # conv2d_5 (Conv2D)            multiple                  9
    # _________________________________________________________________
    # batch_normalization_5 (Batch multiple                  12
    # =================================================================
    # Total params: 41
    # Trainable params: 29
    # Non-trainable params: 12

```
