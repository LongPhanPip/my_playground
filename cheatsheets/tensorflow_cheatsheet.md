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
