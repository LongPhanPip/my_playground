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

    # subset = 'training' | 'validation' | 'testing'

```

## Executing
### Tensor
#### Attribute
```python
    arr = tf.constant([[1,2,3,4], [5,6,7,8]])

    #   dtype
    arr.dtype # -> tf.int32

    #   ndim
    arr.ndim  # -> 2

    #   size
    tf.size(arr).numpy() # -> 8

```

#### Slicing & Indexing
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

#### Reshape
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
#### Cast type
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
