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


```

### Automatic differentiation
```python
    x = tf.Variable([np.pi / 3])
    with tf.GradientTape() as t:
        loss = tf.sin(x)

    grad = t.gradient(loss, x)  # -> <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.49999997], dtype=float32)>
```
