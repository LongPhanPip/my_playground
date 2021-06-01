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
