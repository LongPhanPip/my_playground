# Random generator

## Random integer
```python
    import numpy as np
    np.random.randint(1, 10, size=(3,4))
    # array([[7, 3, 4, 7],
    #        [1, 1, 7, 1],
    #        [5, 3, 8, 2]])
```

## Random 0-1
```python
    np.random.rand(3, 4)
    # array([[0.75278089, 0.98443686, 0.62505059, 0.34584865],
    #        [0.73855392, 0.33858326, 0.2698294 , 0.4020974 ],
    #        [0.33386053, 0.18262546, 0.07247349, 0.60780109]])

```

## Shuffle
```python
    arr = np.array([2, 3, 6, 2, 8, 3, 1])
    np.random.shuffle(arr)

    # arr -> array([8, 3, 2, 3, 1, 6, 2])
```
## Permutation
```python
    np.random.permuation(20)    # -> array([ 8, 17, 15, 13, 10,  9, 11, 12,  0,  1,  4,  3,  5, 18,  2,  7, 14, 19, 16,  6])

```

## Distribution
### Normal
```python
    np.random.normal(1, 0.1, size=(3,4))    # -> (mean, std, size)
    # array([[ 0.01751835, -0.08207739, -0.04763712,  0.03923665],
    #        [-0.12046535, -0.08204021,  0.06481094, -0.05330554],
    #        [-0.17912559, -0.18686909,  0.0380102 ,  0.02023223]])

```
### Uniform
```python
    np.random.uniform(0, 10, size=(3,4))
    # array([[7.18584942, 0.772325  , 9.63969616, 2.74410977],
    #        [2.79755762, 1.16944129, 1.03606607, 5.07048585],
    #        [1.76509709, 0.45632948, 5.04223083, 4.5895126 ]])
```

## Cast type
```python
    arr = np.array([1.0, 2.0, 3.0]).astype(np.int8)
    # array([1, 2, 3], dtype=int8)

```

## Indexing and Slicing
```python
    arr = np.arange(0, 12).reshape(3, 4)
    # array([[ 0,  1,  2,  3],
    #        [ 4,  5,  6,  7],
    #        [ 8,  9, 10, 11]])

    arr[(0, 1, 2), (0, 1, 2)]

    # array([ 0,  5, 10])


```
