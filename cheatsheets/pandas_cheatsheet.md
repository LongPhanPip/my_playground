# I/O processing
## Read/write csv files
```python
    import pandas as pd
    # Read files
    pd.read_csv('filename')

    # Write files
    data.to_csv('filename')

```

# DataFrame
## Constuction
```python
    import numpy as np
    # From array
    data = np.array([[1,2,3,4], [5,6,7,8]])
    pd.DataFrame(data, columns=['A', 'B', 'C', 'D'], index=["first", "second"])
    #         A  B  C  D
    # first   1  2  3  4
    # second  5  6  7  8

    # From dict
    data = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
    pd.DataFrame(data)
    #    a  b
    # 0  1  5
    # 1  2  6
    # 2  3  7
    # 3  4  8
```
### Selection, Delete, Insert
```python
    data = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [5, 3, 2, 6], "d": [7,4,2,5]}
    table = pd.DataFrame(data, index=['first_line', 'second_line', 'third_line', 'fouth_line'])

    table
    #              a  b  c  d
    # first_line   1  5  5  7
    # second_line  2  6  3  4
    # third_line   3  7  2  2
    # fouth_line   4  8  6  5


    table["a"]
    # first_line     1
    # second_line    2
    # third_line     3
    # fouth_line     4
    # Name: a, dtype: int64

    table.loc['first_line']
    # b    5
    # c    5
    # d    7
    # e    0
    # Name: first_line, dtype: int64

    table.pop('a')
    table
    #              b  c  d
    # first_line   5  5  7
    # second_line  6  3  4
    # third_line   7  2  2
    # fouth_line   8  6  5

    table['e'] = [0, 0, 0, 0]
    table
    #              b  c  d  e
    # first_line   5  5  7  0
    # second_line  6  3  4  0
    # third_line   7  2  2  0
    # fouth_line   8  6  5  0

```
### Handle missing data
```python
    data = {"one": [1, 2, 3], "two": [4, np.nan, 5]}
    table = pd.DataFrame(data)
    table.isna()
    #      one    two
    # 0  False  False
    # 1  False   True
    # 2  False  False

    table.fillna(100)
    #    one    two
    # 0    1    4.0
    # 1    2  100.0
    # 2    3    5.0

    table.dropna()
    #    one  two
    # 0    1  4.0
    # 2    3  5.0


```
### Conditional index
```python
    data = {"one": [1, 2, 3], "two": [4, np.nan, 5]}
    table = pd.DataFrame(data)
    table.where(table > 2.0, 100)
    #    one    two
    # 0  100    4.0
    # 1  100  100.0
    # 2    3    5.0

    table[table > 2.0]
       # one  two
    # 0  NaN  4.0
    # 1  NaN  NaN
    # 2  3.0  5.0


```
