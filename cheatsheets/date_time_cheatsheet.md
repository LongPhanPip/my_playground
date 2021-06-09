# Time
```python
    import time

    t = time.time()
    a = 0
    for i in range(1000):
        a += 1
    interval = time.time() - t


    time.sleep(1)   # -> sleep 1s

```

# Datetime
```python
    from datetime import datetime
    now = datetime.now()

    # Time to string
    string = now.strftime("%d/%m/%Y %H:%M:%S")
    # -> '09/06/2021 15:47:22'

    # String to time
    datetime.strptime('09/12/2012 13:55:26', '%d/%m/%Y %H:%M:%S')
    # -> datetime.datetime(2012, 12, 9, 13, 55, 26)



```
