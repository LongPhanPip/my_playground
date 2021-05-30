# Numpy
## Cast type
``` python
Syntax: np.astype(np.type)
Exmaple:
    arr = np.array([1.0, 2.0, 3.0])
    arr.astype(np.uint8) -> np.array([1, 2, 3])

```

# Urllib3
## Download file
``` python
import urllib3
http = urllib3.PoolManager()
with http.request('GET', 'url', preload_content=False) as ref:
    with open(filename, 'wb') as f:
        f.write(ref.data)
```

---
# Extract file
## Gzip
``` python
import shutil
import gzip

with gzip.open('file_to_extract', 'rb') as ref:
        with open('file_to _write', 'wb') as f:
            shutil.copyfileobj(ref, f)
```
---
# OpenCV
| Function | Syntax | Example |
| --- | --- | --- |
| Show image| cv2.imshow(filepath)| cv2.imshow('finn.jpg')|
