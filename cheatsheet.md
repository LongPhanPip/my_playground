# Urllib3
# Download file
``` python
import urllib3
http = urllib3.PoolManager()
with http.request('GET', 'url', preload_content=False) as ref:
    with open(filename, 'wb') as f:
        f.write(ref.data)
```

---
# Gzip
## Extract file
``` python
import shutil
import gzip

with gzip.open('file_to_extract', 'rb') as ref:
    with open('file_to _write', 'wb') as f:
        shutil.copyfileobj(ref, f)
```
---

# Pathlib
## Path attribute
``` python
    from pathlib import Path
    path = Path('/mathplotlib/grid_spec.png')
    path.absolute()     # -> '$HOME/Desktop/playground/matplotlib/grid_spec.png'
    path.parts          # -> ('maplotlib', 'grid_spec.png')
    path.suffixes       # -> ['.png']
    list(path.parents)  # -> [PosixPath('maplotlib'), PosixPath('.')]
```

## Create dir/file
``` python
    path = Path('/numpy/examples/blabla.py')
    try:
        path.mkdir(parents=True)
    except FileNotFoundError as error:
        print({error})
```

## Check whether exist
``` python
    path.is_file()
    path.is_dir()
    path.exists()
```

## Find file
``` python
    path = Path('.')
    path.glob('*.py')   # -> find all the python file at this dir
    path.rglob('*.py')  # -> find all the python file at this dir and its subdirs
```

## Open file
``` python
    with p.open('w', encoding='utf-8') as file:
        file.read()
```

## Remove file
``` python
    path.rmdir()

```
