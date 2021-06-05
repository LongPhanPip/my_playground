import urllib3
import certifi
from pathlib import Path
import os
import shutil

CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/'
FILE_NAME = 'cifar-10-python.tar.gz'
DOWNLOAD_URL = os.path.join(CIFAR_URL, FILE_NAME)

CIFAR_DIR = 'cifar_data'
FILE_LOCATION = os.path.join(CIFAR_DIR, FILE_NAME)


http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                           ca_certs=certifi.where())


def download_file(url, filename):
    with http.request('GET', url, preload_content=False) as ref:
        with open(filename, 'wb') as f:
            f.write(ref.data)


def extract_file(filename, dir):
    shutil.unpack_archive(filename, dir, 'gztar')


if not Path(CIFAR_DIR).exists():
    os.mkdir(CIFAR_DIR)


print('----Downloading file----')
download_file(DOWNLOAD_URL, FILE_LOCATION)

print('----Extracting file----')
extract_file(FILE_LOCATION, CIFAR_DIR)

print('----Finish----')
