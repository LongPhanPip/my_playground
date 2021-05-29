import urllib3
from zipfile import ZipFile
import os

MINIST_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMG_FILE = 'train-images-idx3-ubyte.gz'
TRAIN_LABEL_FILE = 'train-labels-idx1-ubyte.gz'
TEST_IMG_FILE = 't10k-images-idx3-ubyte.gz'
TEST_LABEL_FILE = 't10k-labels-idx1-ubyte.gz'

TRAIN_IMG_FILENAME = 'train_img.gz'
TRAIN_LABEL_FILENAME = 'train_label.gz'
TEST_IMG_FILENAME = 'test_img.gz'
TEST_LABEL_FILENAME = 'test_label.gz'

MINST_DIR = 'minist_data'

http = urllib3.PoolManager()


def download_file(url, filename):
    with http.request('GET', url, preload_content=False) as ref:
        with open(filename, 'wb') as f:
            f.write(ref.data)


def extract_file(filename, directory):
    with zipfile.ZipFile(filename, 'r') as zip_file:
        zip_file.extractall(directory)


# create directory
if not os.path.exists(MINST_DIR):
    os.mkdir(MINST_DIR)
    os.mkdir(os.path.join(MINST_DIR, 'train'))
    os.mkdir(os.path.join(MINST_DIR, 'test'))

train_img_filename = os.path.join(MINST_DIR, 'train', TRAIN_IMG_FILENAME)
train_label_filename = os.path.join(MINST_DIR, 'train', TRAIN_LABEL_FILENAME)
test_img_filename = os.path.join(MINST_DIR, 'test', TEST_IMG_FILENAME)
test_label_filename = os.path.join(MINST_DIR, 'test', TEST_LABEL_FILENAME)

# download file
print('----Downloading train img file----')
download_file(os.path.join(MINIST_URL, TRAIN_IMG_FILE), train_img_filename)
print('----Downloading train label file----')
download_file(os.path.join(MINIST_URL, TRAIN_LABEL_FILE), train_label_filename)
print('----Downloading test img file----')
download_file(os.path.join(MINIST_URL, TEST_IMG_FILE), test_img_filename)
print('----Downloading test label file----')
download_file(os.path.join(MINIST_URL, TEST_LABEL_FILE), test_label_filename)


# extract file
print('----Extracting train img file----')
extract_file(train_img_filename, os.path.join(MINST_DIR, 'train', 'img'))
print('----Extracting train label file----')
extract_file(train_label_filename, os.path.join(MINST_DIR, 'train', 'label'))
print('----Extracting test img file----')
extract_file(test_img_filename, os.path.join(MINST_DIR, 'test', 'img'))
print('----Extracting test label file----')
extract_file(test_label_filename, os.path.join(MINST_DIR, 'test', 'label'))
