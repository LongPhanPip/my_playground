from mnist import MNIST
import random
import numpy as np
import os

MNIST_DIR = 'minist_data'
TRAIN_DIR = os.path.join(MNIST_DIR, 'train')
TEST_DIR = os.path.join(MNIST_DIR, 'test')

mndata = MNIST(TRAIN_DIR)

images, labels = mndata.load_training()

index = random.randrange(0, len(images))
print(mndata.display(images[index]))
