from mnist import MNIST
import random
import numpy as np
import os
import cv2

MNIST_DIR = 'minist_data'
TRAIN_DIR = os.path.join(MNIST_DIR, 'train')
TEST_DIR = os.path.join(MNIST_DIR, 'test')


def load_data():
    mndata = MNIST(TRAIN_DIR)
    X_train, Y_train = mndata.load_training()
    mndata = MNIST(TEST_DIR)
    X_test, Y_test = mndata.load_testing()
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)
