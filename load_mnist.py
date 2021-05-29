import cv2
import numpy as np
import os

MINST_DIR = 'minist_data'
TRAIN_DIR = os.path.join(MINST_DIR, 'train')
TEST_DIR = os.path.join(MINST_DIR, 'test')


def load_data(dir):
    for root, dirs, files in os.walk(dir):
        img = cv2.imread('messi5.jpg')
        print(img)


load_data(os.path.join(MINST_DIR, TRAIN_DIR, 'img'))
