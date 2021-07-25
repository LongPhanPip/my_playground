import concurrent.futures
import time
import os
import re
from multiprocessing import Queue
import multiprocessing as mp
import math
from collections import Counter

with open('notebook/rus-eng/rus.txt', 'r') as f:
    corpus = f.readlines()

def tokenize(sentence):
    return re.findall(r"[\w']+|[.,!?;]", sentence)

def get_eng_rus(corpus):
    lines = [line.split('\t') for line in corpus]
    eng = [tokenize(line[0].lower()) for line in lines]
    rus = [tokenize(line[1].lower()) for line in lines]
    return vocab(eng), vocab(rus)


def vocab(source, min_feq=2, reserve_token=['<sos>', '<pad>', '<eos>']):
    counter = Counter([word for sent in source for word in sent ])
    return counter


def main():
    seq = [i for i in range(1, 100000)]
    start = time.time()


    # pool = mp.Pool(processes=8)
    # pool.map(mul, seq)



    # for result in results:
    #     print(result)

    # raw_text = list(results)
    # print(raw_text)

    print(get_eng_rus(corpus[:10]))

    print(f'Processing time {time.time() - start}')


if __name__ == '__main__':
    main()
