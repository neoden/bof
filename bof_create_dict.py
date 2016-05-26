from __future__ import print_function

import glob
import sys
import os
import cv2
import numpy as np
import pickle
import random


IMAGES_DIR = '/home/xtal/Code/flatsearch/var/images/'
TEMP_DIR = '/home/xtal/'


def describe(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def create_dictionary():
    vocabulary = None

    paths = [path for path in glob.iglob(IMAGES_DIR + '/*.jpeg') if path.split('-')[1] == '2']
    NUM_SAMPLES = int(len(paths) / 10)
    print('Gathering vocabulary from {} files'.format(NUM_SAMPLES))
    for n, path in enumerate(random.sample(paths, NUM_SAMPLES)):
        print('Percent completed: {}\r'.format(int(n / NUM_SAMPLES * 100)), end='')
        _, des = describe(path)
        if vocabulary is None:
            vocabulary = des
        else:
            vocabulary = np.concatenate((vocabulary, des))
    print('Descriptors gathered: {}'.format(vocabulary.shape[0]))

    dictionary_size = 200
    tc = (cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    retries = 1
    flags = cv2.KMEANS_PP_CENTERS

    print('Building dictionary', end='')
    trainer = cv2.BOWKMeansTrainer(dictionary_size, tc, retries, flags)
    dictionary = trainer.cluster(vocabulary)
    print(' completed')

    return dictionary


def main():
    dict_path = TEMP_DIR + sys.argv[1]
    dictionary = create_dictionary()
    pickle.dump(dictionary, open(dict_path, 'wb'))

if __name__ == '__main__':
    main()