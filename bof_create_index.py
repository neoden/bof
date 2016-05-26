# coding=utf-8
from __future__ import print_function

import glob
import sys
import os
import cv2
import numpy as np
import pickle


IMAGES_DIR = '/home/xtal/Code/flatsearch/var/images/'
TEMP_DIR = '/home/xtal/'


def describe(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 4)

def main():
    dict_path = TEMP_DIR + sys.argv[1]
    dictionary = pickle.load(open(dict_path, 'rb'))

    index = {}

    matcher = cv2.FlannBasedMatcher(flann_params, {}) # блять
    detector = cv2.SIFT()
    extractor = cv2.DescriptorExtractor_create('SIFT')
    bowDE = cv2.BOWImgDescriptorExtractor(extractor, matcher)
    bowDE.setVocabulary(dictionary)

    paths = [path for path in glob.iglob(IMAGES_DIR + '/*.jpeg') if path.split('-')[1] == '2']
    for n, path in enumerate(paths):
        print(n)
        #print('Percent completed: {}\r'.format(int(n / len(paths) * 100)), end='')
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        kp = detector.detect(gray, None)
        des = bowDE.compute(gray, kp, None)

        index[path] = des

    index_path = TEMP_DIR + sys.argv[2]
    pickle.dump(index, open(index_path, 'wb'))


if __name__ == '__main__':
    main()