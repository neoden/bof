# coding=utf-8
from __future__ import print_function

import glob
import pprint
import sys
import os
import cv2
import numpy as np
import pickle


IMAGES_DIR = '/home/xtal/Code/flatsearch/var/images/'
TEMP_DIR = '/home/xtal/'
NUM_MATCHES = 10


FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 4)


def describe(path, dictionary):
    matcher = cv2.FlannBasedMatcher(flann_params, {}) # блять
    detector = cv2.SIFT()
    extractor = cv2.DescriptorExtractor_create('SIFT')
    bowDE = cv2.BOWImgDescriptorExtractor(extractor, matcher)
    bowDE.setVocabulary(dictionary)

    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    kp = detector.detect(gray, None)
    des = bowDE.compute(gray, kp, None)

    return des


def clear_directory(result_path):
    for path in os.listdir(result_path):
        os.unlink(os.path.join(result_path, path))


def main():
    dict_path = TEMP_DIR + sys.argv[1]
    dictionary = pickle.load(open(dict_path, 'rb'))

    index_path = TEMP_DIR + sys.argv[2]
    index = pickle.load(open(index_path, 'rb'))

    image_path = IMAGES_DIR + sys.argv[3]
    image_des = describe(image_path, dictionary)

    result_path = TEMP_DIR + sys.argv[4]
    clear_directory(result_path)

    result = []

    for path, des in index.items():
        likelyhood = cv2.compareHist(image_des, des, cv2.cv.CV_COMP_CHISQR)
        result.append((likelyhood, path))

    best_match = sorted(result, key=lambda x: x[0])[:NUM_MATCHES]
    for rank, i in enumerate(best_match):
        likelyhood, path = i
        os.symlink(path, result_path + '/{}.jpeg'.format(rank))

    pprint.pprint(best_match)


if __name__ == '__main__':
    main()