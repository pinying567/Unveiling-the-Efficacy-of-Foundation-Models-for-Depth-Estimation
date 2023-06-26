#!/usr/bin/env python
# -*- coding: utf-8 -*-
#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################
#
# Helper script to convert the NYU Depth v2 dataset Matlab file into a set of
# PNG and JPEG images.
#
# See https://github.com/deeplearningais/curfil/wiki/Training-and-Prediction-with-the-NYU-Depth-v2-Dataset
# Ref: https://github.com/Adonis-galaxy/DepthCLIP/blob/main/DepthCLIP_code/datasets/extract_official_train_test_set_from_mat.py

from __future__ import print_function

import h5py
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import scipy.io
import sys
import cv2

def convert_image(i, scene, depth_raw, image, depth_dense, prefix):

    folder = prefix / scene
    dense_folder = folder / 'dense'
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(dense_folder):
        os.makedirs(dense_folder)

    img_depth = depth_raw * 1000.0
    img_depth_uint16 = img_depth.astype(np.uint16)
    dense_depth = depth_dense * 1000.0
    dense_depth_uint16 = dense_depth.astype(np.uint16)

    cv2.imwrite("%s/sync_depth_%05d.png" % (folder, i), img_depth_uint16)
    cv2.imwrite("%s/sync_depth_dense_%05d.png" % (dense_folder, i), dense_depth_uint16)
    image = image[:, :, ::-1]
    image_black_boundary = np.zeros((480, 640, 3), dtype=np.uint8)
    image_black_boundary[7:474, 7:632, :] = image[7:474, 7:632, :]
    cv2.imwrite("%s/rgb_%05d.jpg" % (folder, i), image_black_boundary)


def convert(h5_file, train_test_split, out_folder):
    # Read data mat
    print("reading", h5_file)
    h5_file = h5py.File(h5_file, "r")

    depth_raw = h5_file['rawDepths']
    depth_dense = h5_file['depths']
    images = h5_file['images']
    scenes = [u''.join(chr(c[0]) for c in h5_file[obj_ref]) for obj_ref in h5_file['sceneTypes'][0]]

    # Read train/test split
    # h5py is not able to open that file. but scipy is
    train_test = scipy.io.loadmat(train_test_split)

    test_images = set([int(x) for x in train_test["testNdxs"]])
    train_images = set([int(x) for x in train_test["trainNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))

    print("processing images")
    out_folder = Path(out_folder)
    for i, image in enumerate(tqdm(images)):
        idx = int(i) + 1
        if idx in train_images:
            train_test = "train"
        else:
            assert idx in test_images, "index %d neither found in training set nor in test set" % idx
            train_test = "test"
        prefix = out_folder / train_test

        convert_image(i, scenes[i], depth_raw[i, :, :].T, image.T, depth_dense[i,:,:].T, prefix)

