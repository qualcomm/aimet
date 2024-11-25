#!/usr/bin/python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2020, 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without 
#  modification, are permitted provided that the following conditions are met:
#  
#  1. Redistributions of source code must retain the above copyright notice, 
#     this list of conditions and the following disclaimer.
#  
#  2. Redistributions in binary form must reproduce the above copyright notice, 
#     this list of conditions and the following disclaimer in the documentation 
#     and/or other materials provided with the distribution.
#  
#  3. Neither the name of the copyright holder nor the names of its contributors 
#     may be used to endorse or promote products derived from this software 
#     without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#  
#  SPDX-License-Identifier: BSD-3-Clause
#  
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" Contains the dependencies for the AcceptanceTests
    This file (dependencies.py) is executed from CMakeLists.txt """

from decimal import Decimal, getcontext
import csv
import pickle
import os
import sys
import wget
import zipfile
import tarfile
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)

"""
This function returns True if DEPENDENCY_DATA_PATH environment variable is set 
otherwise False
"""


def is_tiny_imagenet_cache_set():
    if 'DEPENDENCY_DATA_PATH' in os.environ:
        return True
    else:
        return False


# ##############################################################################################
# Dependency #1
#
# Download and extract ResNet-v2-50 model that was pre-trained on tiny-imagenet dataset.

def download_and_extract_tiny_imagenet_pre_trained_resnet_v2_50():
    """ Download pre-trained vgg16 tf_slim model and keep it under ./data directory. """

    if not os.path.isdir('./data'):
        os.mkdir('./data')

    if not os.path.isdir('./data/models'):
        os.mkdir('./data/models')

    resnet_v2_50_meta_file = os.path.join('./', 'data', 'models', 'tiny_imagenet_base_2018_06_26.ckpt.meta')
    if not os.path.isfile(resnet_v2_50_meta_file):

        resnet50_url = "http://download.tensorflow.org/models/adversarial_logit_pairing/tiny_imagenet_base_2018_06_26.ckpt.tar.gz"
        tar_gz_file = os.path.join('./', 'data/models', 'tiny_imagenet_resnet50.tar.gz')
        logger.info("Downloading tiny-imagenet pre-trained ResNet50 model")
        wget.download(resnet50_url, tar_gz_file)
        logger.info("Downloading completed")

        resnet50_tar = tarfile.open(tar_gz_file, "r:gz")
        resnet50_tar.extractall('./data/models')
        resnet50_tar.close()

        os.remove(tar_gz_file)
        logger.info("Unzipping Completed")

    else:

        logger.info("resnet_v2_50 meta files exists. No need to download.")


# ##############################################################################################
# Dependency #2
#
# Create eval_score dictionary pickle file for resnet_v2_50 model

def create_resnet50_eval_score_pickle():
    """ From a CSV file containing eval scores, create eval score  pickle file """

    eval_scores_csv_file = sys.argv[1]
    eval_score_pkl_file = os.path.join('./', 'data', 'resnet50_eval_scores.pkl')
    if not os.path.isfile(eval_score_pkl_file):
        logger.info('Generating Resnet50 eval score pickle file ')
        model_eval_scores_dict = {}
        layer_eval_scores_dict = {}
        getcontext().prec = 1
        comp_ratios = [0.25, 0.50, 0.75]
        comp_ratios_list = ["%.1f" % item for item in comp_ratios]

        with open(eval_scores_csv_file) as csv_file:
            readCSV = csv.reader(csv_file, delimiter=',')
            for row in readCSV:
                eval_scores = row[1:]
                for ratio, score in zip(comp_ratios_list, eval_scores):
                    layer_eval_scores_dict[Decimal(ratio)] = float(score)
                model_eval_scores_dict[row[0]] = layer_eval_scores_dict
                layer_eval_scores_dict = {}

        with open(eval_score_pkl_file, 'wb') as file:
            pickle.dump(model_eval_scores_dict, file)
    else:
        logger.info('Resnet50 eval score pickle file exists. No need to generate it.')


# ##############################################################################################
# Dependency #3
#
# The Tiny Imagenet Data is used in some acceptance tests.
# For TensorFlow usage, the Tiny ImageNet data is converted to TFRecords format

def download_and_extract_tiny_imagenet_data():
    """ Download tiny imagenet data and keep it under ./data directory. """

    tiny_imagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    tiny_zip_file = os.path.join('./', 'data', 'tiny.zip')

    """ 
      Variable to track if Cache Reading is configured
      if tiny_imagenet_from_cache is set tiny.zip is not deleted
      because we have to reuse it next time 
    """
    tiny_imagenet_from_cache = False

    """
      DEPENDENCY_DATA_PATH is an environment variable configured by the user,
      in case, user wants to reuse the tiny imagenet data rather than
      downloading the data again.
    """

    if 'DEPENDENCY_DATA_PATH' in os.environ:
        tiny_imagenet_from_cache = True
        tiny_zip_file = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'tiny.zip')
        logger.info(" tiny imagenet Cache is set")

        """
        Usually, the path configured in DEPENDENCY_DATA_PATH environment variable
        should be created earlier manually while configuring DEPENDENCY_DATA_PATH.
        In case of user error of not creating that path - Create the path to store
        tiny imagenet in Cache.
        """

        os.makedirs(os.environ.get('DEPENDENCY_DATA_PATH'), exist_ok=True)

    """
     if DEPENDENCY_DATA_PATH is set, downloading Tiny Imagenet in cache, otherwise on locally.
    """
    if not os.path.isfile(tiny_zip_file):
        logger.info("Downloading Tiny Imagenet Data")
        wget.download(tiny_imagenet_url, tiny_zip_file)

    with zipfile.ZipFile(tiny_zip_file, "r") as zip_ref:
        logger.info("Unzipping Tiny Imagenet Data")
        zip_ref.extractall('./data')
    # No need to delete tiny.zip from Cache
    if os.path.isfile(tiny_zip_file) and (tiny_imagenet_from_cache is False):
        os.remove(tiny_zip_file)
        logger.info("Unzipping Completed")


def get_tiny_image_net_data():
    tiny_200_dir = './data/tiny-imagenet-200'
    # tiny_2_num_images_per_class = 100
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    if not os.path.exists(tiny_200_dir):
        download_and_extract_tiny_imagenet_data()
        # create_smaller_tiny_imagenet_dataset(tiny_2_num_images_per_class)
    else:
        logger.info("Tiny Imagenet Data exists. No need to download")


if __name__ == '__main__':

    download_and_extract_tiny_imagenet_pre_trained_resnet_v2_50()
    create_resnet50_eval_score_pickle()
    get_tiny_image_net_data()
