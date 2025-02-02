# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Contains all the dependencies for the Acceptance tests.
    This file (dependencies.py) is executed from CMakeLists.txt """

import os
import sys
import shutil
from decimal import Decimal, getcontext
import zipfile
import csv
import pickle
import wget
import torch
from transformers import BertTokenizer
from models import mnist_torch_model
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)

eval_scores_csv_file = sys.argv[1]
use_cuda = eval(sys.argv[2])

# ##############################################################################################
# Dependency #1
#
# Some Acceptance tests depend on a trained MNIST model.
#
# Check if we need to generate the .pth for CPU or GPU. If not, return
cpu_output_files = os.path.join('./', 'data', 'mnist_trained_on_CPU.pth')
gpu_output_files = os.path.join('./', 'data', 'mnist_trained_on_GPU.pth')

if (not use_cuda and os.path.isfile(cpu_output_files)) or (use_cuda and os.path.isfile(gpu_output_files)):
    logger.info('Mnist model .pth generation not needed')
else:
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    if use_cuda:
        model = mnist_torch_model.Net().to("cuda")
    else:
        model = mnist_torch_model.Net().to("cpu")
    mnist_torch_model.train(model, epochs=1, use_cuda=use_cuda, batch_size=50, batch_callback=None)

    # create directory
    if not os.path.isdir('./data'):
        os.mkdir('./data')

    if use_cuda:
        torch.save(model, gpu_output_files)

    model = model.to("cpu")
    torch.save(model, cpu_output_files)


# ##############################################################################################
# Dependency #2
#
# Multiple acceptance tests depend on Imagenet data.
# The Tiny Imagenet Data is used for those acceptance tests.
#
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


def create_smaller_tiny_imagenet_dataset(num_images: int):
    """ From the Tiny Imagenet dataset, copy only 2 classes and num_images per class in a separate directory

    :param num_images: number of images copied per class
    :return:
    """
    tiny_200_train_dir = './data/tiny-imagenet-200/train'
    tiny_200_val_dir = './data/tiny-imagenet-200/val'
    tiny_2_dir = './data/tiny-imagenet-2'
    tiny_2_train_dir = './data/tiny-imagenet-2/train'
    tiny_2_val_dir = './data/tiny-imagenet-2/val'
    two_class_dirs_list = ['n03584254', 'n02403003']  # Out of 200 classes, selected these two

    if not os.path.exists(tiny_2_dir):
        os.mkdir(tiny_2_dir)
        logger.info("Creating smaller dataset from Tiny Imagenet dataset")

        for dir_name in two_class_dirs_list:

            # Create destination directories for copying Training and Validation images.
            train_src_two_class_dir = os.path.join(tiny_200_train_dir, dir_name, 'images')
            train_dest_two_class_dir = os.path.join(tiny_2_train_dir, dir_name)
            val_src_two_class_dir = os.path.join(tiny_200_val_dir, 'images')
            val_dest_two_class_dir = os.path.join(tiny_2_val_dir, dir_name)
            if not os.path.exists(train_dest_two_class_dir):
                if not os.path.exists(tiny_2_train_dir):
                    os.mkdir(tiny_2_train_dir)
                os.mkdir(train_dest_two_class_dir)
            if not os.path.exists(val_dest_two_class_dir):
                if not os.path.exists(tiny_2_val_dir):
                    os.mkdir(tiny_2_val_dir)
                os.mkdir(val_dest_two_class_dir)

            # Copy the Training and Validation images.
            train_fnames = os.listdir(train_src_two_class_dir)
            val_fnames = os.listdir(val_src_two_class_dir)
            for i in range(num_images):
                shutil.copy(os.path.join(train_src_two_class_dir, train_fnames[i]), train_dest_two_class_dir)
                shutil.copy(os.path.join(val_src_two_class_dir, val_fnames[i]), val_dest_two_class_dir)


tiny_200_dir = './data/tiny-imagenet-200'
tiny_2_num_images_per_class = 10

if not os.path.exists(tiny_200_dir):
    download_and_extract_tiny_imagenet_data()
    create_smaller_tiny_imagenet_dataset(tiny_2_num_images_per_class)
else:
    logger.info("Tiny Imagenet Data exists. No need to download")

# ##############################################################################################
# Dependency #3
#
# Multiple compression acceptance tests depend on a pre-made pickle file containing eval scores for ResNet18.
# The eval score pickle file ic created from the CSV file passed in as argv[1] when this file (dependencies.py)
# is executed from the CMakeLists.txt file.
# The following configuration is used in those test cases.
# Model = Resnet18
# Number of Compression Ratio candidates = 10
# Ignored layers = model.conv1,
#                  model.layer2[0].downsample[0],
#                  model.layer3[0].downsample[0],
#                  model.layer4[0].downsample[0],
#                  model.layer4[1].conv1,
#                  model.layer4[1].conv2
#

eval_score_pkl_file = os.path.join('./', 'data', 'resnet18_eval_scores.pkl')
if not os.path.isfile(eval_score_pkl_file):
    logger.info('Generating Resnet18 eval score pickle file ')
    model_eval_scores_dict = {}
    layer_eval_scores_dict = {}
    getcontext().prec = 1
    comp_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
    logger.info('Resnet18 eval score pickle file exists. No need to generate it.')

huggingface_dir = os.path.join('./', 'data', 'huggingface')
if not os.path.exists(huggingface_dir):
    os.makedirs(huggingface_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.save_pretrained('{}/bert-base-uncased'.format(huggingface_dir))
else:
    logger.info("Huggingface data exists. No need to download")
