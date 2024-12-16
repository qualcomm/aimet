# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: skip-file

from decimal import Decimal
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input, decode_predictions

# imports for AIMET
import aimet_common.defs as aimet_common_defs
from aimet_tensorflow.keras.compress import ModelCompressor
import aimet_tensorflow.defs as aimet_tensorflow_defs


def get_eval_func(dataset_dir, batch_size, num_iterations=50000):
    """
    Sample Function which returns an evaluate function callback which can be
    called to evaluate a model on the provided dataset
    """
    def func_wrapper(model, iterations):
        validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=dataset_dir,
            labels='inferred',
            label_mode='categorical',
            batch_size=batch_size,
            shuffle=False,
            image_size=(224, 224))
        # If no iterations specified, set to full validation set
        if not iterations:
            iterations = num_iterations
        else:
            iterations = iterations * batch_size
        top1 = 0
        total = 0
        inp_data = None
        for (img, label) in validation_ds:
            x = preprocess_input(img)
            inp_data = x if inp_data is None else inp_data
            preds = model.predict(x, batch_size=batch_size)
            label = np.where(label)[1]
            label = [validation_ds.class_names[int(i)] for i in label]
            cnt = sum([1 for a, b in zip(label, decode_predictions(preds, top=1)) if str(a) == b[0][0]])
            top1 += cnt
            total += len(label)
            if total >= iterations:
                break

        return top1/total
    return func_wrapper


def aimet_spatial_svd(model, evaluator: aimet_common_defs.EvalFunction) -> Tuple[tf.keras.Model,
                    aimet_common_defs.CompressionStats]:
    """
    Compresses the model using AIMET's Keras Spatial SVD auto mode compression scheme.

    :param model: The keras model to compress
    :param evaluator: Evaluator used during compression
    :return: A tuple of compressed sess graph and its statistics
    """

    # Desired target compression ratio using Spatial SVD
    # This value denotes the desired compression % of the original model.
    # To compress the model to 20% of original model, use 0.2. This would
    # compress the model by 80%.
    # We are compressing the model by 50% here.
    target_comp_ratio = Decimal(0.5)

    # Number of compression ratio used by the API at each layer
    # API will evaluate 0.1, 0.2, ..., 0.9, 1.0 ratio (total 10 candidates)
    # at each layer
    num_comp_ratio_candidates = 10

    # Creating Greedy selection parameters:
    greedy_params = aimet_common_defs.GreedySelectionParameters(target_comp_ratio=target_comp_ratio,
                                                                num_comp_ratio_candidates=num_comp_ratio_candidates)

    # Ignoring first convolutional layer of the model for compression
    modules_to_ignore = [model.layers[2]]

    # Creating Auto mode Parameters:
    auto_params = aimet_tensorflow_defs.SpatialSvdParameters.AutoModeParams(greedy_select_params=greedy_params,
                                                                            modules_to_ignore=modules_to_ignore)

    # Creating Spatial SVD parameters with Auto Mode:
    params = aimet_tensorflow_defs.SpatialSvdParameters(input_op_names=model.inputs,
                                                        output_op_names=model.outputs,
                                                        mode=aimet_tensorflow_defs.SpatialSvdParameters.Mode.auto,
                                                        params=auto_params)

    # Scheme is Spatial SVD:
    scheme = aimet_common_defs.CompressionScheme.spatial_svd

    # Cost metric is MAC, it can be MAC or Memory
    cost_metric = aimet_common_defs.CostMetric.mac


    # Calling model compression using Spatial SVD:
    # Here evaluator is passed which is used by the API to evaluate the
    # accuracy for various compression ratio of each layer. To speed up
    # the process, only 10 batches of data is being used inside evaluator
    # (by passing eval_iterations=10) instead of running evaluation on
    # complete dataset.
    results = ModelCompressor.compress_model(model=model,
                                             eval_callback=evaluator,
                                             eval_iterations=10,
                                             compress_scheme=scheme,
                                             cost_metric=cost_metric,
                                             parameters=params)

    return results


def compress():
    """
    Example Driver Function Code in which we are compressing Resnet50 model.
    """
    dataset_dir = '/path/to/dataset'
    model = ResNet50(weights='imagenet')
    eval_func = get_eval_func(dataset_dir, batch_size=16)
    compressed_model, stats = aimet_spatial_svd(model=model, evaluator=eval_func)
    print(stats)

