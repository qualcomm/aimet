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

""" Utilities for working with ModelOptimization C++ library """
from typing import List
import tensorflow as tf
import numpy as np

from aimet_common import libpymo as pymo
from aimet_common.defs import CostMetric
from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.layer_database import Layer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class PymoSvdUtils:
    """ Utilities for working with SVD ModelOptimization C++ library """

    @classmethod
    def configure_layers_in_pymo_svd(cls, layers: List[Layer], cost_metric: CostMetric, svd_lib_ref, svd_type=pymo.TYPE_SINGLE):
        """
        Configure layers with the pymo svd library
        :param layers: List of layers to configure
        :param cost_metric: Cost metric to use
        :param svd_lib_ref: Reference to pymo instance
        :param svd_type: Type of SVD to perform Single or Successive. Defaults to Single
        :return:
        """

        # Set the pymo cost metric
        svd_lib_ref.SetCostMetric(cls._get_pymo_cost_metric(cost_metric))

        # Set up the layer attributes for each Conv2D/Linear layer
        for layer in layers:
            attr = pymo.LayerAttributes()
            attr.layerType = cls._get_pymo_layer_type(layer.module)
            attr.mode = svd_type

            conv_parameters = layer.module.get_weights()
            bias_present = False
            if len(conv_parameters) > 1:
                bias_present = True

            weight_shape = conv_parameters[0].shape
            weights = conv_parameters[0]
            output_dims = layer.module.output_shape

            if isinstance(layer.module, tf.keras.layers.Conv2D):
                # TF Conv weight order [KH,KW,ID,OD]
                # SVD expects weight matrix of the form [output_channels, input_channels, kernel_height, kernel_width]
                attr.shape = [weight_shape[3], weight_shape[2], weight_shape[0], weight_shape[1]]

                # activation shape : [height, width]
                attr.activation_dims = (output_dims[1], output_dims[2])         # (H,W)

                # CONV weights are stored in the order {H,W,I,O} in Tensorflow
                # Re-order them to the form {O,I,H,W}
                weights = np.transpose(weights, (3, 2, 0, 1))

            elif isinstance(layer.module, tf.keras.layers.Dense):
                # TF FC weight order [ID,OD], SVD expects [OD,ID]
                # weight shape : [output_channels, input_channels, 1, 1]
                attr.shape = [weight_shape[1], weight_shape[0], 1, 1]

                # activation shape : [height, width]
                attr.activation_dims = (1, 1)

                # Reorder in [OD,ID] from [ID,OD]
                weights = np.transpose(weights, (1, 0))

            else:
                # raise error if layer is not Con2D or Linear/Dense
                raise AssertionError

            params = [weights.flatten()]
            if bias_present:
                bias = conv_parameters[1]
                params.append(bias.flatten())

            attr.blobs = params

            # Save the attributes for this layer
            logger.debug("Storing attributes for layer: %s", format(layer.name))
            svd_lib_ref.StoreLayerAttributes(str(layer.name), attr)

    @staticmethod
    def _get_pymo_layer_type(module: tf.keras.layers):

        if isinstance(module, tf.keras.layers.Conv2D) and \
                not isinstance(module, (tf.keras.layers.SeparableConv2D, tf.keras.layers.DepthwiseConv2D)):
            return pymo.LAYER_TYPE_CONV

        if isinstance(module, tf.keras.layers.Dense):
            return pymo.LAYER_TYPE_FC

        raise AssertionError("Unsupported layer_type. Only Linear and Conv2D layers are currently supported")

    @staticmethod
    def _get_pymo_cost_metric(metric: CostMetric):

        if metric == CostMetric.memory:
            return pymo.COST_TYPE_MEMORY

        if metric == CostMetric.mac:
            return pymo.COST_TYPE_MAC

        raise AssertionError('Unknown cost metric')
