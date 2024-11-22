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

""" This module contains utilities to capture and save intermediate layer-outputs of a model. """

import os
from typing import Union, List, Tuple
import re
from collections import OrderedDict
import json
import numpy as np
import tensorflow as tf
from aimet_tensorflow.keras.quantsim import QcQuantizeWrapper, QcQuantizableMultiHeadAttention
from aimet_common.layer_output_utils import SaveInputOutput
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LayerOutputs)

class LayerOutputUtil:
    """ Implementation to capture and save outputs of intermediate layers of a model (fp32/quantsim) """

    def __init__(self, model: tf.keras.Model, save_dir: str = "./KerasLayerOutput"):
        """
        Constructor for LayerOutputUtil.

        :param model: Keras (fp32/quantsim) model.
        :param save_dir: Directory to save the layer outputs.
        """
        # Freeze the model weights and state
        model.trainable = False

        # Get intermediate model for layer-outputs
        self.intermediate_model = self._get_intermediate_model(model)

        # Get actual Layer output name to modified layer output name dict
        self.original_name_to_modified_name_mapper = self._get_original_name_to_modified_name_mapper(model)

        # Saving the actual layer output name to modified layer output name (valid file name to save) in a json file
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "LayerOutputNameMapper.json"), 'w', encoding='utf-8') as fp:
            json.dump(self.original_name_to_modified_name_mapper, fp=fp, indent=4)

        # Identify the axis-layout used for representing an image tensor
        axis_layout = 'NHWC' if tf.keras.backend.image_data_format() == 'channels_last' else 'NCHW'

        # Utility to save model inputs and their corresponding layer-outputs
        self.save_inp_out_obj = SaveInputOutput(save_dir, axis_layout=axis_layout)

    @classmethod
    def _get_layer_output_name(cls, layer: Union[QcQuantizeWrapper, QcQuantizableMultiHeadAttention, tf.keras.layers.Layer]):
        """
        This function returns the actual layer output name for a given layer
        :param layer: Keras model layer.
        :return: Actual layer output name for the layer
        """
        if isinstance(layer, QcQuantizeWrapper):
            return layer.original_layer.output.name
        return layer.output.name

    @classmethod
    def _get_intermediate_model(cls, model: tf.keras.Model):
        """
        This function instantiates the feature extraction model for per layer outputs
        :param model: Keras model.
        :return: Intermediate keras model for feature extraction
        """
        outputs = [layer.output for layer in model.layers]
        intermediate_model = tf.keras.models.Model(inputs=model.inputs, outputs=outputs)
        intermediate_model.trainable = False
        return intermediate_model

    @classmethod
    def _get_original_name_to_modified_name_mapper(cls, model: tf.keras.Model):
        """
        This function captures the per-layer output name and modifies it to make a valid file name
        (by removing non-word characters) so that the layer output can be easily saved with the modified name.
        :param model: Keras model.
        :return: Actual layer name to modified layer name dict
        """
        original_name_to_modified_name_mapper = OrderedDict()
        for layer in model.layers:
            layer_output_name = cls._get_layer_output_name(layer)

            # Replace all non-word characters with "_" to make it a valid file name for saving the results
            # For Eg.: "conv2d/BiasAdd:0" gets converted to "conv2d_BiasAdd_0"
            modified_layer_output_name = re.sub(r'\W+', "_", layer_output_name)

            original_name_to_modified_name_mapper[layer_output_name] = modified_layer_output_name

        return original_name_to_modified_name_mapper

    def get_outputs(self, input_instance: Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor]]):
        """
        This function captures layer-outputs and renames them as per the AIMET exported model.
        :param input_instance: Single input instance for which we want to obtain layer-outputs.
        :return: layer-output name to layer-output dict
        """
        # Run in inference mode
        outs = self.intermediate_model(input_instance, training=False)
        output_pred = [out.numpy() for out in outs]

        return dict(zip(self.original_name_to_modified_name_mapper.values(), output_pred))

    def generate_layer_outputs(self, input_instance: Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor]]):
        """
        This method captures output of every layer of a model & saves the inputs and corresponding layer-outputs to disk.

        :param input_instance: Single input instance for which layer output need to be generated
        :return: None
        """
        layer_output_dict = self.get_outputs(input_instance)

        # Skip constant scalar layer-outputs
        const_scalar_layer_name = []
        for layer_name, layer_output in layer_output_dict.items():
            if not isinstance(layer_output, np.ndarray):
                const_scalar_layer_name.append(layer_name)
        for layer_name in const_scalar_layer_name:
            logger.info("Skipping constant scalar output of layer %s", layer_name)
            _ = layer_output_dict.pop(layer_name)

        self.save_inp_out_obj.save(np.array(input_instance), layer_output_dict)

        logger.info("Layer outputs saved for input instance %d", self.save_inp_out_obj.input_cntr)
