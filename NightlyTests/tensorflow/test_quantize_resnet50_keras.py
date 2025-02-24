# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

import pytest
import numpy as np
import tensorflow as tf
from packaging import version

from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper
from aimet_tensorflow.keras.utils.common import parse_activation_layer

def disable_input_quantizers(qsim: QuantizationSimModel):
    # Only use this while quantsim config is not implemented yet.
    for wrapper in qsim.quant_wrappers():
        for input_q in wrapper.input_quantizers:
            input_q.disable()

def disable_conv_and_dense_bias_quantizers(qsim: QuantizationSimModel):
    # Only use this while quantsim config is not implemented yet.
    for wrapper in qsim.quant_wrappers():
        if isinstance(wrapper._layer_to_wrap, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)) and \
                len(wrapper.param_quantizers) == 2:
            wrapper.param_quantizers[1].disable()

def disable_bn_quantization(qsim: QuantizationSimModel):
    # Only use this while quantsim config is not implemented yet.
    for wrapper in qsim.quant_wrappers():
        if isinstance(wrapper._layer_to_wrap, tf.keras.layers.BatchNormalization):
            for param_q in wrapper.param_quantizers:
                param_q.disable()
            for output_q in wrapper.output_quantizers:
                output_q.disable()

def disable_conv_relu_supergroups(qsim: QuantizationSimModel):
    # assumes bn fold has taken place. Only use this while quantsim config is not implemented yet.
    # only handles relu that comes in the form of tf.keras.layer.Activation, not tf.keras.layer.ReLU
    for wrapper in qsim.quant_wrappers():
        # Take care of conv -> relu supergroups
        if isinstance(wrapper._layer_to_wrap, tf.keras.layers.Conv2D) and \
                len(wrapper.outbound_nodes) == 1 and \
                isinstance(wrapper.outbound_nodes[0].layer._layer_to_wrap, tf.keras.layers.Activation) and \
                parse_activation_layer(wrapper.outbound_nodes[0].layer._layer_to_wrap)[0] == 'Relu':
            wrapper.output_quantizers[0].disable()
        # Take care of conv -> bn -> relu supergroups
        elif isinstance(wrapper._layer_to_wrap, tf.keras.layers.Conv2D) and \
                len(wrapper.outbound_nodes) == 1 and \
                isinstance(wrapper.outbound_nodes[0].layer._layer_to_wrap, tf.keras.layers.BatchNormalization) and \
                len(wrapper.outbound_nodes[0].layer.outbound_nodes) == 1 and \
                isinstance(wrapper.outbound_nodes[0].layer.outbound_nodes[0].layer._layer_to_wrap,
                           tf.keras.layers.Activation) and \
                parse_activation_layer(wrapper.outbound_nodes[0].layer.outbound_nodes[0].layer._layer_to_wrap)[0] == \
                'Relu':
            wrapper.output_quantizers[0].disable()
            wrapper.outbound_nodes[0].layer.output_quantizers[0].disable()

def disable_add_relu_supergroups(qsim: QuantizationSimModel):
    # only handles relu that comes in the form of tf.keras.layer.Activation, not tf.keras.layer.ReLU
    for wrapper in qsim.quant_wrappers():
        if isinstance(wrapper._layer_to_wrap, tf.keras.layers.Add) and \
                len(wrapper.outbound_nodes) == 1 and \
                isinstance(wrapper.outbound_nodes[0].layer._layer_to_wrap, tf.keras.layers.Activation) and \
                parse_activation_layer(wrapper.outbound_nodes[0].layer._layer_to_wrap)[0] == 'Relu':
            wrapper.output_quantizers[0].disable()

def test_quantize_resnet50_keras():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        rand_inp = np.random.randn(1, 224, 224, 3)
        model = tf.keras.applications.resnet50.ResNet50(weights=None)
        _, model = fold_all_batch_norms(model)
        orig_out = model(rand_inp)
        qsim = QuantizationSimModel(model)
        disable_input_quantizers(qsim)
        disable_bn_quantization(qsim)
        disable_conv_relu_supergroups(qsim)
        disable_conv_and_dense_bias_quantizers(qsim)
        disable_add_relu_supergroups(qsim)
        qsim.compute_encodings(lambda m, _: m(rand_inp), None)
        quant_out = qsim.model(rand_inp)
        assert not np.array_equal(orig_out, quant_out)
        for idx, layer in enumerate(model.layers):
            if isinstance(qsim.model.layers[idx], QcQuantizeWrapper):
                assert layer.name == qsim.model.layers[idx]._layer_to_wrap.name
        qsim.export('./data/', 'resnet50')
