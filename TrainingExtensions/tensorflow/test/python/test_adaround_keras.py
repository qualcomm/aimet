# -*- mode: python -*-
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

""" AdaRound Weights for Keras Unit Test Cases """
import json
import os
import numpy as np
import tensorflow as tf

import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme
from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.keras.adaround.adaround_loss import AdaroundHyperParameters
from aimet_tensorflow.keras.adaround.adaround_wrapper import AdaroundWrapper
from aimet_tensorflow.keras.adaround.adaround_optimizer import AdaroundOptimizer
from aimet_tensorflow.keras.adaround_weight import Adaround, AdaroundParameters
from aimet_tensorflow.keras.adaround.activation_sampler import ActivationSampler


def depthwise_conv2d_model():
    inputs = tf.keras.Input(shape=(10, 10, 3,))
    x = tf.keras.layers.Conv2D(16, (1, 1))(inputs)
    x = tf.keras.layers.SeparableConv2D(10, (2, 2))(x)
    x = tf.keras.layers.DepthwiseConv2D(3, (1, 1))(x)
    x = tf.keras.layers.Conv2D(8, (1, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="depthwise_conv2d_model")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def varied_activations_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (2, 2), input_shape=(16, 16, 3,)),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Conv2D(4, (2, 2), activation=tf.nn.tanh, kernel_regularizer=tf.keras.regularizers.l2(0.5)),
        tf.keras.layers.Conv2D(2, (2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='softmax', name="keras_model")])
    return model


# Adaround weight tests
def test_apply_adaround():
    input_data = np.random.rand(32, 16, 16, 3)
    input_data = input_data.astype(dtype=np.float64)
    batch_size = 2
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)

    model = keras_model()
    params = AdaroundParameters(data_set=dataset, num_batches=2, default_num_iterations=10)

    _ = Adaround.apply_adaround(model, params, path='./data/', filename_prefix='dummy')

    # Test export functionality
    with open('./data/dummy.encodings') as json_file:
        encoding_data = json.load(json_file)

    param_keys = list(encoding_data.keys())

    assert param_keys[0] == "conv2d/kernel:0"
    assert isinstance(encoding_data["conv2d/kernel:0"], list)
    param_encoding_keys = encoding_data["conv2d/kernel:0"][0].keys()
    assert "offset" in param_encoding_keys
    assert "scale" in param_encoding_keys

    # Delete encodings file
    if os.path.exists("./data/dummy.encodings"):
        os.remove("./data/dummy.encodings")


def test_get_module_act_func_pair():
    model = varied_activations_model()
    module_act_func_pairs = Adaround._get_module_act_func_pair(model)
    conv_layer_1 = model.layers[0]
    conv_layer_2 = model.layers[2]
    conv_layer_3 = model.layers[3]
    dense_layer = model.layers[5]
    assert module_act_func_pairs[conv_layer_1] == model.layers[1]
    assert module_act_func_pairs[conv_layer_2] is None
    assert module_act_func_pairs[conv_layer_3] is None
    assert module_act_func_pairs[dense_layer] is None


def test_get_ordered_adaround_layer_indices():
    model = varied_activations_model()
    ordered_adaround_layer_indices = Adaround._get_ordered_adaround_layer_indices(model)
    assert len(ordered_adaround_layer_indices) == 4
    assert ordered_adaround_layer_indices == [0, 2, 3, 5]


# Adaround activation sampler tests
def test_activation_sampler():
    input_data = np.random.rand(32, 16, 16, 3)
    batch_size = 2
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)

    model = keras_model()
    conv_op = model.layers[5]

    activation_sampler = ActivationSampler(dataset, num_batches=16)
    inp_data, out_data = activation_sampler.sample_activation(conv_op, model, conv_op, model)
    assert inp_data.shape == (32, 3, 3, 8)
    assert out_data.shape == (32, 2, 2, 4)

    # Test that passing in more batches than exists causes Keras dataset to still use max number of possible batches
    activation_sampler = ActivationSampler(dataset, num_batches=32)
    inp_data, out_data = activation_sampler.sample_activation(conv_op, model, conv_op, model)
    assert inp_data.shape == (32, 3, 3, 8)
    assert out_data.shape == (32, 2, 2, 4)

    activation_sampler = ActivationSampler(dataset, num_batches=2)
    inp_data, out_data = activation_sampler.sample_activation(conv_op, model, conv_op, model)
    assert inp_data.shape == (4, 3, 3, 8)
    assert out_data.shape == (4, 2, 2, 4)


# Adaround optimizer tests
def test_optimize_rounding_conv2d():
    """ Test optimize rounding for Conv2d """
    model = depthwise_conv2d_model()
    conv = model.layers[1]
    orig_weight = conv.get_weights()[0]

    opt_params = AdaroundHyperParameters(num_iterations=1, reg_param=0.01, beta_range=(20, 2), warm_start=0.2)
    conv_wrapper = AdaroundWrapper(conv, 4, QuantScheme.post_training_tf, False, False, True, False, None, None, None)

    inp_data = np.random.rand(1, 10, 10, 3).astype('float32')
    out_data = np.random.rand(1, 10, 10, 16).astype('float32')
    hard_rounded_weight, soft_rounded_weight = AdaroundOptimizer().optimize_rounding(conv_wrapper, tf.nn.relu,
                                                                                     inp_data, out_data,
                                                                                     opt_params)

    assert orig_weight.shape == hard_rounded_weight.shape
    assert np.allclose(orig_weight, hard_rounded_weight, atol=2 * conv_wrapper.encoding.delta)

    assert orig_weight.shape == soft_rounded_weight.shape
    assert np.allclose(orig_weight, soft_rounded_weight, atol=2 * conv_wrapper.encoding.delta)


def test_optimize_rounding_matmul():
    """ Test optimize rounding for MatMul """
    model = depthwise_conv2d_model()
    matmul = model.layers[6]
    orig_weight = matmul.get_weights()[0]

    opt_params = AdaroundHyperParameters(num_iterations=1, reg_param=0.01, beta_range=(20, 2), warm_start=0.2)
    matmul_wrapper = AdaroundWrapper(matmul, 4, QuantScheme.post_training_tf,
                                     False, False, True, False, None, None, None)

    inp_data = np.random.rand(1, 392).astype('float32')
    out_data = np.random.rand(1, 10).astype('float32')
    hard_rounded_weight, soft_rounded_weight = AdaroundOptimizer().optimize_rounding(matmul_wrapper, tf.nn.relu,
                                                                                     inp_data, out_data,
                                                                                     opt_params)

    assert orig_weight.shape == hard_rounded_weight.shape
    assert np.allclose(orig_weight, hard_rounded_weight, atol=2 * matmul_wrapper.encoding.delta)

    assert orig_weight.shape == soft_rounded_weight.shape
    assert np.allclose(orig_weight, soft_rounded_weight, atol=2 * matmul_wrapper.encoding.delta)


def test_optimize_rounding_depthwise_conv2d():
    """ Test optimize rounding for Depthwise Conv2d """
    model = depthwise_conv2d_model()
    depthwise_conv2d = model.layers[3]
    orig_weight = depthwise_conv2d.get_weights()[0]

    opt_params = AdaroundHyperParameters(num_iterations=1, reg_param=0.01, beta_range=(20, 2), warm_start=0.2)
    depthwise_conv_wrapper = AdaroundWrapper(depthwise_conv2d, 4, QuantScheme.post_training_tf,
                                             False, False, True, False, None, None, None)

    inp_data = np.random.rand(1, 5, 5, 10).astype('float32')
    out_data = np.random.rand(1, 3, 3, 10).astype('float32')
    hard_rounded_weight, soft_rounded_weight = AdaroundOptimizer().optimize_rounding(depthwise_conv_wrapper,
                                                                                     tf.nn.relu, inp_data,
                                                                                     out_data, opt_params)

    assert orig_weight.shape == hard_rounded_weight.shape
    assert np.allclose(orig_weight, hard_rounded_weight, atol=2 * depthwise_conv_wrapper.encoding.delta)

    assert orig_weight.shape == soft_rounded_weight.shape
    assert np.allclose(orig_weight, soft_rounded_weight, atol=2 * depthwise_conv_wrapper.encoding.delta)


def test_compute_output_with_adarounded_weights():
    """ Test compute output with adarounded weights for Conv layer """
    np.random.seed(0)
    # tf.compat.v1.set_random_seed(0)
    quant_scheme = QuantScheme.post_training_tf_enhanced
    weight_bw = 8

    # Create weight data in common format then convert into tensorflow format
    weight_data = np.random.rand(4, 4, 1, 1).astype(dtype='float32')
    weight_data = np.transpose(weight_data, (2, 3, 1, 0))
    weight_tensor = tf.convert_to_tensor(weight_data, dtype=tf.float32)

    inp_data = np.random.rand(1, 4, 10, 10).astype(dtype='float32')
    inp_data_t = np.transpose(inp_data, (0, 2, 3, 1))
    inp_tensor = tf.convert_to_tensor(inp_data_t, dtype=tf.float32)
    out_data = np.random.rand(1, 4, 10, 10).astype(dtype='float32')
    out_data_t = np.transpose(out_data, (0, 2, 3, 1))
    out_tensor = tf.convert_to_tensor(out_data_t, dtype=tf.float32)

    _ = tf.nn.conv2d(inp_tensor, weight_tensor, strides=[1, 1, 1, 1], padding='SAME',
                     data_format="NHWC", name='Conv2D')
    conv_layer = tf.keras.layers.Conv2D(4, 1, padding='same')
    conv_layer.build(input_shape=(1, 10, 10, 4))
    conv_layer.set_weights([weight_tensor] + conv_layer.get_weights()[1:])
    conv_wrapper = AdaroundWrapper(conv_layer, weight_bw, quant_scheme, False, False, True, False, None, None, None)

    hard_recons_error, soft_recons_error = AdaroundOptimizer._eval_recons_err_metrics(conv_wrapper, None, inp_tensor,
                                                                                      out_tensor)
    assert np.isclose(hard_recons_error, 0.61097431182861, atol=1e-4)
    assert np.isclose(soft_recons_error, 0.61089688539505, atol=1e-4)


# Adaround wrapper tests
def test_get_conv_args():
    layer = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(2, 2), padding='valid',
                                            data_format='channels_first', dilation_rate=(1, 1))
    conv_args = AdaroundWrapper._get_conv_args(layer)
    assert conv_args['padding'] == 'VALID'
    assert conv_args['data_format'] == 'NCHW'
    assert conv_args['strides'] == [1, 1, 2, 2]
    assert conv_args['dilations'] == (1, 1)

    layer = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same',
                                            data_format='channels_last', dilation_rate=1)
    conv_args = AdaroundWrapper._get_conv_args(layer)
    assert conv_args['padding'] == 'SAME'
    assert conv_args['data_format'] == 'NHWC'
    assert conv_args['strides'] == [1, 2, 2, 1]
    assert conv_args['dilations'] == (1, 1)

    layer = tf.keras.layers.DepthwiseConv2D(kernel_size=3)
    conv_args = AdaroundWrapper._get_conv_args(layer)
    assert conv_args['padding'] == 'VALID'
    assert conv_args['data_format'] == 'NHWC'
    assert conv_args['strides'] == [1, 1, 1, 1]
    assert conv_args['dilations'] == (1, 1)


def test_calculate_alpha():
    np.random.seed(0)
    weight = np.random.rand(32, 3, 12, 12)
    encoding = libpymo.TfEncoding()
    encoding.bw = 4
    encoding.offset = -127.0
    encoding.delta = 0.001551126479

    alpha = AdaroundWrapper._calculate_alpha(weight, encoding, enable_per_channel=False, ch_axis=None)
    assert np.isclose(alpha[0, 0, :1, :1], 1.1715, atol=1e-4)


def test_adaround_weights():
    """ test adaround weights """
    model = depthwise_conv2d_model()

    # 1) Conv2D
    conv = model.layers[1]
    conv_wrapper = AdaroundWrapper(conv, 4, QuantScheme.post_training_tf, False, False, True, False, None, None, None)

    # 2) MatMul
    matmul = model.layers[6]
    matmul_wrapper = AdaroundWrapper(matmul, 4, QuantScheme.post_training_tf,
                                     False, False, True, False, None, None, None)

    # 3) Depthwise Conv2D
    depthwise_conv = model.layers[3]
    depthwise_conv_wrapper = AdaroundWrapper(depthwise_conv, 4, QuantScheme.post_training_tf,
                                             False, False, True, False, None, None, None)

    matmul_wrapper.use_soft_rounding.assign(False)
    quantized_weight = matmul_wrapper.adaround_weights()
    orig_weight = matmul_wrapper._weight_tensor
    assert orig_weight.shape == quantized_weight.shape
    assert np.allclose(orig_weight, quantized_weight, atol=2 * matmul_wrapper.encoding.delta)

    matmul_wrapper.use_soft_rounding.assign(True)
    quantized_weight = matmul_wrapper.adaround_weights()
    orig_weight = matmul_wrapper._weight_tensor
    assert orig_weight.shape == quantized_weight.shape
    assert np.allclose(orig_weight, quantized_weight, atol=2 * matmul_wrapper.encoding.delta)

    conv_wrapper.use_soft_rounding.assign(False)
    quantized_weight = conv_wrapper.adaround_weights()
    orig_weight = conv_wrapper._weight_tensor
    assert orig_weight.shape == quantized_weight.shape
    assert np.allclose(orig_weight, quantized_weight, atol=2 * conv_wrapper.encoding.delta)

    conv_wrapper.use_soft_rounding.assign(True)
    quantized_weight = conv_wrapper.adaround_weights()
    orig_weight = conv_wrapper._weight_tensor
    assert orig_weight.shape == quantized_weight.shape
    assert np.allclose(orig_weight, quantized_weight, atol=2 * conv_wrapper.encoding.delta)

    depthwise_conv_wrapper.use_soft_rounding.assign(False)
    quantized_weight = depthwise_conv_wrapper.adaround_weights()
    orig_weight = depthwise_conv_wrapper._weight_tensor
    assert orig_weight.shape == quantized_weight.shape
    assert np.allclose(orig_weight, quantized_weight, atol=2 * depthwise_conv_wrapper.encoding.delta)

    depthwise_conv_wrapper.use_soft_rounding.assign(True)
    quantized_weight = depthwise_conv_wrapper.adaround_weights()
    orig_weight = depthwise_conv_wrapper._weight_tensor
    assert orig_weight.shape == quantized_weight.shape
    assert np.allclose(orig_weight, quantized_weight, atol=2 * depthwise_conv_wrapper.encoding.delta)


def test_apply_adaround_per_channel_conv2d_transpose():
    """ test adaround apply specific to Conv2DTranspose"""
    inputs = tf.keras.Input(shape=(16, 16, 3))
    outputs = tf.keras.layers.Conv2DTranspose(8, (2, 2))(inputs)
    dataset_size = 32
    batch_size = 16
    possible_batches = dataset_size // batch_size
    input_data = np.random.rand(dataset_size, 16, 16, 3).astype(dtype=np.float64)

    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)

    params = AdaroundParameters(data_set=dataset, num_batches=possible_batches, default_num_iterations=10)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    quantsim_config = {
        "defaults": {
            "ops": {},
            "params": {
                "is_symmetric": "True"
            },
            "per_channel_quantization": "True",
        },
        "params": {},
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {
            "is_output_quantized": "True"
        }
    }

    with open('./config.json', 'w') as f:
        json.dump(quantsim_config, f)

    Adaround.apply_adaround(
        model, params, path='./', filename_prefix='conv2d_transpose',
        default_param_bw=8, default_quant_scheme=QuantScheme.post_training_tf,
        config_file='config.json',
    )

    with open('./conv2d_transpose.encodings') as json_file:
        encoding_data = json.load(json_file)

    param_keys = list(encoding_data.keys())

    assert(param_keys[0] == "conv2d_transpose/kernel:0")
    conv_transpose_encoding_data = encoding_data['conv2d_transpose/kernel:0']
    assert (isinstance(conv_transpose_encoding_data, list))
    assert len(conv_transpose_encoding_data) == 8


def test_adaround_layer_selection():
    """To test if the desired layers are selected for adaround or not"""
    model = depthwise_conv2d_model()

    layer_indices = Adaround._get_ordered_adaround_layer_indices(model)

    assert len(layer_indices) == 4
    assert [1, 3, 4, 6] == layer_indices
