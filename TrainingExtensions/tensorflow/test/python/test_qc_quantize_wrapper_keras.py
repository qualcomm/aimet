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
""" Unit tests for Keras qc quantize wrapper """
import random
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from packaging import version

import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_tensorflow.examples.test_models import keras_sequential_conv_net
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper, QuantizerSettings
from aimet_tensorflow.keras.quant_sim.tensor_quantizer import ParamPerChannelQuantizer
from aimet_tensorflow.keras.utils.common import get_number_of_outputs_and_axis_handling


def dense_functional():
    inp = tf.keras.layers.Input(shape=(5,))
    x = tf.keras.layers.Dense(units=2)(inp)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name="dense_functional")
    return model


def dense_sequential():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=2, input_shape=(5,)))
    model.add(tf.keras.layers.Softmax())
    return model


class DenseSubclassing(tf.keras.Model):
    def __init__(self):
        super(DenseSubclassing, self).__init__()
        self.linear1 = tf.keras.layers.Dense(units=2)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        x = self.linear1(inputs)
        x = self.softmax(x)
        return x


def test_wrapper():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        tf.keras.backend.clear_session()
        test_inp = np.array([[1.5, 2.5]])
        inp = tf.keras.layers.Input(shape=(2,))
        dense = tf.keras.layers.Dense(3,
                                      kernel_initializer=tf.initializers.Constant([[2.3, -1.4, .5], [-.6, 3.1, -.2]]),
                                      bias_initializer=tf.initializers.Constant([5.0]))
        # run forward pass on dense to generate weights
        _ = dense(test_inp)
        x = QcQuantizeWrapper(dense,
                              QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
                              QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
                              num_inputs=1)(inp)
        model = tf.keras.Model(inputs=inp, outputs=x)

        _ = model.predict(test_inp)

        # Disable input quantizer and check later that quantizer mode remains passThrough, and encoding is None
        model.layers[1].input_quantizers[0].disable()
        model.layers[1].param_quantizers[1].disable()
        model.layers[1].compute_encoding()
        assert model.layers[1].input_quantizers[0].quant_mode == 3
        assert model.layers[1].output_quantizers[0].quant_mode == 2
        assert model.layers[1].input_quantizers[0].encoding is None
        assert model.layers[1].output_quantizers[0].encoding is not None

        model.layers[1].output_quantizers[0].disable()
        param_quant_only = model.predict(test_inp)
        model.layers[1].output_quantizers[0].enable()
        param_and_output_quant = model.predict(test_inp)
        assert np.allclose(param_quant_only, np.array([[6.9411764145, 10.6735286713, 5.2558822632]]), rtol=1e-2)
        assert np.allclose(param_and_output_quant, np.array([[6.9482579231, 10.6735286713, 5.2739787102]]), rtol=1e-2)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.MeanSquaredError())
        test_out = np.random.randn(1, 3)
        for i in range(10):
            starting_weights = [tf.keras.backend.get_value(param) for param in model.layers[1]._layer_to_wrap.weights]
            model.fit(x=test_inp, y=test_out, batch_size=1)
            weights_after_fit = [tf.keras.backend.get_value(param) for param in model.layers[1]._layer_to_wrap.weights]
            for idx, weight in enumerate(starting_weights):
                assert not np.array_equal(weight, weights_after_fit[idx])
            _ = model.predict(test_inp)
            weights_after_predict = [tf.keras.backend.get_value(param) for param in
                                     model.layers[1]._layer_to_wrap.weights]
            for idx, weight in enumerate(weights_after_predict):
                assert np.array_equal(weight, weights_after_fit[idx])


def test_wrapper_settings():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        tf.keras.backend.clear_session()
        tf.random.set_seed(10)
        test_inp = np.array([[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        inp = tf.keras.layers.Input(shape=(12,))
        identity = tf.keras.layers.Lambda(lambda x: x)
        out = QcQuantizeWrapper(identity,
                                QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf_enhanced', False, False, False),
                                QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf_enhanced', False, False, False),
                                num_inputs=1)(inp)
        model = tf.keras.Model(inputs=inp, outputs=out)

        _ = model.predict(test_inp)
        model.layers[1].input_quantizers[0].compute_encoding()
        quant_out_0 = model.predict(test_inp)

        model.layers[1].input_quantizers[0].is_symmetric = True
        model.layers[1].input_quantizers[0].use_strict_symmetric = True
        _ = model.predict(test_inp)
        model.layers[1].input_quantizers[0].compute_encoding()

        # Uncomment when calculate_delta_offset correctly accounts for strict symmetric
        # assert model.layers[1].input_quantizers[0].encoding.offset == -127  # Test strict symmetric
        quant_out_1 = model.predict(test_inp)
        assert quant_out_0[0][0] != quant_out_1[0][0]  # Test that changed settings take effect

        model.layers[1].input_quantizers[0].quant_scheme = QuantScheme.post_training_tf_enhanced
        model.layers[1].input_quantizers[0].round_mode = 'stochastic'
        model.layers[1].input_quantizers[0].bitwidth = 2
        model.layers[1].input_quantizers[0].is_symmetric = False
        model.layers[1].input_quantizers[0].use_strict_symmetric = False

        _ = model.predict(test_inp)
        model.layers[1].input_quantizers[0].compute_encoding()
        quant_out = model.predict(test_inp)

        # Check that by changing bitwidth to 2, the number of distinct quant/dequant values in the output is 4
        out_values = set()
        for num in quant_out[0]:
            out_values.add(num)
        assert len(out_values) == 3 or len(out_values) == 4


def test_keras_add_layer():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        tf.keras.backend.clear_session()
        inp = tf.keras.layers.Input(shape=(2,))
        inp_2 = tf.keras.layers.Input(shape=(2,))
        x = inp + inp_2
        model = tf.keras.Model(inputs=(inp, inp_2), outputs=x, name="model_with_add")

        i1 = np.array([[-.3, .5]])
        i2 = np.array([[-.2, .8]])

        wrapped_add = QcQuantizeWrapper(model.layers[2],
                                        QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
                                        QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
                                        num_inputs=2)
        _ = wrapped_add(i1, y=i2)
        wrapped_add.compute_encoding()
        assert len(wrapped_add.input_quantizers) == 2
        assert wrapped_add.input_quantizers[0].encoding is not None
        assert wrapped_add.input_quantizers[1].encoding is not None
        assert wrapped_add.input_quantizers[0].encoding.max != wrapped_add.input_quantizers[1].encoding.max


def test_freeze_encodings():
    tf.keras.backend.clear_session()
    test_inp = np.array([[1.5, 2.5]])
    dense = tf.keras.layers.Dense(3, kernel_initializer=tf.initializers.Constant([[2.3, -1.4, .5], [-.6, 3.1, -.2]]),
                                  bias_initializer=tf.initializers.Constant([5.0]))
    # run forward pass on dense to generate weights
    _ = dense(test_inp)
    wrapper = QcQuantizeWrapper(dense,
                                QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
                                QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
                                num_inputs=1)
    wrapper(test_inp)
    wrapper.param_quantizers[0].compute_encoding()
    weight_min = wrapper.param_quantizers[0].encoding.min
    wrapper._layer_to_wrap.set_weights([np.array([[-5.5, -4.5, -3.5], [3.0, 4.0, 5.0]])] +
                                       wrapper._layer_to_wrap.get_weights()[1:])
    wrapper(test_inp)
    wrapper.param_quantizers[0].compute_encoding()
    weight_min_2 = wrapper.param_quantizers[0].encoding.min
    assert weight_min != weight_min_2
    param_name = wrapper._layer_to_wrap.weights[0].name
    wrapper.set_and_freeze_param_encoding({param_name: [{'bitwidth': 4, 'max': 30.0, 'min': 0.0, 'offset': 0,
                                                         'scale': 2.0, 'is_symmetric': 'False'}]})
    wrapper(test_inp)
    wrapper.param_quantizers[0].compute_encoding()
    weight_min_3 = wrapper.param_quantizers[0].encoding.min
    assert weight_min_3 == 0.0
    assert wrapper.param_quantizers[0]._is_encoding_frozen
    assert wrapper.param_quantizers[0].quant_mode == int(libpymo.TensorQuantizerOpMode.quantizeDequantize)


# pylint: disable=too-many-locals
def test_per_channel_qc_quantizer_conv2d():
    """ Tests Conv2D Per Channel """
    tf.keras.backend.clear_session()
    input_shape = (1, 2, 2, 2)
    inp = tf.keras.layers.Input(shape=input_shape[1:])
    test_inp = np.array(
        [[[[-0.89, 0.35],
           [-0.92, -0.97]],
          [[-0.34, 0.094],
           [-0.50, 0.411]]]]
    )

    kernel_init_constant = np.array(
        [[[[-1.4, -0.61],
           [1.59, 0.68]]]], dtype=np.float64
    )

    conv2d = tf.keras.layers.Conv2D(
        2, 1, input_shape=input_shape[1:],
        kernel_initializer=tf.initializers.Constant(
            kernel_init_constant
        ),
        bias_initializer=tf.initializers.Constant([5.0])
    )

    # run forward pass on conv2d to generate weights
    _ = conv2d(test_inp)
    x = QcQuantizeWrapper(
        conv2d,
        QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
        QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
        num_inputs=1,
        per_channel_quantization_enabled=True
    )(inp)

    model = tf.keras.Model(inputs=inp, outputs=x)
    _ = model.predict(test_inp)

    # Disable all quantizers and run model as normal
    model.layers[1].input_quantizers[0].disable()
    model.layers[1].param_quantizers[0].disable()
    model.layers[1].param_quantizers[1].disable()
    model.layers[1].output_quantizers[0].disable()

    assert model.layers[1].input_quantizers[0].encoding is None
    assert model.layers[1].param_quantizers[0].encoding is None
    assert model.layers[1].output_quantizers[0].encoding is None

    no_quant_output = model.predict(test_inp)
    assert np.allclose(no_quant_output, np.array([[[[6.8025, 5.7809],
                                                    [4.7457, 4.9016]],
                                                   [[5.6254, 5.2713],
                                                    [6.3534, 5.5844]]]]), rtol=0.01), \
        f"no_quant_output was {no_quant_output}"

    # Check per channel encodings
    model.layers[1].param_quantizers[0].enable()
    model.layers[1].param_quantizers[0].compute_encoding()

    def _get_expected_encoding_dict(ch):
        bitwidth = 8
        ch_min = np.min(kernel_init_constant[:, :, :, ch])
        ch_max = np.max(kernel_init_constant[:, :, :, ch])
        ch_delta = (ch_max - ch_min) / (2 ** bitwidth - 1)
        ch_offset = np.round(ch_min / ch_delta)
        return {'bw': bitwidth, 'min': ch_min, 'max': ch_max, 'delta': ch_delta, 'offset': ch_offset}

    per_channel_encodings_expected = [
        _get_expected_encoding_dict(0),  # channel 0
        _get_expected_encoding_dict(1),  # channel 1
    ]

    all_per_channel_encodings = model.layers[1].param_quantizers[0].encoding

    for expected, current_encoding in zip(per_channel_encodings_expected, all_per_channel_encodings):
        actual = {
            'bw': current_encoding.bw,
            'min': current_encoding.min,
            'max': current_encoding.max,
            'delta': current_encoding.delta,
            'offset': current_encoding.offset
        }

        # Make new dictionary containing key of encoding parameter with values from both
        # expected and actual that failed to match
        incorrect_encodings = {key: [expected[key], actual[key]] for key in expected
                               if not np.allclose(expected[key], actual[key], rtol=0.01)}
        assert not incorrect_encodings, f"Key pairs for expected and actual did not match. " \
                                        f"key: [expected, actual] {incorrect_encodings}"

    # Check output of model after encoding enabled
    per_channel_quant_output_only = model.predict(test_inp)
    assert np.allclose(per_channel_quant_output_only, np.array([[[[6.7999, 5.7820],
                                                                  [4.7368, 4.9056]],
                                                                 [[5.6243, 5.2718],
                                                                  [6.3530, 5.5846]]]]), rtol=0.01)


# pylint: disable=too-many-locals
def test_per_channel_qc_quantizer_conv2d_transpose():
    """ Tests Conv2DTranspose for Per Channel"""
    tf.keras.backend.clear_session()
    random.seed(0)
    tf.random.set_seed(0)
    np.random.seed(0)
    input_shape = (1, 16, 16, 3)
    inputs = tf.keras.Input(shape=input_shape[1:])
    kernel_init_constant = np.random.randn(2, 2, 2, 3)
    transposed_kernel = K.permute_dimensions(kernel_init_constant, [0, 1, 3, 2])
    conv2d_transpose = tf.keras.layers.Conv2DTranspose(
        2, (2, 2),
        input_shape=input_shape[1:],
        kernel_initializer=tf.initializers.Constant(kernel_init_constant),
        bias_initializer=tf.initializers.Constant([5.0])
    )
    input_data = np.random.rand(*input_shape).astype(dtype=np.float64)
    assert np.allclose(input_data[0, 0, 0, 0], 0.45615033221654855,
                       rtol=0.01), "Random seeds not set correctly for either TF, python.random, or numpy"
    # run forward pass on conv2d_transpose to generate weights
    _ = conv2d_transpose(input_data)
    x = QcQuantizeWrapper(
        conv2d_transpose,
        QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
        QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
        num_inputs=1,
        per_channel_quantization_enabled=True
    )(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    _ = model.predict(input_data)

    # Check per channel encodings
    model.layers[1].param_quantizers[0].compute_encoding()

    def _get_expected_encoding_dict(ch):
        bitwidth = 8
        # channel 0
        ch_min = np.min(transposed_kernel[:, :, :, ch])
        ch_max = np.max(transposed_kernel[:, :, :, ch])
        ch_delta = (ch_max - ch_min) / (2 ** bitwidth - 1)
        ch_offset = np.round(ch_min / ch_delta)
        return {'bw': bitwidth, 'min': ch_min, 'max': ch_max, 'delta': ch_delta, 'offset': ch_offset}

    per_channel_encodings_expected = [
        _get_expected_encoding_dict(0),  # channel 0
        _get_expected_encoding_dict(1),  # channel 1
    ]

    all_per_channel_encodings = model.layers[1].param_quantizers[0].encoding

    for expected, current_encoding in zip(per_channel_encodings_expected, all_per_channel_encodings):
        actual = {
            'bw': current_encoding.bw,
            'min': current_encoding.min,
            'max': current_encoding.max,
            'delta': current_encoding.delta,
            'offset': current_encoding.offset
        }

        # Make new dictionary containing key of encoding parameter with values from both
        # expected and actual that failed to match
        incorrect_encodings = {key: [expected[key], actual[key]] for key in expected
                               if not np.allclose(expected[key], actual[key], rtol=0.01)}
        assert not incorrect_encodings, f"Key pairs for expected and actual did not match. " \
                                        f"key: [expected, actual] {incorrect_encodings}"


def test_per_channel_qc_quantizer_depthwise_conv():
    """Tests DepthwiseConv for per channel """
    tf.keras.backend.clear_session()
    input_shape = (2, 3, 3, 2)
    inp = tf.keras.layers.Input(shape=input_shape[1:])
    test_inp = np.array(
        [[[[1.8811, -1.2067],
           [-0.8310, -0.1358],
           [-1.1847, -1.8356]],
          [[1.7378, -0.4557],
           [0.7994, -1.1451],
           [-1.2045, -0.8023]],
          [[1.0628, -0.1527],
           [-1.5484, -0.6253],
           [1.1102, 0.9032]]],
         [[[2.4775, -0.4758],
           [1.4877, -0.2516],
           [0.5612, 1.0585]],
          [[1.4934, 0.3120],
           [0.1401, -0.6864],
           [-0.7637, 1.0061]],
          [[-0.1268, 0.1278],
           [-0.8578, 0.4477],
           [0.7155, 0.2616]]]]
    )

    depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        3,
        depthwise_initializer=tf.initializers.Constant([5.0]),
        bias_initializer=tf.initializers.Constant([5.0])
    )

    # run forward pass on conv2d_transpose to generate weights
    _ = depthwise_conv(test_inp)
    x = QcQuantizeWrapper(
        depthwise_conv,
        QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
        QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
        num_inputs=1,
        per_channel_quantization_enabled=True
    )(inp)

    model = tf.keras.Model(inputs=inp, outputs=x)
    _ = model.predict(test_inp)

    # Disable all quantizers and run model as normal
    model.layers[1].input_quantizers[0].disable()
    model.layers[1].param_quantizers[0].disable()
    model.layers[1].param_quantizers[1].disable()
    model.layers[1].output_quantizers[0].disable()

    assert model.layers[1].input_quantizers[0].encoding is None
    assert model.layers[1].param_quantizers[0].encoding is None
    assert model.layers[1].output_quantizers[0].encoding is None

    no_quant_output = model.predict(test_inp)

    assert np.allclose(no_quant_output, np.array([[[[14.1135, -22.2799]]],
                                                  [[[30.6354, 13.9995]]]]), rtol=0.01), \
        f"no_quant_output was {no_quant_output}"

    # Check per channel encodings
    model.layers[1].param_quantizers[0].enable()
    model.layers[1].param_quantizers[0].compute_encoding()

    per_channel_encodings_expected = [
        {'bw': 8, 'min': 0.0, 'max': 5.0, 'delta': 0.01960784314, 'offset': 0},  # channel 0
        {'bw': 8, 'min': 0.0, 'max': 5.0, 'delta': 0.019607843149, 'offset': 0},  # channel 1
    ]

    all_per_channel_encodings = model.layers[1].param_quantizers[0].encoding

    for expected, current_encoding in zip(per_channel_encodings_expected, all_per_channel_encodings):
        actual = {
            'bw': current_encoding.bw,
            'min': current_encoding.min,
            'max': current_encoding.max,
            'delta': current_encoding.delta,
            'offset': current_encoding.offset
        }

        # Make new dictionary containing key of encoding parameter with values from both
        # expected and actual that failed to match
        incorrect_encodings = {key: [expected[key], actual[key]] for key in expected
                               if not np.allclose(expected[key], actual[key], rtol=0.01)}
        assert not incorrect_encodings, f"Key pairs for expected and actual did not match. " \
                                        f"key: [expected, actual] {incorrect_encodings}"

    # Check output of model after encoding enabled
    per_channel_quant_output_only = model.predict(test_inp)
    assert np.allclose(per_channel_quant_output_only, np.array([[[[14.1135, -22.2799]]],
                                                                [[[30.6354, 13.9995]]]]), rtol=0.01)


def test_per_channel_qc_quantizer_separable_conv():
    """Tests Separable for per channel """
    tf.keras.backend.clear_session()
    input_shape = (2, 3, 3, 2)
    inp = tf.keras.layers.Input(shape=input_shape[1:])
    test_inp = np.array(
        [[[[1.8811, -1.2067],
           [-0.8310, -0.1358],
           [-1.1847, -1.8356]],
          [[1.7378, -0.4557],
           [0.7994, -1.1451],
           [-1.2045, -0.8023]],
          [[1.0628, -0.1527],
           [-1.5484, -0.6253],
           [1.1102, 0.9032]]],
         [[[2.4775, -0.4758],
           [1.4877, -0.2516],
           [0.5612, 1.0585]],
          [[1.4934, 0.3120],
           [0.1401, -0.6864],
           [-0.7637, 1.0061]],
          [[-0.1268, 0.1278],
           [-0.8578, 0.4477],
           [0.7155, 0.2616]]]]
    )

    separable_conv = tf.keras.layers.SeparableConv2D(
        2,
        kernel_size=2,
        depthwise_initializer=tf.initializers.Constant([5.0]),
        bias_initializer=tf.initializers.Constant([5.0])
    )

    # run forward pass on conv2d_transpose to generate weights
    _ = separable_conv(test_inp)
    x = QcQuantizeWrapper(
        separable_conv,
        QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
        QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
        num_inputs=1,
        per_channel_quantization_enabled=True
    )(inp)

    model = tf.keras.Model(inputs=inp, outputs=x)
    _ = model.predict(test_inp)

    # Check per channel encodings
    model.layers[1].param_quantizers[0].enable()
    model.layers[1].param_quantizers[0].compute_encoding()

    per_channel_encodings_expected = [
        {'bw': 8, 'min': 0.0, 'max': 5.0, 'delta': 0.01960784314, 'offset': 0},  # channel 0
        {'bw': 8, 'min': 0.0, 'max': 5.0, 'delta': 0.019607843149, 'offset': 0},  # channel 1
    ]

    all_per_channel_encodings = model.layers[1].param_quantizers[0].encoding

    for expected, current_encoding in zip(per_channel_encodings_expected, all_per_channel_encodings):
        actual = {
            'bw': current_encoding.bw,
            'min': current_encoding.min,
            'max': current_encoding.max,
            'delta': current_encoding.delta,
            'offset': current_encoding.offset
        }

        # Make new dictionary containing key of encoding parameter with values from both
        # expected and actual that failed to match
        incorrect_encodings = {key: [expected[key], actual[key]] for key in expected
                               if not np.allclose(expected[key], actual[key], rtol=0.01)}
        assert not incorrect_encodings, f"Key pairs for expected and actual did not match. " \
                                        f"key: [expected, actual] {incorrect_encodings}"

def test_per_channel_qc_quantizer_Dense():
    """ Tests Dense Per Channel """
    tf.keras.backend.clear_session()
    input_shape = (1, 2)
    inp = tf.keras.layers.Input(shape=input_shape[1:])
    test_inp = np.random.random(input_shape)
    kernel_init_constant = np.array(
        [[-1.4, -0.61],[1.59, 0.68]],dtype=np.float64
    )  
    dense = tf.keras.layers.Dense(2, kernel_initializer=tf.initializers.Constant(kernel_init_constant))
    # run forward pass on conv2d to generate weights
    _ = dense(test_inp)
    x = QcQuantizeWrapper(
        dense,
        QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
        QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
        num_inputs=1,
        per_channel_quantization_enabled=True
    )(inp)
    model = tf.keras.Model(inputs=inp, outputs=x)
    baseline_output = model.predict(test_inp)

    # Disable all quantizers and run model as normal
    model.layers[1].input_quantizers[0].disable()
    model.layers[1].param_quantizers[0].disable()
    model.layers[1].param_quantizers[1].disable()
    model.layers[1].output_quantizers[0].disable()

    assert model.layers[1].input_quantizers[0].encoding is None
    assert model.layers[1].param_quantizers[0].encoding is None
    assert model.layers[1].output_quantizers[0].encoding is None

    no_quant_output = model.predict(test_inp)
    assert np.allclose(no_quant_output, baseline_output,atol=0.01)

    # Check per channel encodings
    model.layers[1].param_quantizers[0].enable()
    model.layers[1].param_quantizers[0].compute_encoding()
    assert model.layers[1].param_quantizers[0].encoding


    def _get_expected_encoding_dict(ch):
        bitwidth = 8
        ch_min = np.min(kernel_init_constant[:, ch])
        ch_max = np.max(kernel_init_constant[:,ch])
        ch_delta = (ch_max - ch_min) / (2 ** bitwidth - 1)
        ch_offset = np.round(ch_min / ch_delta)
        return {'bw': bitwidth, 'min': ch_min, 'max': ch_max, 'delta': ch_delta, 'offset': ch_offset}

    per_channel_encodings_expected = [
        _get_expected_encoding_dict(0),  # channel 0
        _get_expected_encoding_dict(1),  # channel 1
    ]

    all_per_channel_encodings = model.layers[1].param_quantizers[0].encoding

    for expected, current_encoding in zip(per_channel_encodings_expected, all_per_channel_encodings):
        actual = {
            'bw': current_encoding.bw,
            'min': current_encoding.min,
            'max': current_encoding.max,
            'delta': current_encoding.delta,
            'offset': current_encoding.offset
        }

        # Make new dictionary containing key of encoding parameter with values from both
        # expected and actual that failed to match
        incorrect_encodings = {key: [expected[key], actual[key]] for key in expected
                               if not np.allclose(expected[key], actual[key], rtol=0.01)}
        assert not incorrect_encodings, f"Key pairs for expected and actual did not match. " \
                                        f"key: [expected, actual] {incorrect_encodings}"


def test_bn_QcQuantizeWrapper():
    def _test_bn_correctness(model, bn_layers, dummy_inputs):
        # original mean, var, momentum
        output_false = model(dummy_inputs, training=False)
        bn_mean_false = {layer.name: layer.moving_mean.numpy() for layer in bn_layers}
        bn_var_false = {layer.name: layer.moving_variance.numpy() for layer in bn_layers}

        output_true = model(dummy_inputs, training=True)
        bn_mean_true = {layer.name: layer.moving_mean.numpy() for layer in bn_layers}
        bn_var_true = {layer.name: layer.moving_variance.numpy() for layer in bn_layers}
        assert not np.allclose(output_true, output_false)
        assert not all(np.allclose(bn_mean_false[key], bn_mean_true[key]) for key in bn_mean_true)
        assert not all(np.allclose(bn_var_false[key], bn_var_true[key]) for key in bn_var_true)

    tf.keras.backend.clear_session()
    np.random.seed(0)
    input_data = np.random.randn(1024, 32,32,3).astype(np.float32)
    batch_size = 4
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)
    it = iter(dataset)
    dummy_inputs = next(it)

    inputs = tf.keras.Input(shape=(32, 32, 3,))
    bn = tf.keras.layers.BatchNormalization(fused=True, beta_initializer='random_uniform',
                                            gamma_initializer='random_uniform',
                                            moving_mean_initializer='random_uniform',
                                            moving_variance_initializer='ones')

    bn_fp32 = bn(inputs)
    model_fp32 = tf.keras.Model(inputs=inputs, outputs=bn_fp32)
    bn_layers = [model_fp32.layers[1]]
    _test_bn_correctness(model_fp32, bn_layers, dummy_inputs)

    bn_wrapper = QcQuantizeWrapper(bn,
                            QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
                            QuantizerSettings(8, QuantizationDataType.int, 'nearest', 'tf', False, False, False),
                          num_inputs=1)(inputs)
    model_wrapper = tf.keras.Model(inputs=inputs, outputs=bn_wrapper)
    bn_layers_wrapper = [model_wrapper.layers[1].original_layer]
    _test_bn_correctness(model_wrapper, bn_layers_wrapper, dummy_inputs)

def test_keras_fp16_pc():
    model = keras_sequential_conv_net()
    rand = np.random.rand(1, 28, 28, 3)
    _ = model(rand)
    
    conv_layer = model.layers[0]
    
    num_c, axis = get_number_of_outputs_and_axis_handling(conv_layer, conv_layer.weights[0].shape, "weight")
    ppcq = ParamPerChannelQuantizer(conv_layer,
                                    name=conv_layer.weights[0].name.split(':')[0],
                                    quant_scheme=QuantScheme.post_training_tf_enhanced,
                                    round_mode='nearest',
                                    bitwidth=16,
                                    data_type=QuantizationDataType.float,
                                    is_symmetric=False,
                                    use_strict_symmetric=False,
                                    use_unsigned_symmetric=False,
                                    axis_handling=axis,
                                    num_output_channels=num_c,
                                    enabled=True)
    
    ppcq_rand_tensor = np.random.rand(1, 28, 28, 4)
    _ = ppcq(ppcq_rand_tensor)