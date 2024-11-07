# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import List

import aimet_common.libpymo as libpymo
import numpy as np
import pytest
import tensorflow as tf
from aimet_common.libpymo import TfEncoding
from packaging import version
from tensorflow import keras

from aimet_common.defs import QuantScheme, RANGE_LEARNING_SCHEMES
from aimet_common import quantsim
from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.keras.utils.quantizer_utils import SaveModelWithoutQuantsimWrappersCallback
from aimet_tensorflow.keras.cross_layer_equalization import equalize_model
from aimet_tensorflow.keras.quant_sim.tensor_quantizer import ParamPerTensorQuantizer, ParamPerChannelQuantizer
from aimet_tensorflow.keras.quant_sim.qc_mha_wrapper import QcQuantizableMultiHeadAttention
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.rnn.qc_quant_LSTM import QuantizedLSTM
from test_models_keras import tiny_conv_net


def conv_functional():
    input_shape = (128, 28, 28, 1)
    inp = tf.keras.Input(shape=input_shape[1:])
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3))(inp)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5, trainable=False)(x)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs=x, name='conv_functional')
    return model


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


def model_with_tf_indexedslices_grad():
    input_dim = 10
    output_dim = 5
    input_layer = keras.Input(shape=(input_dim,))
    embedding_layer = keras.layers.Embedding(input_dim, output_dim, input_length=input_dim)(input_layer)
    sum_embedding = tf.reduce_sum(embedding_layer, axis=1)
    output_layer = keras.layers.Dense(output_dim, activation="relu")(sum_embedding)
    return keras.Model(inputs=input_layer, outputs=output_layer, name="model_with_tf_indexedslices")

class DenseSubclassing(tf.keras.Model):
    def __init__(self):
        super(DenseSubclassing, self).__init__()
        self.linear1 = tf.keras.layers.Dense(units=2)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        x = self.linear1(inputs)
        x = self.softmax(x)
        return x


def model_with_lambda_operators():
    inp = tf.keras.layers.Input(shape=(5,))
    inp_2 = tf.keras.layers.Input(shape=(3,))
    x1 = tf.keras.layers.Dense(units=2)(inp)
    x2 = tf.keras.layers.Dense(units=2)(inp_2)
    x = x1 + x2
    x = x - 1.0
    x = x * x1
    x = x / x2
    model = tf.keras.Model(inputs=(inp, inp_2), outputs=x, name="model_with_lambda_operators")
    return model


def model_with_tf_op_lambda_operators_multi_tf_keras_input():
    input_layer = tf.keras.Input(batch_input_shape=(1, 16, 32, 3))
    x1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)(input_layer)
    x2 = tf.transpose(x1, perm=[0, 1, 3, 2])
    x = tf.matmul(x1, x2)
    output = tf.image.resize(x, [tf.shape(input_layer)[1], tf.shape(input_layer)[2]])
    return tf.keras.Model(
        inputs=input_layer,
        outputs=output,
        name="model_with_tf_op_lambda_operators_multi_tf_keras_input"
    )


def model_with_tf_op_lambda_operators_multi_tf_static_inputs():
    input_1 = tf.keras.Input(shape=(10,))
    input_2 = tf.keras.Input(shape=(20,))
    keras_concat = tf.keras.layers.Concatenate(axis=-1)([input_1, input_2])
    const = tf.constant(3.14, dtype=tf.float32, shape=(1,))
    const_expanded = tf.expand_dims(const, axis=0)
    tf_concat = tf.concat([keras_concat, const_expanded], axis=-1)
    output = tf.keras.layers.Dense(3)(tf_concat)

    return tf.keras.Model(
        inputs=[input_1, input_2],
        outputs=output,
        name="model_with_tf_op_lambda_operators_multi_tf_static_inputs"
    )


def model_with_reused_layer():
    relu = tf.keras.layers.ReLU()
    inp = tf.keras.layers.Input(shape=(5,))
    x = relu(inp)
    x = tf.keras.layers.Dense(units=2)(x)
    x = relu(x)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name="model_with_reused_layer")
    return model


class DenseReluLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DenseReluLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(units=2)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.relu(x)
        return x


def test_quantsim_basic():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = dense_functional()
        rand_inp = np.random.randn(100, 5)
        orig_out = model.predict(rand_inp)

        qsim = QuantizationSimModel(model, quant_scheme='tf')
        quant_wrappers = [quant_wrapper for quant_wrapper in qsim.quant_wrappers()]
        assert len(quant_wrappers) == 2
        assert len(quant_wrappers[0].param_quantizers) == 2
        for quant_wrapper in quant_wrappers:
            assert quant_wrapper.input_quantizers[0].quant_scheme == QuantScheme.post_training_tf
            assert quant_wrapper.input_quantizers[0].round_mode == libpymo.RoundingMode.ROUND_NEAREST
            assert quant_wrapper.output_quantizers[0].quant_scheme == QuantScheme.post_training_tf
            assert quant_wrapper.output_quantizers[0].round_mode == libpymo.RoundingMode.ROUND_NEAREST
        assert len(qsim.model.layers[1].input_quantizers) == 1
        assert len(qsim.model.layers[1].output_quantizers) == 1
        assert len(qsim.model.layers[1].param_quantizers) == 2
        assert len(qsim.model.layers[2].input_quantizers) == 1
        assert len(qsim.model.layers[2].output_quantizers) == 1
        assert len(qsim.model.layers[2].param_quantizers) == 0

        # Test that model output remains same prior to compute encodings
        # Disable param quantizers first, otherwise one shot quant/dequant will affect output
        qsim.model.layers[1].param_quantizers[0].disable()
        qsim.model.layers[1].param_quantizers[1].disable()
        quant_out = qsim.model.predict(rand_inp)
        assert np.array_equal(orig_out, quant_out)

        qsim.model.layers[1].param_quantizers[0].enable()
        qsim.model.layers[1].param_quantizers[1].enable()

        # Run one more forward pass after enabling param quantizers
        qsim.compute_encodings(lambda m, _: m(rand_inp), None)

        assert qsim.model.layers[1].param_quantizers[0].encoding is not None
        quant_out = qsim.model.predict(rand_inp)
        assert not np.array_equal(orig_out, quant_out)

        qsim.export('./data', 'test_export')


def test_quantsim_export_quantizer_args():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = dense_functional()
        rand_inp = np.random.randn(100, 5)

        qsim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced, default_param_bw=16,
                                    default_output_bw=16)

        qsim.export('./data', 'test_export_with_quant_args')

        with open('./data/test_export_with_quant_args.encodings') as json_file:
            encoding_data = json.load(json_file)

        assert "quantizer_args" in encoding_data
        quantizer_args = encoding_data["quantizer_args"]
        assert quantizer_args["activation_bitwidth"] == 16
        assert quantizer_args["param_bitwidth"] == 16
        assert not quantizer_args["per_channel_quantization"]
        assert quantizer_args["quant_scheme"] == QuantScheme.post_training_tf_enhanced.name
        assert quantizer_args["dtype"] == "int"
        assert quantizer_args["is_symmetric"]


def test_quantsim_with_custom_config_file():
    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "False"
            },
            "params": {
                "is_quantized": "False",
                "is_symmetric": "False"
            }
        },
        "params": {},
        "op_type": {},
        "supergroups": [
            {
                "op_list": ["Conv", "BatchNormalization"]
            },
            {
                "op_list": ["Relu", "MaxPool"]
            },
            {
                "op_list": ["Conv", "Relu", "AveragePool"]
            }
        ],
        "model_input": {},
        "model_output": {}
    }
    with open("./data/quantsim_config.json", "w") as f:
        json.dump(quantsim_config, f)

    model = tiny_conv_net()
    qsim = QuantizationSimModel(model, quant_scheme='tf', config_file="./data/quantsim_config.json")

    layers = qsim.model.layers
    conv1, relu1, conv2, conv3 = layers[1], layers[3], layers[5], layers[8]
    relu3 = layers[9]
    bn1, maxpool, bn2, avgpool = layers[2], layers[4], layers[6], layers[10]
    for layer in layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        # Check configs for starts of supergroups
        if layer in [conv1, relu1, conv2, conv3]:
            for q in layer.output_quantizers:
                assert not q.is_enabled()
        # Check configs for middle ops in supergroups
        elif layer == relu3:
            for q in layer.input_quantizers:
                assert not q.is_enabled()
            for q in layer.output_quantizers:
                assert not q.is_enabled()
        # Check configs for ends of supergroups
        elif layer in [bn1, maxpool, bn2, avgpool]:
            for q in layer.input_quantizers:
                assert not q.is_enabled()
            for q in layer.output_quantizers:
                assert q.is_enabled()
        else:
            for q in layer.input_quantizers:
                assert not q.is_enabled()
            for q in layer.output_quantizers:
                assert q.is_enabled()

    if os.path.exists("./data/quantsim_config.json"):
        os.remove("./data/quantsim_config.json")


def test_quantsim_handling_folded_bn_layer():
    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "False"
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": "False"
            }
        },
        "params": {},
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }
    with open("./data/quantsim_config.json", "w") as f:
        json.dump(quantsim_config, f)

    model = tiny_conv_net()
    cle_applied_model = equalize_model(model)
    qsim = QuantizationSimModel(cle_applied_model, quant_scheme='tf', config_file="./data/quantsim_config.json")

    layers = qsim.model.layers
    # Check that the BatchNormalization layers are removed
    assert not isinstance(layers[2], tf.keras.layers.BatchNormalization)
    assert not isinstance(layers[6], tf.keras.layers.BatchNormalization)
    assert len(cle_applied_model.layers) == len(model.layers) - 2

    for layer in layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        for q in layer.output_quantizers:
            assert q.is_enabled()
        for q in layer.param_quantizers:
            assert q.is_enabled()

    if os.path.exists("./data/quantsim_config.json"):
        os.remove("./data/quantsim_config.json")


def test_quantsim_with_specific_op_type_per_channel_quantization() -> None:
    """
    Test whether TfEncoding is set correctly when specific op type has per_channel_quantization property
    """
    quantsim_config = {
        "defaults": {
            "ops": {"is_output_quantized": "True"},
            "params": {
                "is_quantized": "True",
                "is_symmetric": "True"
            },
            "strict_symmetric": "False",
            "unsigned_symmetric": "True",
            "per_channel_quantization": "True"
        },
        "params": {
            "bias": {"is_quantized": "False"}
        },
        "op_type": {
            "Gemm": {"per_channel_quantization": "False"}
        },
        "supergroups": [],
        "model_input": {"is_input_quantized": "True"},
        "model_output": {}
    }
    with open("./data/quantsim_config.json", "w") as f:
        json.dump(quantsim_config, f)

    model = tiny_conv_net()
    cle_applied_model = equalize_model(model)
    qsim = QuantizationSimModel(cle_applied_model, quant_scheme='tf',
                                config_file="./data/quantsim_config.json")

    dummy_input = np.random.randn(4, 32, 32, 3)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)

    layers = qsim.model.layers
    conv1, conv2, conv3, dense = layers[1], layers[4], layers[6], layers[10]

    for conv_layer in [conv1, conv2, conv3]:
        # Conv type will follow default per_channel_quantization=True
        for q in conv_layer.param_quantizers:
            if q.is_enabled():
                assert isinstance(q.encoding, List)

    for q in dense.param_quantizers:
        # Gemm type will follow op_type per_channel_quantization=False
        if q.is_enabled():
            assert isinstance(q.encoding, TfEncoding)

    if os.path.exists("./data/quantsim_config.json"):
        os.remove("./data/quantsim_config.json")


def test_model_with_lambda_operators():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = model_with_lambda_operators()
        rand_inp_1 = np.random.randn(10, 5)
        rand_inp_2 = np.random.randn(10, 3)
        _ = model.predict((rand_inp_1, rand_inp_2))

        qsim = QuantizationSimModel(model, quant_scheme='tf')
        qsim.compute_encodings(lambda m, _: m((rand_inp_1, rand_inp_2)), None)
        qsim.export('./data', 'model_with_lambda_operators')
        assert len(list(qsim.quant_wrappers())) == 6

        with open("./data/model_with_lambda_operators.encodings", "r") as encodings_file:
            encodings = json.load(encodings_file)

        assert len(encodings['activation_encodings']) == 8
        # Note: Disable bias quantization in default_config.json
        assert len(encodings['param_encodings']) == 2


def test_model_with_tf_op_lambda_operators_multi_tf_keras_input():
    model = model_with_tf_op_lambda_operators_multi_tf_keras_input()
    random_input = tf.random.uniform((1, 16, 32, 3))

    with tempfile.TemporaryDirectory() as temp_dir:
        qsim = QuantizationSimModel(model, quant_scheme='tf')
        qsim.compute_encodings(lambda m, _: m(random_input), None)
        qsim.export(temp_dir, model.name)

        with open(os.path.join(temp_dir, f"{model.name}.encodings"), "r") as encodings_file:
            encodings = json.load(encodings_file)

        assert "transpose" in qsim.model.layers[2].original_layer.name, "This QCQuantizeWrapper should wrap the `tf.transpose` TF Op Lambda Layer"
        assert "matmul" in qsim.model.layers[5].original_layer.name, "This QCQuantizeWrapper should house the `tf.matmul` TF Op Lambda Layer"
        assert "image.resize" in qsim.model.layers[8].original_layer.name, "This QCQuantizeWrapper should house the `tf.image.resize` TF Op Lambda Layer"

        assert len(qsim.model.layers[2].input_quantizers) == 1, "tf.transpose should have only 1 input_quantizer"
        assert len(qsim.model.layers[5].input_quantizers) == 2, "tf.matmul should have 2 input_quantizer for a @ b"
        assert len(qsim.model.layers[8].input_quantizers) == 3, "tf.image.resize should have 3 input_quantizers for input, and the tensors for the new height and width for tf.shape"
        assert len(encodings['activation_encodings']) == 5
        assert len(encodings['param_encodings']) == 1, "Only the Dense layer in this model should have param_encoding"


def test_model_with_tf_op_lambda_operators_multi_tf_static_inputs():
    model = model_with_tf_op_lambda_operators_multi_tf_static_inputs()
    random_input = [tf.random.uniform(shape=(1, *shape[1:])) for shape in model.input_shape]

    qsim = QuantizationSimModel(model, quant_scheme="tf")
    qsim.compute_encodings(lambda m, _: m(random_input), None)
    with tempfile.TemporaryDirectory() as temp_dir:
        qsim.export(temp_dir, model.name, convert_to_pb=False)

        with open(os.path.join(temp_dir, f"{model.name}.encodings"), "r") as encodings_file:
            encodings = json.load(encodings_file)

        assert isinstance(qsim.model.layers[2].original_layer, tf.keras.layers.Concatenate), \
            "The layer should have a Concatenate layer"

        assert qsim.model.layers[3].original_layer.get_config()["name"] == "tf.concat", \
            "The layer should have a tf.concat"

        assert len(qsim.model.layers[2].input_quantizers) == 2,\
            "This QCQuantizeWrapper should have tf.keras.layers.Concatenate and have 2 input_quantizers"

        assert len(qsim.model.layers[3].input_quantizers) == 2, \
            "This QCQuantizeWrapper should have the tf.concat TFOpLambda layer and have 2 input_quantizers"

        assert len(encodings["activation_encodings"]) == 5
        assert len(encodings['param_encodings']) == 1, "Only the Dense layer in this model should have param_encoding"

def test_qat():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = dense_functional()
        rand_inp = np.random.randn(10, 5)
        rand_out = np.random.randn(10, 2)
        qsim = QuantizationSimModel(model, quant_scheme='tf', default_param_bw=8, default_output_bw=8)
        qsim.compute_encodings(lambda m, _: m.predict(rand_inp), None)
        qsim.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss=tf.keras.losses.MeanSquaredError())
        # Track weights for dense layer to check that they are updated during fit
        running_weights = [tf.keras.backend.get_value(param) for
                           param in qsim.model.layers[1]._layer_to_wrap.weights]
        # Track encoding max for dense output quantizer to check that it is not updated during fit
        running_dense_output_quantizer_encoding_max = \
            tf.keras.backend.get_value(qsim.model.layers[1].output_quantizers[0]._encoding_max)

        with tempfile.TemporaryDirectory() as tmp_dir:
            epochs = 10
            save_model_callback = SaveModelWithoutQuantsimWrappersCallback(qsim, tmp_dir, "test_qat")
            for i in range(epochs):
                _ = qsim.model.fit(x=rand_inp, y=rand_out, batch_size=1, callbacks=save_model_callback)
                ending_weights = [tf.keras.backend.get_value(param) for
                                  param in qsim.model.layers[1]._layer_to_wrap.weights]
                new_dense_output_quantizer_encoding_max = \
                    tf.keras.backend.get_value(qsim.model.layers[1].output_quantizers[0]._encoding_max)
                for idx, weight in enumerate(running_weights):
                    assert not np.array_equal(weight, ending_weights[idx])
                assert np.array_equal(new_dense_output_quantizer_encoding_max,
                                      running_dense_output_quantizer_encoding_max)
                running_weights = ending_weights
                running_dense_output_quantizer_encoding_max = new_dense_output_quantizer_encoding_max

            h5s = encodings = saved_models_folders = 0
            for file in os.listdir(tmp_dir):
                if file.endswith('h5'):
                    h5s += 1
                elif file.endswith('encodings'):
                    encodings += 1
                else:
                    saved_models_folders += 1

            for file_count in [h5s, encodings, saved_models_folders]:
                assert file_count == 1, f"QAT Save Callback did not work"

def test_range_learning():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        tf.keras.backend.clear_session()

        model = dense_functional()
        rand_inp = np.random.randn(10, 5)
        rand_out = np.random.randn(10, 2)
        qsim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                    default_param_bw=8, default_output_bw=8)
        for wrapper in qsim.quant_wrappers():
            wrapper.input_quantizers[0].disable()

        qsim.compute_encodings(lambda m, _: m.predict(rand_inp), None)
        qsim.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                     loss=tf.keras.losses.MeanSquaredError())
        # Track weights for dense layer to check that they are updated during fit
        running_weights = [tf.keras.backend.get_value(param) for
                           param in qsim.model.layers[1]._layer_to_wrap.weights]
        # Track encoding max for dense output quantizer to check that it is updated during fit
        running_dense_output_quantizer_encoding_max = \
            tf.keras.backend.get_value(qsim.model.layers[1].output_quantizers[0]._encoding_max)

        for i in range(10):
            _ = qsim.fit(x=rand_inp, y=rand_out, batch_size=1)
            ending_weights = [tf.keras.backend.get_value(param) for
                              param in qsim.model.layers[1]._layer_to_wrap.weights]
            new_dense_output_quantizer_encoding_max = \
                tf.keras.backend.get_value(qsim.model.layers[1].output_quantizers[0]._encoding_max)
            for idx, weight in enumerate(running_weights):
                assert not np.array_equal(weight, ending_weights[idx])
            assert not np.array_equal(new_dense_output_quantizer_encoding_max,
                                      running_dense_output_quantizer_encoding_max)
            running_weights = ending_weights
            running_dense_output_quantizer_encoding_max = new_dense_output_quantizer_encoding_max

        # Check that exporting encodings exports the learned encodings
        qsim.export('./data', 'dense_functional')
        with open("./data/dense_functional.encodings", "r") as encodings_file:
            encodings = json.load(encodings_file)
        assert np.allclose(encodings['activation_encodings']['dense/BiasAdd:0'][0]['max'],
                           running_dense_output_quantizer_encoding_max,
                           atol=encodings['activation_encodings']['dense/BiasAdd:0'][0]['scale'])


def test_qat_with_tf_indexedslices():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        tf.keras.backend.clear_session()

        epochs = 10
        model = model_with_tf_indexedslices_grad()
        X = tf.constant([[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]], dtype=tf.float32)
        y = tf.constant([[0, 1, 0, 0, 0]], dtype=tf.float32)
        sim = QuantizationSimModel(
            model,
            quant_scheme=QuantScheme.training_range_learning_with_tf_init,
            default_param_bw=8,
            default_output_bw=8
        )
        for wrapper in sim.quant_wrappers():
            wrapper.input_quantizers[0].disable()

        sim.compute_encodings(
            lambda m, _: m.predict(X),
            forward_pass_callback_args=None
        )
        sim.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.MeanSquaredError()
        )

        # Verify that tf.IndexedSlices should be in the gradients used for training
        with tf.GradientTape() as tape:
            outputs = sim.model(X)
            loss = tf.reduce_sum(tf.square(outputs - y))
        gradients = tape.gradient(loss, sim.model.trainable_variables)
        assert isinstance(gradients[0], tf.IndexedSlices), "Should house tf.IndexedSlices type for gradient"
        _ = sim.model.fit(x=X, y=y, epochs=epochs, batch_size=1)  # Run through and train normal, if it works we're OK


def test_assert_on_reused_layer():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = model_with_reused_layer()
        rand_inp = np.random.randn(100, 5)
        _ = model.predict(rand_inp)

        with pytest.raises(NotImplementedError):
            _ = QuantizationSimModel(model, quant_scheme='tf')


def test_quantizable_mha_basic():
    B = 5
    T = 8
    S = 4

    # STAGE 1 MODEL - model created with layers.MultiHeadAttention
    stage_1_q_inputs = keras.Input(shape=(T, 16))
    stage_1_v_inputs = keras.Input(shape=(S, 16))
    stage_1_output = keras.layers.MultiHeadAttention(key_dim=2, num_heads=2)(stage_1_q_inputs, stage_1_v_inputs)
    stage_1_model = keras.Model(inputs=[stage_1_q_inputs, stage_1_v_inputs], outputs=stage_1_output)

    # STAGE 2 MODEL - model manually created with QcQuantizeableMultiHeadAttention
    stage_2_q_inputs = keras.Input(shape=(T, 16))
    stage_2_v_inputs = keras.Input(shape=(S, 16))
    stage_2_output = QcQuantizableMultiHeadAttention(key_dim=2, num_heads=2)(stage_2_q_inputs, stage_2_v_inputs)
    stage_2_model = keras.Model(inputs=[stage_2_q_inputs, stage_2_v_inputs], outputs=stage_2_output)

    # STAGE 3 MODEL - model created using QuantSim
    stage_3_model = QuantizationSimModel(stage_1_model)

    query = np.ones([B, T, 16])
    value = np.ones([B, S, 16])

    output_1_tensor = stage_1_model([query, value])
    output_2_tensor = stage_2_model([query, value])
    output_3_tensor = stage_3_model.model([query, value])

    for layer in stage_3_model.model.layers:
        if isinstance(layer, QcQuantizableMultiHeadAttention): layer.deactivate_quantizers()
    output_3_tensor_without_quantizers = stage_3_model.model([query, value])
    for layer in stage_3_model.model.layers:
        if isinstance(layer, QcQuantizableMultiHeadAttention): layer.reactivate_quantizers()

    # check that all output tensors have the same shape
    assert output_1_tensor.shape == output_2_tensor.shape == output_3_tensor.shape

    # check that QcQuantizableMultiHeadAttention does not exist in original model.layers
    assert not any(isinstance(layer, QcQuantizableMultiHeadAttention) for layer in stage_1_model.layers)

    # check that QcQuantizableMultiHeadAttention exists in QuantSim model.layers
    assert any(isinstance(layer, QcQuantizableMultiHeadAttention) for layer in stage_3_model.model.layers)

    # check that QuantSim generated model has same output as original model when quantizers are disabled
    assert tf.equal(output_1_tensor, output_3_tensor_without_quantizers).numpy().flatten().all()


def test_quantizable_mha_with_value():
    B = 5
    T = 8
    S = 4

    q_inputs = keras.Input(shape=(T, 16))
    v_inputs = keras.Input(shape=(S, 16))
    k_inputs = keras.Input(shape=(S, 16))
    model_output = keras.layers.MultiHeadAttention(key_dim=2, num_heads=2)(q_inputs, v_inputs, k_inputs)
    unquantized_model = keras.Model(inputs=[q_inputs, v_inputs, k_inputs], outputs=model_output)

    quantized_model = QuantizationSimModel(unquantized_model)

    query = np.ones([B, T, 16])
    value = np.ones([B, S, 16])
    key = np.ones([B, S, 16])

    unquantized_model_tensor = unquantized_model([query, value, key])
    quantized_model_tensor = quantized_model.model([query, value, key])

    for layer in quantized_model.model.layers:
        if isinstance(layer, QcQuantizableMultiHeadAttention): layer.deactivate_quantizers()
    quantized_model_tensor_without_quantizers = quantized_model.model([query, value, key])
    for layer in quantized_model.model.layers:
        if isinstance(layer, QcQuantizableMultiHeadAttention): layer.reactivate_quantizers()

    # check that all output tensors have the same shape
    assert unquantized_model_tensor.shape == quantized_model_tensor.shape == \
           quantized_model_tensor_without_quantizers.shape

    # check that QuantSim generated model has same output as original model when quantizers are disabled
    assert tf.equal(unquantized_model_tensor, quantized_model_tensor_without_quantizers).numpy().flatten().all()

    # check that QcQuantizableMultiHeadAttention does not exist in original model.layers
    assert not any(isinstance(layer, QcQuantizableMultiHeadAttention) for layer in unquantized_model.layers)

    # check that QcQuantizableMultiHeadAttention exists in QuantSim model.layers
    assert any(isinstance(layer, QcQuantizableMultiHeadAttention) for layer in quantized_model.model.layers)


def test_quantizable_mha_with_mask():
    B = 5
    T = 8
    S = 4

    q_inputs = keras.Input(shape=(T, 16))
    v_inputs = keras.Input(shape=(S, 16))
    k_inputs = keras.Input(shape=(S, 16))
    m_inputs = keras.Input(shape=(T, S))
    model_output = keras.layers.MultiHeadAttention(key_dim=2, num_heads=2)(q_inputs, v_inputs, k_inputs, m_inputs)
    unquantized_model = keras.Model(inputs=[q_inputs, v_inputs, k_inputs, m_inputs], outputs=model_output)

    quantized_model = QuantizationSimModel(unquantized_model)

    query = np.ones([B, T, 16])
    value = np.ones([B, S, 16])
    key = np.ones([B, S, 16])
    mask = np.zeros([B, T, S])

    unquantized_model_tensor = unquantized_model([query, value, key, mask])
    quantized_model_tensor = quantized_model.model([query, value, key, mask])

    for layer in quantized_model.model.layers:
        if isinstance(layer, QcQuantizableMultiHeadAttention): layer.deactivate_quantizers()
    quantized_model_tensor_without_quantizers = quantized_model.model([query, value, key, mask])
    for layer in quantized_model.model.layers:
        if isinstance(layer, QcQuantizableMultiHeadAttention): layer.reactivate_quantizers()

    # check that all output tensors have the same shape
    assert unquantized_model_tensor.shape == quantized_model_tensor.shape == \
           quantized_model_tensor_without_quantizers.shape

    # check that QuantSim generated model has same output as original model when quantizers are disabled
    assert tf.equal(unquantized_model_tensor, quantized_model_tensor_without_quantizers).numpy().flatten().all()

    # check that QcQuantizableMultiHeadAttention does not exist in original model.layers
    assert not any(isinstance(layer, QcQuantizableMultiHeadAttention) for layer in unquantized_model.layers)

    # check that QcQuantizableMultiHeadAttention exists in QuantSim model.layers
    assert any(isinstance(layer, QcQuantizableMultiHeadAttention) for layer in quantized_model.model.layers)


def test_quantizable_mha_encodings():
    B = 5
    T = 8
    S = 4

    q_inputs = keras.Input(shape=(T, 16))
    v_inputs = keras.Input(shape=(S, 16))
    k_inputs = keras.Input(shape=(S, 16))
    m_inputs = keras.Input(shape=(T, S))
    model_output = keras.layers.MultiHeadAttention(key_dim=2, num_heads=2)(q_inputs, v_inputs, k_inputs, m_inputs)
    unquantized_model = keras.Model(inputs=[q_inputs, v_inputs, k_inputs, m_inputs], outputs=model_output)

    quantized_model = QuantizationSimModel(unquantized_model)

    rng = np.random.default_rng(seed=42)
    query = rng.random([B, T, 16])
    value = rng.random([B, S, 16])
    key = rng.random([B, S, 16])
    mask = np.zeros([B, T, S])

    quantized_model.compute_encodings(lambda m, _: m([query, value, key, mask]), None)

    query = query * 10
    value = value * 10
    key = key * 10

    unquantized_model_tensor = unquantized_model([query, value, key, mask]).numpy().flatten()
    quantized_model_tensor = quantized_model.model([query, value, key, mask]).numpy().flatten()

    output_encoding_min = quantized_model.model.layers[-1]._wrapped_layers[-1].output_quantizers[0]._encoding_min
    output_encoding_max = quantized_model.model.layers[-1]._wrapped_layers[-1].output_quantizers[0]._encoding_max

    # checking to make sure all outputs fall within the limits set by the output quantizer
    FLOAT_DELTA = 0.0001
    assert all((quantized_model_tensor >= output_encoding_min - FLOAT_DELTA) &
               (quantized_model_tensor <= output_encoding_max + FLOAT_DELTA))


def test_quantizable_mha_export_encodings():
    B = 5
    T = 8
    S = 4

    # STAGE 1 MODEL - model created with layers.MultiHeadAttention
    stage_1_q_inputs = keras.Input(shape=(T, 16))
    stage_1_v_inputs = keras.Input(shape=(S, 16))
    stage_1_output = keras.layers.MultiHeadAttention(key_dim=2, num_heads=2)(stage_1_q_inputs, stage_1_v_inputs)
    stage_1_model = keras.Model(inputs=[stage_1_q_inputs, stage_1_v_inputs], outputs=stage_1_output)

    # STAGE 3 MODEL - model created using QuantSim
    stage_3_model = QuantizationSimModel(stage_1_model)

    rng = np.random.default_rng(seed=42)
    query = rng.random([B, T, 16]) * 100
    value = rng.random([B, S, 16]) * 100

    stage_3_model.compute_encodings(lambda m, _: m([query, value]), None)
    stage_3_model.export('./data', 'mha')

    with open("./data/mha.encodings", "r") as encodings_file:
        encodings = json.load(encodings_file)

    for wrapper in stage_3_model.model.layers[2]._wrapped_layers:
        for i_quantizer in wrapper.input_quantizers:
            if i_quantizer.encoding is not None:
                tensor_name = wrapper.name + "/" + i_quantizer.name + ":0"
                encoding_dict = QuantizationSimModel._get_encoding_dict_for_quantizer(i_quantizer)
                assert tensor_name in encodings['activation_encodings']
                assert encodings['activation_encodings'][tensor_name] == encoding_dict
        for o_quantizer in wrapper.output_quantizers:
            if o_quantizer.encoding is not None:
                tensor_name = wrapper.name + ":0"
                encoding_dict = QuantizationSimModel._get_encoding_dict_for_quantizer(o_quantizer)
                assert tensor_name in encodings['activation_encodings']
                assert encodings['activation_encodings'][tensor_name] == encoding_dict
        for idx, param_quantizer in enumerate(wrapper.param_quantizers):
            if param_quantizer.encoding is not None:
                param_name = wrapper._layer_to_wrap.weights[idx].name
                encoding_dict = QuantizationSimModel._get_encoding_dict_for_quantizer(param_quantizer)
                assert param_name in encodings['param_encodings']
                assert encodings['param_encodings'][param_name] == encoding_dict


def _common_stays_valid_after_export_helper(model, rand_inp, config=None):
    tf.keras.backend.clear_session()
    sim = QuantizationSimModel(model, quant_scheme='tf', config_file=config)
    # Turn off random quantization layers and setting different bw to see if the rebuilt model reflects the same changes
    for i, layer in enumerate(sim.model.layers):
        if not isinstance(layer, tf.keras.layers.InputLayer) and i % 2 == 0:
            layer.bitwidth = 16
        if i % 2 != 0:
            layer.input_quantizers.enabled = False
            layer.output_quantizers.enabled = False
            layer.param_quantizers.enabled = False

    sim.compute_encodings(lambda m, _: m.predict(rand_inp), None)

    original_sim_output = sim.model.predict(rand_inp)
    original_sim_model_weights = sim.model.get_weights()
    original_layer_and_quantizers = {}
    for layer in sim.model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        original_layer_and_quantizers[layer.name] = {}
        original_layer_and_quantizers[layer.name]["input_quantizers"] = layer.input_quantizers
        original_layer_and_quantizers[layer.name]["output_quantizers"] = layer.output_quantizers
        original_layer_and_quantizers[layer.name]["param_quantizers"] = layer.param_quantizers

    # Make tmp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        sim.export(path=tmp_dir, filename_prefix="test")

    try:
        _ = sim.model.predict(rand_inp)
    except ValueError:
        pytest.fail("Model is no longer valid after export")

    for i, _ in enumerate(original_sim_model_weights):
        np.testing.assert_array_equal(original_sim_model_weights[i], sim.model.get_weights()[i])

    def check_quantizers(original_quantizer, new_quantizer):
        if not original_quantizer and not new_quantizer:
            return

        if not isinstance(original_quantizer.tensor_quantizer, List):
            original_quantizer_tensor_quantizers = [original_quantizer.tensor_quantizer]
            new_quantizer_tensor_quantizers = [new_quantizer.tensor_quantizer]
        else:
            original_quantizer_tensor_quantizers = original_quantizer.tensor_quantizer
            new_quantizer_tensor_quantizers = new_quantizer.tensor_quantizer

        assert original_quantizer.is_enabled() == new_quantizer.is_enabled(), f"original: {original_quantizer.enabled}, new: {new_quantizer.enabled}"
        assert original_quantizer.is_encoding_valid() == new_quantizer.is_encoding_valid(), f"original: {original_quantizer.is_encoding_valid()}, new: {new_quantizer.is_encoding_valid()}"
        assert original_quantizer._is_encoding_frozen == new_quantizer._is_encoding_frozen, f"original: {original_quantizer._is_encoding_frozen}, new: {new_quantizer._is_encoding_frozen}"

        for orig_tq, new_tq in zip(original_quantizer_tensor_quantizers, new_quantizer_tensor_quantizers):
            assert orig_tq.isEncodingValid == new_tq.isEncodingValid, f"original: {orig_tq.is_encoding_valid()}, new: {new_tq.is_encoding_valid()}"

    def check_encodings(original_encoding, new_encoding):
        if not original_encoding and not new_encoding:
            return

        if not isinstance(original_encoding, List):
            original_encoding = [original_encoding]
            new_encoding = [new_encoding]

        for org_quant, new_quant in zip(original_encoding, new_encoding):
            assert org_quant.bw == new_quant.bw, f"original: {org_quant.bw}, new: {new_quant.bw}"
            assert org_quant.delta == new_quant.delta, f"original: {org_quant.delta}, new: {new_quant.delta}"
            assert org_quant.offset == new_quant.offset, f"original: {org_quant.offset}, new: {new_quant.offset}"
            assert org_quant.min == new_quant.min, f"original: {org_quant.min}, new: {new_quant.min}"
            assert org_quant.max == new_quant.max, f"original: {org_quant.max}, new: {new_quant.max}"

    for layer in sim.model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        assert len(layer.input_quantizers) == len(original_layer_and_quantizers[layer.name][
                                                      "input_quantizers"]), f"Not the same number of input quantizers for layer {layer.name}"
        for i, _ in enumerate(layer.input_quantizers):
            check_quantizers(original_layer_and_quantizers[layer.name]["input_quantizers"][i],
                             layer.input_quantizers[i])
            check_encodings(original_layer_and_quantizers[layer.name]["input_quantizers"][i].encoding,
                            layer.input_quantizers[i].encoding)
        assert len(layer.output_quantizers) == len(original_layer_and_quantizers[layer.name][
                                                       "output_quantizers"]), f"Not the same number of output quantizers for layer {layer.name}"
        for i, _ in enumerate(layer.output_quantizers):
            check_quantizers(original_layer_and_quantizers[layer.name]["output_quantizers"][i],
                             layer.output_quantizers[i])
            check_encodings(original_layer_and_quantizers[layer.name]["output_quantizers"][i].encoding,
                            layer.output_quantizers[i].encoding)

        assert len(layer.param_quantizers) == len(original_layer_and_quantizers[layer.name][
                                                      "param_quantizers"]), f"Not the same number of param quantizers for layer {layer.name}"
        for i, _ in enumerate(layer.param_quantizers):
            check_quantizers(original_layer_and_quantizers[layer.name]["param_quantizers"][i],
                             layer.param_quantizers[i])
            check_encodings(original_layer_and_quantizers[layer.name]["param_quantizers"][i].encoding,
                            layer.param_quantizers[i].encoding)

    np.testing.assert_array_equal(original_sim_output, sim.model.predict(rand_inp),
                                  err_msg="Model output changed after export")


def test_model_stays_valid_after_export_per_tensor():
    model = conv_functional()
    rand_inp = tf.random.normal(shape=(128, *model.input_shape[1:]))
    _ = model.predict(rand_inp)

    _common_stays_valid_after_export_helper(model, rand_inp)


def test_model_stays_valid_after_export_per_channel():
    model = conv_functional()
    rand_inp = tf.random.normal(shape=(128, *model.input_shape[1:]))
    _ = model.predict(rand_inp)

    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True"
            },
            "params": {
                "is_symmetric": "True",
                "is_quantized": "True"
            },
            "per_channel_quantization": "True",
        },
        "params": {},
        "op_type": {
            "Conv": {
                "is_input_quantized": "True",
                "is_output_quantized": "True"
            },
            "ConvTranspose": {
                "is_input_quantized": "True",
                "is_output_quantized": "True"
            },
            "Gemm": {
                "is_input_quantized": "True",
                "is_output_quantized": "True"
            },
            "MatMul": {
                "is_input_quantized": "True",
                "is_output_quantized": "True"
            },
            "MaxPooling2D": {
                "is_input_quantized": "True",
                "is_output_quantized": "True"
            }
        },
        "supergroups": [],
        "model_input": {},
        "model_output": {
            "is_output_quantized": "True"
        }
    }

    tmp_config_file = os.path.join(tempfile.mkdtemp(), 'config.json')
    with open(tmp_config_file, 'w') as f:
        json.dump(quantsim_config, f)

    _common_stays_valid_after_export_helper(model, rand_inp, config=tmp_config_file)

    if os.path.exists(tmp_config_file):
        os.remove(tmp_config_file)


@pytest.mark.parametrize('param_encodings', [{'conv2d_1/kernel:0':
                                                  [{'bitwidth': 4, "dtype": 'int', 'is_symmetric': "False",
                                                    'max': 0.14584073424339294, 'min': -0.12761062383651733,
                                                    'offset': -7.0, 'scale': 0.01823008991777897}]},
                                             {'conv2d_1/kernel:0': [{'bitwidth': 16, 'dtype': 'float'}]}])
@pytest.mark.parametrize('activation_encodings', [{"conv2d_1/Tanh:0":
                                                       [{"bitwidth": 8, "dtype": "int", "is_symmetric": "False",
                                                         "max": 5.99380955882352939, "min": -7.77575294117647056,
                                                         "offset": -144, "scale": 0.05399828431372549}]},
                                                  {"conv2d_1/Tanh:0": [{'bitwidth': 16, 'dtype': 'float'}]}])
def test_load_encodings(param_encodings, activation_encodings):
    """ Test load encodings functionality """
    tf.compat.v1.reset_default_graph()

    model = keras_model()

    sim = QuantizationSimModel(model)

    dummy_encodings = {"activation_encodings": activation_encodings,
                       "param_encodings": param_encodings}

    # export encodings to JSON file
    encoding_file_path = os.path.join('./', 'dummy.encodings')
    with open(encoding_file_path, 'w') as encoding_fp:
        json.dump(dummy_encodings, encoding_fp, sort_keys=True, indent=4)

    sim.load_encodings_to_sim(encoding_file_path='./dummy.encodings')

    extracted_encoding = sim.get_encodings_dict()

    # For param
    expected_encoding = param_encodings['conv2d_1/kernel:0'][0]
    actual_encoding = extracted_encoding["param_encodings"]['conv2d_1/kernel:0'][0]
    assert actual_encoding.get('dtype') == expected_encoding.get('dtype')
    assert actual_encoding.get('bitwidth') == expected_encoding.get('bitwidth')
    if actual_encoding.get('dtype') == 'int':
        assert actual_encoding.get('offset') == expected_encoding.get('offset')
        assert actual_encoding.get('is_symmetric') == expected_encoding.get('is_symmetric')
        assert np.allclose(actual_encoding.get('min'), expected_encoding.get('min'), atol=1e-5)
        assert np.allclose(actual_encoding.get('max'), expected_encoding.get('max'), atol=1e-5)

    # For activation
    expected_encoding = activation_encodings["conv2d_1/Tanh:0"][0]
    actual_encoding = extracted_encoding["activation_encodings"]["conv2d_1/Tanh:0"][0]
    assert actual_encoding.get('dtype') == expected_encoding.get('dtype')
    assert actual_encoding.get('bitwidth') == expected_encoding.get('bitwidth')
    if actual_encoding.get('dtype') == 'int':
        assert actual_encoding.get('offset') == expected_encoding.get('offset')
        assert actual_encoding.get('is_symmetric') == expected_encoding.get('is_symmetric')
        assert np.allclose(actual_encoding.get('min'), expected_encoding.get('min'), atol=1e-5)
        assert np.allclose(actual_encoding.get('max'), expected_encoding.get('max'), atol=1e-5)

    # Delete encodings JSON file
    if os.path.exists("./dummy.encodings"):
        os.remove("./dummy.encodings")

@pytest.mark.parametrize('param_encodings', [{'conv2d_1/kernel:0': [{'bitwidth': 16, 'dtype': 'float'}]}])
@pytest.mark.parametrize('activation_encodings', [{"conv2d_1/Tanh:0":
                                                       [{"bitwidth": 8, "dtype": "int", "is_symmetric": "False",
                                                         "max": 5.99380955882352939, "min": -7.77575294117647056,
                                                         "offset": -144, "scale": 0.05399828431372549}]},
                                                  {"conv2d_1/Tanh:0": [{'bitwidth': 16, 'dtype': 'float'}]}])
def test_load_encodings_with_custom_config(param_encodings, activation_encodings):
    """ Test load encodings functionality """
    tf.compat.v1.reset_default_graph()

    model = keras_model()

    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "True"
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": "True"
            },
            "per_channel_quantization": "True",
        },
        "params": {},
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        custom_config = os.path.join(tmp_dir, 'quantsim_config.json')
        with open(custom_config, 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(model, config_file=custom_config)

        dummy_encodings = {"activation_encodings": activation_encodings,
                           "param_encodings": param_encodings}

        # export encodings to JSON file
        encoding_file_path = os.path.join(tmp_dir, 'dummy.encodings')
        with open(encoding_file_path, 'w') as encoding_fp:
            json.dump(dummy_encodings, encoding_fp, sort_keys=True, indent=4)

        # Before loading encodings param quantizer will be PCQ
        for wrapper in sim.quant_wrappers():
            for idx, param_quantizer in enumerate(wrapper.param_quantizers):
                param_name = wrapper._layer_to_wrap.weights[idx].name
                if param_name == 'conv2d_1/kernel:0':
                    assert isinstance(param_quantizer, ParamPerChannelQuantizer)

        sim.load_encodings_to_sim(encoding_file_path=encoding_file_path)

        # After loading, as the encoding is float, the quantizer must be modified to PTQ
        for wrapper in sim.quant_wrappers():
            for idx, param_quantizer in enumerate(wrapper.param_quantizers):
                param_name = wrapper._layer_to_wrap.weights[idx].name
                if param_name == 'conv2d_1/kernel:0':
                    assert isinstance(param_quantizer, ParamPerTensorQuantizer)


def test_load_encodings_with_disabled_param():
    """ Test load encodings functionality with PCQ """

    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "True"
            },
            "params": {
                "is_quantized": "False",
                "is_symmetric": "True"
            },
        },
        "params": {},
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }
    with open('./quantsim_config.json', 'w') as f:
        json.dump(quantsim_config, f)
    tf.compat.v1.reset_default_graph()

    model = keras_model()

    sim = QuantizationSimModel(model, config_file='./quantsim_config.json')
    param_encodings = {'conv2d_1/kernel:0': [{'bitwidth': 4, 'dtype': 'int', 'is_symmetric': "False",
                                              'max': 0.14584073424339294,
                                              'min': -0.12761062383651733,
                                              'offset': -7.0, 'scale': 0.01823008991777897}]}
    activation_encodings = {"conv2d_1/Tanh:0": [
        {
            "bitwidth": 8,
            "dtype": "int",
            "is_symmetric": "False",
            "max": 5.99380955882352939,
            "min": -7.77575294117647056,
            "offset": -144,
            "scale": 0.05399828431372549
        }
    ]}

    dummy_encodings = {"activation_encodings": activation_encodings,
                       "param_encodings": param_encodings}

    # export encodings to JSON file
    encoding_file_path = os.path.join('./', 'dummy.encodings')
    with open(encoding_file_path, 'w') as encoding_fp:
        json.dump(dummy_encodings, encoding_fp, sort_keys=True, indent=4)

    sim.load_encodings_to_sim(encoding_file_path='./dummy.encodings')

    extracted_encoding = sim.get_encodings_dict()

    # For param
    # expected_encoding = param_encodings['conv2d_1/kernel:0']
    # actual_encoding   = extracted_encoding["param_encodings"]['conv2d_1/kernel:0']
    # for i in range(4):
    #     assert actual_encoding[i].get('min') == expected_encoding[i].get('min')
    #     assert actual_encoding[i].get('max') == expected_encoding[i].get('max')

    assert 'conv2d_1/kernel:0' not in extracted_encoding["param_encodings"]

    # For activation
    expected_encoding = activation_encodings["conv2d_1/Tanh:0"][0]
    actual_encoding = extracted_encoding["activation_encodings"]["conv2d_1/Tanh:0"][0]
    assert actual_encoding.get('bitwidth') == expected_encoding.get('bitwidth')
    assert actual_encoding.get('offset') == expected_encoding.get('offset')
    assert actual_encoding.get('is_symmetric') == expected_encoding.get('is_symmetric')
    assert np.allclose(actual_encoding.get('min'), expected_encoding.get('min'), atol=1e-5)
    assert np.allclose(actual_encoding.get('max'), expected_encoding.get('max'), atol=1e-5)

    # Delete encodings JSON file
    if os.path.exists("./dummy.encodings"):
        os.remove("./dummy.encodings")


def test_load_encodings_pcq():
    """ Test load encodings functionality with PCQ """

    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "True"
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": "True"
            },
            "per_channel_quantization": "True",
        },
        "params": {},
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }
    with open('./quantsim_config.json', 'w') as f:
        json.dump(quantsim_config, f)
    tf.compat.v1.reset_default_graph()

    model = keras_model()

    sim = QuantizationSimModel(model, config_file='./quantsim_config.json')
    param_encodings = {'conv2d_1/kernel:0': [{'bitwidth': 4, 'dtype': 'int', 'is_symmetric': "False",
                                              'max': 0.14584073424339294,
                                              'min': -0.12761062383651733,
                                              'offset': -7.0, 'scale': 0.01823008991777897},
                                             {'bitwidth': 4, 'dtype': 'int', 'is_symmetric': "False",
                                              'max': 0.14584073424339294,
                                              'min': -0.12761062383651733,
                                              'offset': -7.0, 'scale': 0.01823008991777897},
                                             {'bitwidth': 4, 'dtype': 'int', 'is_symmetric': "False",
                                              'max': 0.14584073424339294,
                                              'min': -0.12761062383651733,
                                              'offset': -7.0, 'scale': 0.01823008991777897},
                                             {'bitwidth': 4, 'dtype': 'int', 'is_symmetric': "False",
                                              'max': 0.14584073424339294,
                                              'min': -0.12761062383651733,
                                              'offset': -7.0, 'scale': 0.01823008991777897}]}
    activation_encodings = {"conv2d_1/Tanh:0": [
        {
            "bitwidth": 8,
            "dtype": "int",
            "is_symmetric": "False",
            "max": 5.99380955882352939,
            "min": -7.77575294117647056,
            "offset": -144,
            "scale": 0.05399828431372549
        }
    ]}

    dummy_encodings = {"activation_encodings": activation_encodings,
                       "param_encodings": param_encodings}

    # export encodings to JSON file
    encoding_file_path = os.path.join('./', 'dummy.encodings')
    with open(encoding_file_path, 'w') as encoding_fp:
        json.dump(dummy_encodings, encoding_fp, sort_keys=True, indent=4)

    sim.load_encodings_to_sim(encoding_file_path='./dummy.encodings')

    extracted_encoding = sim.get_encodings_dict()

    # For param
    expected_encoding = param_encodings['conv2d_1/kernel:0']
    actual_encoding = extracted_encoding["param_encodings"]['conv2d_1/kernel:0']
    for i in range(4):
        assert actual_encoding[i].get('bitwidth') == expected_encoding[i].get('bitwidth')
        assert actual_encoding[i].get('offset') == expected_encoding[i].get('offset')
        assert actual_encoding[i].get('is_symmetric') == expected_encoding[i].get('is_symmetric')
        assert np.allclose(actual_encoding[i].get('min'), expected_encoding[i].get('min'), atol=1e-5)
        assert np.allclose(actual_encoding[i].get('max'), expected_encoding[i].get('max'), atol=1e-5)

    # For activation
    expected_encoding = activation_encodings["conv2d_1/Tanh:0"][0]
    actual_encoding = extracted_encoding["activation_encodings"]["conv2d_1/Tanh:0"][0]
    assert actual_encoding.get('bitwidth') == expected_encoding.get('bitwidth')
    assert actual_encoding.get('offset') == expected_encoding.get('offset')
    assert actual_encoding.get('is_symmetric') == expected_encoding.get('is_symmetric')
    assert np.allclose(actual_encoding.get('min'), expected_encoding.get('min'), atol=1e-5)
    assert np.allclose(actual_encoding.get('max'), expected_encoding.get('max'), atol=1e-5)

    # Delete encodings JSON file
    if os.path.exists("./dummy.encodings"):
        os.remove("./dummy.encodings")


@pytest.mark.cuda
@pytest.mark.parametrize(
    "quant_scheme",
    [QuantScheme.post_training_tf, QuantScheme.training_range_learning_with_tf_init,
     QuantScheme.post_training_tf_enhanced, QuantScheme.training_range_learning_with_tf_enhanced_init]
)
def test_initialization_and_export_non_strict_symmetric(quant_scheme) -> None:
    """
    Test initial encoding min/max and result of export value
        under non-strict symmetric per-tensor quantization
    """
    tf.compat.v1.reset_default_graph()

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(2, (3, 3), input_shape=(32, 32, 4,)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='softmax', name="keras_model")])

    sim = QuantizationSimModel(model, quant_scheme=quant_scheme)

    # Enable input
    sim.compute_encodings(lambda m, _: m(np.random.randn(1, 32, 32, 4)), None)
    conv_op = sim.layers[1]
    initialized_encoding_min = tf.keras.backend.get_value(conv_op.param_quantizers[0].encoding_min)
    initialized_encoding_max = tf.keras.backend.get_value(conv_op.param_quantizers[0].encoding_max)

    if quant_scheme in RANGE_LEARNING_SCHEMES:
        # range learning scheme calibrates min value. encoding_min == -encoding_max
        assert initialized_encoding_min == -initialized_encoding_max
    else:
        # post_training scheme doesn't calibrate min value. encoding_min == -encoding_max - delta
        assert initialized_encoding_min != -initialized_encoding_max

    with tempfile.TemporaryDirectory() as tmp_dir:
        sim.export(tmp_dir, "quant_sim_model")
        with open(Path(tmp_dir, "quant_sim_model.encodings")) as json_file:
            encoding_data = json.load(json_file)

            param_encodings = encoding_data["param_encodings"]
            for encodings in param_encodings.values():
                for encoding_info in encodings:
                    encoding_min = encoding_info["min"]
                    encoding_max = encoding_info["max"]
                    scale = encoding_info["scale"]
                    offset = encoding_info["offset"]

                # We are enforcing strict symmetric flag to True in case of Range Learning. So encoding_min == -encoding_max.
                if quant_scheme in RANGE_LEARNING_SCHEMES:
                    assert encoding_min == -encoding_max
                    assert offset == -127
                    assert np.isclose(encoding_max, encoding_min + scale * 254, atol=1e-6)
                else:
                    # In post training scheme case, it doesn't seem to match exactly due to floating point arithmetic
                    assert np.isclose(encoding_min, -encoding_max - scale)
                    assert offset == -128
                    assert np.isclose(encoding_max, encoding_min + scale * 255, atol=1e-6)
                assert np.isclose(encoding_min, scale * offset, atol=1e-6)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "quant_scheme",
    [QuantScheme.post_training_tf, QuantScheme.training_range_learning_with_tf_init,
     QuantScheme.post_training_tf_enhanced, QuantScheme.training_range_learning_with_tf_enhanced_init]
)
def test_initialization_and_export_non_strict_symmetric_per_channel(quant_scheme) -> None:
    """
    Test initial encoding min/max and result of export value
        under non-strict symmetric per-channel quantization
    """
    tf.compat.v1.reset_default_graph()
    quantsim_config = {
        "defaults": {
            "ops": {"is_output_quantized": "True"},
            "params": {
                "is_quantized": "True",
                "is_symmetric": "True"
            },
            "strict_symmetric": "False",
            "per_channel_quantization": "True"
        },
        "params": {"bias": {"is_quantized": "False"}},
        "op_type": {
            "Squeeze": {"is_output_quantized": "False"},
            "Pad": {"is_output_quantized": "False"},
            "Mean": {"is_output_quantized": "False"},
            "Gemm": {"per_channel_quantization": "False"}
        },
        "supergroups": [
            {"op_list": ["Conv", "Relu"]},
            {"op_list": ["Conv", "Clip"]},
            {"op_list": ["Add", "Relu"]},
            {"op_list": ["Gemm", "Relu"]}
        ],
        "model_input": {"is_input_quantized": "True"},
        "model_output": {}
    }
    with open("./quantsim_config.json", "w") as f:
        json.dump(quantsim_config, f)

    tf.compat.v1.reset_default_graph()

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(2, (3, 3), input_shape=(32, 32, 4,)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='softmax', name="keras_model")])

    sim = QuantizationSimModel(model, quant_scheme=quant_scheme, config_file='./quantsim_config.json')

    # Enable input
    sim.compute_encodings(lambda m, _: m(np.random.randn(1, 32, 32, 4)), None)
    conv_op = sim.layers[1]
    initialized_encoding_min = tf.keras.backend.get_value(conv_op.param_quantizers[0].encoding_min)
    initialized_encoding_max = tf.keras.backend.get_value(conv_op.param_quantizers[0].encoding_max)

    if quant_scheme in RANGE_LEARNING_SCHEMES:
        # range learning scheme calibrates min value. encoding_min == -encoding_max
        assert all(initialized_encoding_min == -initialized_encoding_max)
    else:
        # post_training scheme doesn't calibrate min value. encoding_min == -encoding_max - delta
        assert not all(initialized_encoding_min == -initialized_encoding_max)

    with tempfile.TemporaryDirectory() as tmp_dir:
        sim.export(tmp_dir, "quant_sim_model")
        with open(Path(tmp_dir, "quant_sim_model.encodings")) as json_file:
            encoding_data = json.load(json_file)

            param_encodings = encoding_data["param_encodings"]
            for encodings in param_encodings.values():
                for encoding_info in encodings:
                    encoding_min = encoding_info["min"]
                    encoding_max = encoding_info["max"]
                    scale = encoding_info["scale"]
                    offset = encoding_info["offset"]

                # We are enforcing strict symmetric flag to True in case of Range Learning. So encoding_min == -encoding_max.
                if quant_scheme in RANGE_LEARNING_SCHEMES:
                    assert encoding_min == -encoding_max
                    assert offset == -127
                    assert np.isclose(encoding_max, encoding_min + scale * 254, atol=1e-6)
                else:
                    # In post training scheme case, it doesn't seem to match exactly due to floating point arithmetic
                    assert np.isclose(encoding_min, -encoding_max - scale)
                    assert np.isclose(encoding_max, encoding_min + scale * 255, atol=1e-6)
                    assert offset == -128
                assert np.isclose(encoding_min, scale * offset, atol=1e-6)


def test_quant_scheme_percentile():
    """
    This test case ensures that the quantization is working fine with percentile scheme
    :return:
    """
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = dense_functional()

        qsim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, default_param_bw=16,
                                    default_output_bw=16)
        _, _, output_quantizers = qsim._get_quantizer_list()
        with pytest.raises(RuntimeError):
            for quantizer in output_quantizers:
                quantizer.set_percentile_value(99.99)

        for quantizer in output_quantizers:
            quantizer.quant_scheme = QuantScheme.post_training_percentile
            quantizer.set_percentile_value(99.99)
            assert np.allclose(quantizer.get_percentile_value(), 99.99)


def test_quant_scheme_percentile_setting_using_str():
    """
    This test case ensures that the quantization is working fine with percentile scheme
    :return:
    """
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = dense_functional()

        qsim = QuantizationSimModel(model, quant_scheme="percentile", default_param_bw=16, default_output_bw=16)
        inp_quatizer, paramater_quantizer, output_quantizers = qsim._get_quantizer_list()

        for quantizer in inp_quatizer + paramater_quantizer + output_quantizers:
            assert quantizer.quant_scheme == QuantScheme.post_training_percentile

def test_multi_output_model():
    """
    Test Quantsim with a model that has multi output layers
    """

    inputs = tf.keras.Input(shape=(480, 1088, 3))
    x = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    )(inputs)

    # tf operators -> TFOpLambda
    x = tf.reshape(x, [1, -1, 2])
    x, y = tf.split(x, 2, axis=2)
    x = tf.concat([x, y], axis=2)

    output_1 = tf.keras.layers.Dense(units=32)(x)
    output_2 = tf.keras.layers.Dense(units=32)(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[output_1, output_2])

    sim = QuantizationSimModel(model)

    # Check ConnectedGraph
    assert sim.connected_graph._split_count == 1
    for idx in range(2):
        assert sim.connected_graph._name_to_layer[f"tf.split/split:{idx}"].get_config()["name"] == "tf.split"

    assert len(sim.model.layers[3].output_quantizers) == 2

    sim.compute_encodings(
        lambda m, _: m(tf.random.uniform(shape=(1, *model.input_shape[1:]))), None
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        sim.export(tmp_dir, "multi_output_model")

def test_quantsim_with_separable_conv():
    """Tests Quantsim with seperable conv layer without model preparer """

    tf.keras.backend.clear_session()
    input_shape = (2, 3, 3, 2)
    inp = tf.keras.layers.Input(shape=input_shape[1:])
    test_inp = np.random.rand(*input_shape)

    separable_conv = tf.keras.layers.SeparableConv2D(
        2,
        kernel_size=2,
        depthwise_initializer=tf.initializers.Constant([5.0]),
        bias_initializer=tf.initializers.Constant([5.0])
    )(inp)
    model = tf.keras.models.Model(inputs=[inp], outputs=[separable_conv])

    _ = model(test_inp)

    with pytest.raises(AssertionError) as exc_info:
        sim = QuantizationSimModel(model, quant_scheme='tf_enhanced', rounding_mode='nearest',
                                   default_output_bw=8, default_param_bw=8)

    assert str(exc_info.value) == 'SeparableConv2D found in the model. Please run model preparer before calling ' \
                                  'QuantizationSimModel'

def test_quantizable_lstm_basic():
    batch_size = 1
    timesteps = 10
    features = 16
    units = 32

    # STAGE 1 MODEL - Functional model created with layers.lstm
    stage_1_inputs = keras.Input(shape=(timesteps, features))
    stage_1_output = keras.layers.LSTM(units, unroll=True)(stage_1_inputs)
    stage_1_model = keras.Model(inputs=stage_1_inputs, outputs=stage_1_output)
    
    # STAGE 2 MODEL - Sequential model created with layers.lstm
    stage_2_model = keras.models.Sequential(
        [
            keras.layers.LSTM(units, input_shape=(timesteps, features), unroll=True),
        ]
    )

    # STAGE 3 MODEL - Quantized model created through QuantSim functional original
    stage_3_model = QuantizationSimModel(stage_1_model)
    
    # STAGE 4 MODEL - Quantized model created through QuantSim sequential original
    stage_4_model = QuantizationSimModel(stage_2_model)

    input_data = np.ones([batch_size, timesteps, features])

    output_1_tensor = stage_1_model(input_data)
    output_2_tensor = stage_2_model(input_data)
    output_3_tensor = stage_3_model.model(input_data)
    output_4_tensor = stage_4_model.model(input_data)

    # check that all output tensors have the same shape
    assert output_1_tensor.shape == output_3_tensor.shape
    assert output_2_tensor.shape == output_4_tensor.shape

    # check that QcQuantizableMultiHeadAttention does not exist in original model.layers
    assert not any(isinstance(layer, QuantizedLSTM) for layer in stage_1_model.layers)
    assert not any(isinstance(layer, QuantizedLSTM) for layer in stage_2_model.layers)

    # check that QcQuantizableMultiHeadAttention exists in QuantSim model.layers
    assert any(isinstance(layer, QuantizedLSTM) for layer in stage_3_model.model.layers)
    assert any(isinstance(layer, QuantizedLSTM) for layer in stage_4_model.model.layers)

def test_quantizable_lstm_export_encodings():
    batch_size = 1
    timesteps = 10
    features = 16
    units = 32

    # STAGE 1 MODEL - Functional model created with layers.lstm
    stage_1_inputs = keras.Input(shape=(timesteps, features))
    stage_1_output = keras.layers.LSTM(units, unroll=True)(stage_1_inputs)
    stage_1_model = keras.Model(inputs=stage_1_inputs, outputs=stage_1_output)
    
    # STAGE 2 MODEL - Sequential model created with layers.lstm
    stage_2_model = keras.models.Sequential(
        [
            keras.layers.LSTM(units, input_shape=(timesteps, features), unroll=True),
        ]
    )

    rng = np.random.default_rng(seed=42)
    rn_input = rng.random([batch_size, timesteps, features])

    # STAGE 3 MODEL - Quantized model created through QuantSim functional original
    stage_3_model = QuantizationSimModel(stage_1_model)   

    stage_3_model.compute_encodings(lambda m, _: m(rn_input), None)
    stage_3_model.export('./data', 'lstm_functional')

    # STAGE 4 MODEL - Quantized model created through QuantSim sequential original
    stage_4_model = QuantizationSimModel(stage_2_model)
   
    stage_4_model.compute_encodings(lambda m, _: m(rn_input), None)
    stage_4_model.export('./data', 'lstm_sequential')
   
    #Check QuantizationSimModel model created through originals
    with open("./data/lstm_functional.encodings", "r") as encodings_file_functional, \
        open("./data/lstm_sequential.encodings", "r") as encodings_file_sequential:
        encodings_functional = json.load(encodings_file_functional)
        encodings_sequential = json.load(encodings_file_sequential)

    for (model_layer, encodings) in (stage_3_model.model.layers[1], encodings_functional), (stage_4_model.model.layers[0], encodings_sequential):
        for wrapper in model_layer._wrapped_layers:
            #LSTM input is passing through the wrapper as is. Output and input encoding will be same.
            for o_quantizer in wrapper.output_quantizers:
                if o_quantizer.encoding is not None:
                    tensor_name = wrapper.name + ":0"
                    encoding_dict = QuantizationSimModel._get_encoding_dict_for_quantizer(o_quantizer)
                    assert tensor_name in encodings['activation_encodings']
                    assert encodings['activation_encodings'][tensor_name] == encoding_dict
            for idx, param_quantizer in enumerate(wrapper.param_quantizers):
                if param_quantizer.encoding is not None:
                    param_name = wrapper._layer_to_wrap.weights[idx].name
                    encoding_dict = QuantizationSimModel._get_encoding_dict_for_quantizer(param_quantizer)
                    assert param_name in encodings['param_encodings']
                    assert encodings['param_encodings'][param_name] == encoding_dict

def test_quantsim_export_to_1_0_0():
    @contextlib.contextmanager
    def _swap_encoding_version():
        old_version = quantsim.encoding_version
        quantsim.encoding_version = '1.0.0'

        yield

        quantsim.encoding_version = old_version

    model = dense_functional()
    rand_inp = np.random.randn(100, 5)


    qsim = QuantizationSimModel(model, quant_scheme='tf')
    qsim.compute_encodings(lambda m, _: m(rand_inp), None)

    with tempfile.TemporaryDirectory() as temp_dir, _swap_encoding_version():
        assert quantsim.encoding_version == '1.0.0'
        qsim.export(temp_dir, 'test_export')
        assert quantsim.encoding_version == '1.0.0'

        with open(os.path.join(temp_dir, 'test_export.encodings'), 'r') as encoding_file:
            encodings = json.load(encoding_file)
        assert encodings['version'] == '0.6.1'
