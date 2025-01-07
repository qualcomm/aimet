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

""" Unit tests for keras model preparer """
import logging
import os
import tempfile
from typing import List

import pytest
import numpy as np
import tensorflow as tf

from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.connectedgraph import ConnectedGraph
from aimet_tensorflow.keras.model_preparer import prepare_model, _KerasModelPreparer

get_models_custom_objects = _KerasModelPreparer._get_models_custom_objects
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops, get_all_output_ops

from test_models_keras import resnet_34, tiny_conv_net

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
AimetLogger.set_level_for_all_areas(logging.DEBUG)


# Begin of Subclass Models to for Testing
class TwoConvs(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TwoConvs, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(32,
                                           kernel_size=(3, 3),
                                           activation="relu",
                                           name="class_conv")

        self.conv_transpose = tf.keras.layers.Conv2DTranspose(64,
                                                              kernel_size=(3, 3),
                                                              activation="relu",
                                                              name="class_conv_transpose")

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.conv_transpose(x)
        return x


class ConvTimesThree(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConvTimesThree, self).__init__(**kwargs)
        self.depth_conv = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1,
                                                          kernel_size=(3, 3),
                                                          activation="relu",
                                                          name="class_conv_depth")
        self.two_convs = TwoConvs()

    def call(self, x, **kwargs):
        return self.depth_conv(self.two_convs(x))


###########################################################################################################
#                                                                                                         #
# Below models are based on Deep Learning with Python by Francois Chollet Second Edition (page 182 - 185) #
#                                                                                                         #
###########################################################################################################

# Only Subclassing
class CustomerTicketModel(tf.keras.Model):
    def __init__(self, num_departments):
        super().__init__()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.mixing_layer = tf.keras.layers.Dense(64, activation="relu")
        self.priority_scorer = tf.keras.layers.Dense(1, activation="sigmoid")
        self.department_classifier = tf.keras.layers.Dense(num_departments, activation="softmax")

    def call(self, inputs, **kwargs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department


# Functional model that includes subclassed layers
class Classifier(tf.keras.Model):
    def __init__(self, num_classes=4):
        super().__init__()
        if num_classes == 2:
            num_units = 1
            activation = "sigmoid"
        else:
            num_units = num_classes
            activation = "softmax"
        self.dense = tf.keras.layers.Dense(num_units, activation=activation)

    def call(self, inputs, **kwargs):
        return self.dense(inputs)


# Linear class source: https://keras.io/guides/making_new_layers_and_models_via_subclassing/
class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weight",
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )

        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class KerasDefinedAndAcceptableDefined(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__(name="keras_defined_and_acceptable_defined")
        self.conv = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), )
        self.linear = Linear()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.linear(x)
        x = self.relu(x)
        return x


# Text classification Transformer model from https://keras.io/examples/nlp/text_classification_with_transformer/
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim), ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training, **kwargs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x, **kwargs):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        x = x + positions
        return x


# Layer with multiple math operations in call
class MultiMathOperations(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MultiMathOperations, self).__init__(**kwargs)
        self.r = tf.constant(np.random.rand(1, 1), dtype=tf.float32)
        self.g = tf.constant(np.random.rand(1, 1), dtype=tf.float32)
        self.b = tf.constant(np.random.rand(1, 1), dtype=tf.float32)

    def call(self, inputs, **kwargs):
        r = tf.math.multiply(inputs[:, :, :, 0:1], self.r)
        g = tf.math.multiply(inputs[:, :, :, 1:2], self.g)
        b = tf.math.multiply(inputs[:, :, :, 2:3], self.b)
        return tf.concat([r, g, b], axis=3)


# Layer with multiple inputs
class MultiInputLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(name="multi_input_layer")

    def call(self, inputs, **kwargs):
        input1 = inputs[0]
        input2 = inputs[1]
        return input1 + input2


# Layer with multiple outputs
class MultiOutLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(name="multi_out_layer")

    def call(self, inputs, **kwargs):
        out1 = inputs * 2.0
        out2 = inputs * 3.0
        return [out1, out2]


class TestModelPreparer:
    """ Class for Testing aimet_tensorflow.keras Model Preparer"""

    # Model Getters
    @staticmethod
    def get_conv_sub_class():
        input_shape = (128, 28, 28, 1)
        inp = tf.keras.Input(batch_shape=input_shape)
        x = ConvTimesThree()(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.5)(x, training=False)
        x = tf.keras.layers.Dense(10, activation="softmax")(x)

        model = tf.keras.Model(inputs=inp, outputs=x, name="conv_classes")
        return model

    @staticmethod
    def get_functional_model_with_subclassed_layers():
        inputs = tf.keras.layers.Input(shape=(3,))
        features = tf.keras.layers.Dense(64, activation="relu")(inputs)
        outputs = Classifier(num_classes=10)(features)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def get_subclass_model_with_functional_layers():
        inputs = tf.keras.Input(shape=(64,))
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(inputs)
        binary_classifier = tf.keras.Model(inputs=inputs, outputs=outputs)

        class MyFunctionalModel(tf.keras.Model):
            def __init__(self):
                super().__init__(name="my_functional_model")
                self.dense = tf.keras.layers.Dense(64, activation="relu")
                self.classifier = binary_classifier

            def call(self, inputs, **kwargs):
                features = self.dense(inputs)
                return self.classifier(features)

        model = MyFunctionalModel()
        return model

    @staticmethod
    def get_keras_text_classification_example_model_and_data_input():
        # Download and prepare dataset
        vocab_size = 20000  # Only consider the top 20k words
        maxlen = 200  # Only consider the first 200 words of each movie review

        random_input = np.random.random((10, 200))

        # Create classifier model using transformer layer
        embed_dim = 32  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer

        inputs = tf.keras.layers.Input(shape=(maxlen,))
        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(20, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs, name="text_classification")

    @staticmethod
    def get_multi_math_op_model():
        input_layer = tf.keras.Input(shape=(32, 32, 1))
        output = MultiMathOperations()(input_layer)
        return tf.keras.Model(inputs=input_layer, outputs=output)

    @staticmethod
    def get_model_with_multiple_inputs():
        input1 = tf.keras.Input(shape=(28, 28, 1), name="input_1")
        input2 = tf.keras.Input(shape=(28, 28, 1), name="input_2")
        x = tf.keras.layers.Conv2D(16, 3, activation="relu")(input1)
        y = tf.keras.layers.Conv2D(16, 3, activation="relu")(input2)
        outputs = MultiInputLayer()([x, y])

        return tf.keras.Model(inputs=[input1, input2], outputs=outputs, name="multi_input")

    @staticmethod
    def get_model_with_multiple_outputs(use_lambdas):
        inputs = tf.keras.Input(shape=(28, 28, 1), name="img")
        x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inputs)
        if use_lambdas:
            x0, x1 = tf.split(x, [8, 8], 3)
            x1a = x1 * 2.0
            outputs = tf.concat([x0, x1a], 3)
        else:
            outputs = MultiOutLayer()(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"multi_output_with_lambda_{use_lambdas}")

    @staticmethod
    def get_model_with_acceptable_subclass_layers():
        inputs = tf.keras.Input(shape=(32,))
        x = Linear()(inputs)
        x = tf.keras.layers.ReLU()(x)
        x = Linear()(x)
        outputs = tf.keras.layers.ReLU()(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs, name="acceptable_subclasses")

    @staticmethod
    def get_model_with_keras_and_acceptable_subclass_layers_defined():
        inputs = tf.keras.Input(shape=(28, 28, 1,))
        outputs = KerasDefinedAndAcceptableDefined()(inputs)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    # Helper Functions
    def get_model_and_set_random_input(self, get_model_func):
        original_model = get_model_func()
        if isinstance(original_model.input_shape, List):
            self.random_input = [
                np.random.rand(1, *input_shape[1:]) for input_shape in original_model.input_shape
            ]
        else:
            self.random_input = np.random.rand(1, *original_model.input_shape[1:])
        return original_model

    @staticmethod
    def get_model_inherited_from_Model_without_config() -> tf.keras.Model:
        class LayerWithoutConfig(tf.keras.Model):  # Note: Inherits from Model
            def __init__(self, value_to_save_in_config, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.dense = tf.keras.layers.Dense(value_to_save_in_config, 'relu')

            def call(self, inputs, training=None, mask=None):
                return self.dense(inputs)


        class MyTempModel(tf.keras.Model):
            def __init__(self):
                super(MyTempModel, self).__init__()
                self.layer_without_config = LayerWithoutConfig(3)

            def call(self, inputs, training=None, mask=None):
                return self.layer_without_config(inputs)

        model = MyTempModel()
        model.build(input_shape=(1, 28, 28, 3))
        return model

    # Tests
    @pytest.fixture(scope="function", autouse=True)
    def pass_prepared_model_through_quantsim(self):
        tf.keras.backend.clear_session()
        yield
        if not hasattr(self, "prepared_model"):
            return

        # TODO: multi_output_with_lambda currently fails in ConnectedGraph -> AIMET-2682
        if self.prepared_model.name.startswith("multi_output_with_lambda"):
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            sim = QuantizationSimModel(self.prepared_model)
            sim.compute_encodings(lambda model, _: model(self.random_input), None)
            sim.export(temp_dir, self.prepared_model.name, custom_objects=self.__dict__.get("custom_objects"))
        del self.prepared_model

    @pytest.mark.skip("Need to verify how functional model should look")
    def test_full_subclass_to_functional(self):
        vocabulary_size = 10000
        num_tags = 100
        num_departments = 4
        num_samples = 1280

        title_data = np.random.randint(0, 2, size=(1, num_samples, vocabulary_size))
        text_body_data = np.random.randint(0, 2, size=(1, num_samples, vocabulary_size))
        tags_data = np.random.randint(0, 2, size=(1, num_samples, num_tags))

        model = CustomerTicketModel(num_departments=num_departments)
        _ = model({"title": title_data,
                   "text_body": text_body_data,
                   "tags": tags_data})
        # Since this model is fully subclassed, specifically at the beginning, we call prepare model with
        # the inputs to have Keras symoblic tensor fit the rest of the layers correctly.
        input_layers = [tf.keras.Input(shape=(num_samples, vocabulary_size,), name="title"),
                        tf.keras.Input(shape=(num_samples, vocabulary_size,), name="text_body"),
                        tf.keras.Input(shape=(num_samples, num_tags,), name="tags")]
        _ = prepare_model(model, input_layers)

    @pytest.mark.parametrize("model_func_str", [
        "get_multi_math_op_model",
        "get_model_with_multiple_inputs",
        "get_keras_text_classification_example_model_and_data_input",
        "get_model_with_keras_and_acceptable_subclass_layers_defined",
        "get_model_with_multiple_outputs"
    ])
    def test_models_with_common_flow(self, model_func_str):
        model_func = self.__getattribute__(model_func_str)
        if model_func_str == "get_model_with_multiple_outputs":
            model_func = lambda: self.__getattribute__(model_func_str)(False)
        original_model = self.get_model_and_set_random_input(model_func)

        # NOTE: Verification of the model occurs in the prepare_model function.
        self.prepared_model = prepare_model(original_model)
        self.custom_objects = get_models_custom_objects(self.prepared_model)

    def test_functional_model_with_subclassed_layers_to_functional(self):
        original_model = self.get_model_and_set_random_input(self.get_functional_model_with_subclassed_layers)

        self.prepared_model = prepare_model(original_model)

        connected_graph = ConnectedGraph(self.prepared_model)
        assert "Unknown" not in connected_graph.ordered_ops

        product_dict = connected_graph.get_all_products()
        assert "Gemm_0_to_Gemm_1" in product_dict

        input_ops = get_all_input_ops(connected_graph)
        assert len(input_ops) == 1
        assert input_ops[0].get_module() == self.prepared_model.layers[1]
        assert isinstance(input_ops[0].get_module(), type(original_model.layers[1]))

        output_ops = get_all_output_ops(connected_graph)
        assert len(output_ops) == 1
        assert output_ops[0].get_module() == self.prepared_model.layers[-1]
        assert isinstance(output_ops[0].get_module(), type(original_model.layers[-1].dense))

    def test_input_layer_missing(self):
        model = self.get_subclass_model_with_functional_layers()
        input_shape = (32, 64)
        self.random_input = np.random.rand(*input_shape)
        _ = model(self.random_input)
        with pytest.raises(ValueError):
            prepare_model(model)

    def test_subclass_model_with_subclassed_layers_to_functional(self):
        original_model = self.get_subclass_model_with_functional_layers()
        input_shape = (32, 64)
        self.random_input = np.random.rand(*input_shape)
        _ = original_model(self.random_input)

        self.prepared_model = prepare_model(original_model, tf.keras.Input(shape=input_shape[1:]))

        connected_graph = ConnectedGraph(self.prepared_model)
        assert "Unknown" not in connected_graph.ordered_ops

        product_dict = connected_graph.get_all_products()
        assert "Gemm_0_to_Gemm_1" in product_dict

        input_ops = get_all_input_ops(connected_graph)
        assert len(input_ops) == 1
        assert input_ops[0].get_module() == self.prepared_model.layers[1]
        assert isinstance(input_ops[0].get_module(), type(original_model.layers[0]))

        output_ops = get_all_output_ops(connected_graph)
        assert len(output_ops) == 1
        assert output_ops[0].get_module() == self.prepared_model.layers[-1]
        assert isinstance(output_ops[0].get_module(), type(original_model.layers[-1].layers[-1]))

    def test_conv_times_three_subclass_to_functional(self):
        original_model = self.get_model_and_set_random_input(self.get_conv_sub_class)

        self.prepared_model = prepare_model(original_model)

        connected_graph = ConnectedGraph(self.prepared_model)
        assert "Unknown" not in connected_graph.ordered_ops

        product_dict = connected_graph.get_all_products()
        assert "Conv_0_to_ConvTranspose_1" in product_dict
        assert "ConvTranspose_1_to_Conv_2" in product_dict

        input_ops = get_all_input_ops(connected_graph)
        assert len(input_ops) == 1
        assert input_ops[0].get_module() == self.prepared_model.layers[1]
        assert isinstance(input_ops[0].get_module(), type(original_model.layers[1].two_convs.conv))

        output_ops = get_all_output_ops(connected_graph)
        assert len(output_ops) == 1
        assert output_ops[0].get_module() == self.prepared_model.layers[-1]
        assert isinstance(output_ops[0].get_module(), type(original_model.layers[-1]))

    def test_non_nested_layered_model(self):
        original_model = self.get_model_and_set_random_input(tiny_conv_net)

        self.prepared_model = prepare_model(original_model)

        assert original_model.get_config() == self.prepared_model.get_config(), \
            "Prepare model did not give back the original model. This model does not need to be prepared."

    def test_multi_output_with_lambdas(self):
        original_model = self.get_model_and_set_random_input(lambda: self.get_model_with_multiple_outputs(True))

        # NOTE: Verification of the model occurs in the prepare_model function.
        self.prepared_model = prepare_model(original_model)
        assert original_model.get_config() == self.prepared_model.get_config(), \
            "The original model does not contain any nested layers. The original model should be returned."

    def test_model_with_acceptable_subclass_layers(self):
        original_model = self.get_model_and_set_random_input(self.get_model_with_acceptable_subclass_layers)

        self.prepared_model = prepare_model(original_model)
        assert original_model.get_config() == self.prepared_model.get_config(), \
            "The original model contains acceptable subclass layers. The original model should be returned."
        self.custom_objects = get_models_custom_objects(self.prepared_model)

    def test_model_inherits_from_Model_without_get_config_defined(self):
        original_model = self.get_model_inherited_from_Model_without_config()

        with pytest.raises(TypeError):
            prepare_model(original_model, input_layer=tf.keras.Input(shape=[28, 28, 3]))

    def test_resnet_34(self):
        original_model = resnet_34()
        input_shape = (480, 480, 3)
        self.random_input = np.random.rand(1, *input_shape)
        _ = original_model(self.random_input)

        # NOTE: Verification of the model occurs in the prepare_model function.
        _ = prepare_model(original_model, input_layer=tf.keras.Input(shape=input_shape))
