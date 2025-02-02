# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Models for use in unit testing """
from typing import Optional, List, Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten, AvgPool2D, MaxPool2D, Input, Layer


def keras_model():
    """ Function for returning a basic keras model """

    model = Sequential([
        Conv2D(8, (2, 2), input_shape=(16, 16, 3,)),
        BatchNormalization(momentum=.3, epsilon=.65),
        AvgPool2D(),
        MaxPool2D(),
        BatchNormalization(momentum=.4, epsilon=.25),
        Conv2D(4, (2, 2), activation=tf.nn.tanh, kernel_regularizer=tf.keras.regularizers.l2(0.5)),
        Flatten(),
        Dense(2, activation='softmax', name="keras_model")])
    return model


def single_residual():
    """ Function for returning single residual model """

    inputs = tf.keras.Input(shape=(16, 16, 3,))
    x = tf.keras.layers.Conv2D(16, (3, 3))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.MaxPool2D()(x)
    residual = x
    residual = tf.keras.layers.Conv2D(8, (1, 1))(residual)
    residual = tf.nn.relu(residual)

    x = tf.keras.layers.Conv2D(8, (1, 1))(x)
    x = tf.keras.layers.Conv2D(8, (1, 1))(x)
    x = tf.keras.layers.BatchNormalization(momentum=.4, epsilon=.25)(x)
    x = tf.add(x, residual)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Conv2D(8, (3, 3))(x)
    x = tf.keras.layers.AvgPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="single_residual")(x)

    return outputs


def simple_sequential_with_input_shape():
    """
    Simple sequential model with input shape (Two layer MLP)
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(3,)),
            tf.keras.layers.Dense(5, activation=tf.nn.softmax),
        ]
    )


def simple_sequential_without_input_shape():
    """
    Simple sequential model without input shape (Two layer MLP)
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(4, activation=tf.nn.relu),
            tf.keras.layers.Dense(5, activation=tf.nn.softmax),
        ]
    )


def simple_functional():
    """
    Simple functional model (Two layer MLP)
    """
    inputs = tf.keras.Input(shape=(3,))
    x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def simple_subclassing():
    """
    Simple subclassing model (Two layer MLP)
    """

    class SimpleSubclassing(tf.keras.Model):
        """
        Two layer MLP implemented by subclassing
        """

        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
            self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

        def call(self, inputs):
            """
            Call forward pass
            """
            x = self.dense1(inputs)
            return self.dense2(x)

    return SimpleSubclassing()


def multi_input_subclassing():
    """
    Multi input subclassing model
    """
    class MultiInputSubclassing(tf.keras.Model):
        """
        Multi input network implemented by subclassing
        """

        def __init__(self):
            super(MultiInputSubclassing, self).__init__()
            self.dense1 = tf.keras.layers.Dense(4)
            self.dense2 = tf.keras.layers.Dense(5)
            self.dense3 = tf.keras.layers.Dense(6)

        def call(self, inputs):
            """
            Call forward pass
            """
            input1, input2 = inputs
            x = self.dense1(input1)
            y = self.dense2(input2)
            z = tf.keras.layers.concatenate([x, y])
            return self.dense3(z)

    return MultiInputSubclassing()


def residual_subclassing():
    """
    Residual connection subclassing model
    """
    class Residual(tf.keras.Model):
        """The Residual block"""

        def __init__(self):
            super().__init__()
            self.conv1 = tf.keras.layers.Conv2D(5, padding="same", kernel_size=5)
            self.conv2 = tf.keras.layers.Conv2D(3, kernel_size=3, padding="same")
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()

        def call(self, inputs):
            """
            Call forward pass
            """
            output = tf.keras.activations.relu(self.bn1(self.conv1(inputs)))
            output = self.bn2(self.conv2(output))
            output += inputs
            return tf.keras.layers.ReLU()(output)

    return Residual()


def concat_functional():
    """
    Functional model containing concat operation
    """
    input1 = tf.keras.layers.Input(shape=(1,))
    input2 = tf.keras.layers.Input(shape=(1,))
    x = tf.keras.layers.Dense(1)(input2)
    input3 = tf.keras.layers.Input(shape=(1,))
    y = tf.keras.layers.Dense(1)(input3)
    y = tf.keras.layers.Dense(1)(y)

    merged = tf.keras.layers.Concatenate(axis=1)([input1, x, y])
    dense1 = tf.keras.layers.Dense(
        2, input_dim=2, activation=tf.keras.activations.sigmoid, use_bias=True
    )(merged)
    output = tf.keras.layers.Dense(
        1, activation=tf.keras.activations.relu, use_bias=True
    )(dense1)
    model = tf.keras.models.Model(inputs=[input1, input2, input3], outputs=output)

    return model


def single_residual_model(num_classes=10):
    """
    Single residual model implemented by Functional style
    """
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(
        32, kernel_size=2, strides=2, padding="same", use_bias=False
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(strides=2, padding="same")(x)

    residual = x

    x = tf.keras.layers.Conv2D(
        16, kernel_size=2, strides=1, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        8, kernel_size=2, strides=1, padding="same", use_bias=False
    )(x)

    residual = tf.keras.layers.Conv2D(8, kernel_size=2, strides=1, padding="same")(
        residual
    )
    residual = tf.keras.layers.AveragePooling2D(strides=1, padding="same")(residual)

    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=3)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def nested_sequential_model(num_classes=3):
    """
    Nested sequential model implemented by Sequential style
    """
    inner_seq = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(16, kernel_size=2, strides=2, padding="same", use_bias=False, input_shape=(32, 32, 3)),
            tf.keras.layers.BatchNormalization(),
        ]
    )

    return tf.keras.Sequential(
        [
            inner_seq,
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(strides=1, padding="same"),
            tf.keras.layers.Conv2D(8, kernel_size=2, strides=1, padding="same"),
            tf.keras.layers.Conv2D(4, kernel_size=2, strides=1, padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes),
        ]
    )


def nested_functional_model():
    """
    Nested Functional model implemented by Functional style
    """
    def inner_block1(inp):
        blk = tf.keras.layers.Conv2D(
            16, kernel_size=2, strides=2, padding="same", use_bias=False
        )(inp)
        blk = tf.keras.layers.BatchNormalization()(blk)
        return blk

    def inner_block2(inp):
        blk = tf.keras.layers.MaxPool2D()(inp)
        blk = tf.keras.layers.BatchNormalization()(blk)
        return blk

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = inner_block1(inputs)
    x = tf.keras.layers.ReLU()(x)
    outputs = inner_block2(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def sequential_in_functional():
    """
    Sequential in Functional model
    """
    inner_seq = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, kernel_size=2, strides=2, padding="same", use_bias=False
            ),
            tf.keras.layers.BatchNormalization(),
        ]
    )

    outer_seq = tf.keras.Sequential(
        [inner_seq, tf.keras.layers.ReLU(), tf.keras.layers.BatchNormalization()]
    )

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = outer_seq(inputs)
    x = tf.keras.layers.ReLU()(x)
    outputs = tf.keras.layers.MaxPool2D()(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def tiny_conv_net():
    """
    Simple convolution network
    """
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(
        32, kernel_size=2, strides=2, padding="same", use_bias=False
    )(inputs)
    x = tf.keras.layers.BatchNormalization(beta_initializer="glorot_uniform", gamma_initializer="glorot_uniform")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(
        16, kernel_size=2, strides=1, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization(beta_initializer="glorot_uniform", gamma_initializer="glorot_uniform")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        8, kernel_size=2, strides=1, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=3)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def multi_input_and_multi_output_model() -> tf.keras.Model:
    """
    Functional model having multiple inputs and multiple outputs
    """
    input_a = tf.keras.Input(shape=(64,), name="foo")
    input_b = tf.keras.Input(shape=(128,), name="bar")

    x = tf.keras.layers.Dense(64, activation="relu")(input_a)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    output_a = tf.keras.layers.Dense(10, activation="softmax", name="output_foo")(x)

    y = tf.keras.layers.Dense(64, activation="relu")(input_b)
    y = tf.keras.layers.Dense(32, activation="relu")(y)
    output_b = tf.keras.layers.Dense(10, activation="softmax", name="output_bar")(y)

    output_concat = tf.keras.layers.Concatenate(name="output_baz")([output_a, output_b])
    model = tf.keras.Model(
        inputs=[input_a, input_b], outputs=[output_a, output_b, output_concat]
    )
    return model


def get_custom_objects_based_model(size, fill_value):
    """
    Sample Keras model with custom_objects
    """
    class ConstantOfShapeTF(Layer):
        """
        ConstantOfShape Layer
        """
        def __init__(self, name: str = 'const_of_shape_tf', **kwargs):
            super().__init__(**kwargs)
            self.op_name = name

        def get_config(self):
            """
            config required to serialize custom modules
            """
            config = super().get_config()
            config.update({
                "name": self.op_name,
            })
            return config

        def call(self, size, fill_value):
            """
            Forward pass call
            """
            return tf.fill(dims=size, value=fill_value, name=self.op_name)

    input_tensor = Input(shape=size)
    const_shape = ConstantOfShapeTF()
    const = const_shape(size, fill_value)
    output_tensor = input_tensor + const
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model, {'ConstantOfShapeTF': ConstantOfShapeTF}


def get_model_with_pack_unpack(input_shape, axis):

    class PackUnPack(tf.keras.layers.Layer):
        def __init__(self, axis: int, name: str = 'pack_tf', **kwargs):
            super().__init__(**kwargs)
            self.op_name = name
            self.axis = axis

        def get_config(self):
            config = super().get_config()
            config.update({"name": self.op_name, "axis": self.axis})
            return config

        def call(self, inputs):
            """
            Call Forward-pass
            """
            ip1, ip2, ip3 = inputs
            output = tf.stack([ip1, ip2, ip3], axis=self.axis, name=self.op_name)
            out1 = tf.unstack(output, axis=self.axis, name="un"+self.op_name)
            return output, out1

    input1, input2 = tf.keras.Input(shape=input_shape, batch_size=1), tf.keras.Input(shape=input_shape, batch_size=1)
    input3 = tf.keras.Input(shape=input_shape, batch_size=1)
    pack_unpack_layer = PackUnPack(axis)
    out_tensor = pack_unpack_layer((input1, input2, input3))
    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=out_tensor, name="pack_unpack_model")
    custom_object = {"PackUnPack": PackUnPack}

    return model, custom_object


def model_with_space_to_batch(input_shape, block_shape, pad_amount):
    """
    Sample Keras model with SpaceToBatch Op
    :param input_shape: Shape of input tensor
    :param block_shape: block_shape param to instantiate SpaceToBatch layer
    :param pad_amount: pad_amount param to instantiate SpaceToBatch layer
    :return: returns Keras model along with custom_objects
    """
    class SpaceToBatchTF(tf.keras.layers.Layer):
        """
        Keras Layer implementation of SpaceToBatch
        """
        def __init__(self, block_shape: list, pad_amount: list, name: str = 'space_to_batch_tf', **kwargs):
            super().__init__(**kwargs)
            self.block_shape = block_shape
            self.pad_amount = pad_amount if pad_amount is not None else [[0, 0], [0, 0]]
            self.op_name = name

        def get_config(self):
            config = super().get_config()
            config.update({
                "block_shape": self.block_shape,
                "pad_amount": self.pad_amount,
                "name": self.op_name,
            })
            return config

        def call(self, inputs, *args, **kwargs):
            """
            Forward-pass routine for SpaceToBatch Keras Layer
            """
            output = tf.space_to_batch(inputs, self.block_shape, self.pad_amount, name=self.op_name)
            return output

    input_tensor = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        32, kernel_size=3, strides=1, padding="same", use_bias=False
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = SpaceToBatchTF(block_shape, pad_amount)(x)
    x = tf.keras.layers.Conv2D(
        32, kernel_size=3, strides=1, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    outputs = x + 2

    model = tf.keras.Model(inputs=input_tensor, outputs=outputs)
    custom_objects = {'SpaceToBatchTF': SpaceToBatchTF}
    return model, custom_objects

def model_with_batch_to_space(input_shape, block_shape, crops):
    """
    Sample Keras model with BatchToSpace Op
    :param input_shape: Shape of input tensor
    :param block_shape: block_shape param to instantiate BatchToSpace layer
    :param crops: pad_amount param to instantiate BatchToSpace layer
    :return: returns Keras model along with custom_objects
    """
    class BatchToSpaceTF(tf.keras.layers.Layer):
        """
        Keras Layer implementation of BatchToSpace
        """
        def __init__(self, block_shape: list, crops: list, name: str = 'batch_to_space_tf', **kwargs):
            super().__init__(**kwargs)
            self.block_shape = block_shape
            self.crops = crops if crops is not None else [[0, 0], [0, 0]]
            self.op_name = name

        def get_config(self):
            config = super().get_config()
            config.update({
                "block_shape": self.block_shape,
                "crops": self.crops,
                "name": self.op_name,
            })
            return config

        def call(self, inputs, *args, **kwargs):
            """
            Forward-pass routine for BatchToSpace Keras Layer
            """
            output = tf.batch_to_space(inputs, self.block_shape, self.crops, name=self.op_name)
            return output

    input_tensor = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        32, kernel_size=3, strides=1, padding="same", use_bias=False
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = BatchToSpaceTF(block_shape, crops)(x)
    x = tf.keras.layers.Conv2D(
        32, kernel_size=3, strides=1, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    outputs = x + 2

    model = tf.keras.Model(inputs=input_tensor, outputs=outputs)
    custom_objects = {'BatchToSpaceTF': BatchToSpaceTF}
    return model, custom_objects


def get_model_with_moments(input_shape, axes, keep_dims):

    class Moments(tf.keras.layers.Layer):

        def __init__(self, axes: int, keep_dims: bool, **kwargs):
            super().__init__(**kwargs)
            self.axes = axes
            self.keep_dims = keep_dims

        def get_config(self):
            config = super().get_config()
            config.update({"axes": self.axes, "keep_dims": self.keep_dims})
            return config

        def call(self, inputs):
            """
            Call Forward-pass
            """
            y = tf.nn.moments(inputs, self.axes, shift=None, keepdims=self.keep_dims, name=None)
            return y

    input1 = tf.keras.Input(shape=input_shape)
    moments = Moments(axes=axes, keep_dims=keep_dims)
    p = moments(input1)
    model = tf.keras.Model(inputs=input1, outputs=p, name="moments")
    custom_object = {"Moments": Moments}

    return model, custom_object


def get_model_with_crop_and_resize(batch_size, image_shape, crop_size, interpolation_mode, extrapolation_value):
    """
    Sample Keras model with tf.image.crop_and_resize
    """
    input_1 = tf.keras.Input(shape=image_shape, batch_size=batch_size)
    input_2 = tf.keras.Input(shape=(4))
    input_3 = tf.keras.Input(shape=(), dtype=tf.int32)

    x = tf.image.crop_and_resize(input_1, input_2, input_3, crop_size, interpolation_mode, extrapolation_value)
    x = tf.keras.layers.Conv2D(
        32, kernel_size=3, strides=1, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    output_tensor = x + 2

    model = tf.keras.Model(inputs = (input_1, input_2, input_3), outputs=output_tensor)
    return model, None


def get_model_with_data_movement_custom_ops(batch_size, image_shape, block_shape, crops_pad_amount):
    input_1 = tf.keras.Input(shape=image_shape, batch_size=batch_size)

    x = tf.keras.layers.Conv2D(
        32, kernel_size=3, strides=1, padding="same", use_bias=False
    )(input_1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.batch_to_space(x, block_shape, crops_pad_amount)
    x = tf.space_to_batch(x, block_shape, crops_pad_amount)
    output_tensor = x + 2

    model = tf.keras.Model(inputs=input_1, outputs=output_tensor)
    return model

