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
"""
Models for use in unit testing
"""
import tensorflow as tf


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


def single_residual(num_classes=10):
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
            tf.keras.layers.Conv2D(
                16, kernel_size=2, strides=2, padding="same", use_bias=False
            ),
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


def lambda_model():
    """ Model using TF functional calls """

    inputs = tf.keras.Input(batch_size= 1, shape=(32, 32, 3))

    conv1 = tf.keras.layers.Conv2D(16, kernel_size=3)(inputs)
    conv2 = tf.keras.layers.Conv2D(16, kernel_size=3)(inputs)
    x = tf.concat([conv1, conv2], axis=1)
    shape = tf.shape(x)
    
    y = [tf.multiply(2, s) for s in shape]
    y = y[0]  # getitem
    y = tf.cast(y, tf.float32)
    y = tf.reshape(y, [1,-1])
    x = tf.add(x, y)

    return tf.keras.Model(inputs=inputs, outputs=x)


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


def transposed_conv_model():
    """
    Transposed Conv Model
    """

    inp = tf.keras.Input((4, 4, 10))
    x = tf.keras.layers.Conv2DTranspose(10, 3)(inp)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(10, 3)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return tf.keras.Model(inputs=[inp], outputs=[x])


def resnet_34():
    """
    ResNet34 Model

    Taken from here https://github.com/safwankdb/ResNet34-TF2/blob/master/model.py
    """

    class ResBlock(tf.keras.Model):
        def __init__(self, channels, stride=1, name=''):
            super(ResBlock, self).__init__(name=name)
            self.flag = (stride != 1)
            self.channels = channels
            self.stride = stride
            self.conv1 = tf.keras.layers.Conv2D(channels, 3, stride, padding='same')
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.conv2 = tf.keras.layers.Conv2D(channels, 3, padding='same')
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.relu = tf.keras.layers.ReLU()
            self.relu_2 = tf.keras.layers.ReLU()
            if self.flag:
                self.bn3 = tf.keras.layers.BatchNormalization()
                self.conv3 = tf.keras.layers.Conv2D(channels, 1, stride)

        def get_config(self):
            config = super().get_config()
            config.update({
                'channels': self.channels,
                'stride': self.stride,
                'name': self.name
            })
            return config

        def call(self, x):
            x1 = self.conv1(x)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x1 = self.conv2(x1)
            x1 = self.bn2(x1)
            if self.flag:
                x = self.conv3(x)
                x = self.bn3(x)
            x1 = tf.keras.layers.add([x, x1])
            x1 = self.relu_2(x1)
            return x1

    class ResNet34(tf.keras.Model):
        def __init__(self):
            super(ResNet34, self).__init__(name='ResNet34')
            self.conv1 = tf.keras.layers.Conv2D(64, 7, 2, padding='same')
            self.bn = tf.keras.layers.BatchNormalization()
            self.relu = tf.keras.layers.ReLU()
            self.mp1 = tf.keras.layers.MaxPooling2D(3, 2)

            self.conv2_1 = ResBlock(64, name='ResBlock')
            self.conv2_2 = ResBlock(64, name='ResBlock_1')
            self.conv2_3 = ResBlock(64, name='ResBlock_2')

            self.conv3_1 = ResBlock(128, 2, name='ResBlock_3')
            self.conv3_2 = ResBlock(128, name='ResBlock_4')
            self.conv3_3 = ResBlock(128, name='ResBlock_5')
            self.conv3_4 = ResBlock(128, name='ResBlock_6')

            self.conv4_1 = ResBlock(256, 2, name='ResBlock_7')
            self.conv4_2 = ResBlock(256, name='ResBlock_8')
            self.conv4_3 = ResBlock(256, name='ResBlock_9')
            self.conv4_4 = ResBlock(256, name='ResBlock_10')
            self.conv4_5 = ResBlock(256, name='ResBlock_11')
            self.conv4_6 = ResBlock(256, name='ResBlock_12')

            self.conv5_1 = ResBlock(512, 2, name='ResBlock_13')
            self.conv5_2 = ResBlock(512, name='ResBlock_14')
            self.conv5_3 = ResBlock(512, name='ResBlock_15')

            self.pool = tf.keras.layers.GlobalAveragePooling2D()
            self.fc1 = tf.keras.layers.Dense(512, activation='relu')
            self.dp1 = tf.keras.layers.Dropout(0.5)
            self.fc2 = tf.keras.layers.Dense(512, activation='relu')
            self.dp2 = tf.keras.layers.Dropout(0.5)
            self.fc3 = tf.keras.layers.Dense(64)

        def call(self, x):
            x = self.conv1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.mp1(x)

            x = self.conv2_1(x)
            x = self.conv2_2(x)
            x = self.conv2_3(x)

            x = self.conv3_1(x)
            x = self.conv3_2(x)
            x = self.conv3_3(x)
            x = self.conv3_4(x)

            x = self.conv4_1(x)
            x = self.conv4_2(x)
            x = self.conv4_3(x)
            x = self.conv4_4(x)
            x = self.conv4_5(x)
            x = self.conv4_6(x)

            x = self.conv5_1(x)
            x = self.conv5_2(x)
            x = self.conv5_3(x)

            x = self.pool(x)
            x = self.fc1(x)
            x = self.dp1(x)
            x = self.fc2(x)
            x = self.dp2(x)
            x = self.fc3(x)
            return x

    model = ResNet34()
    model.build(input_shape=(1, 480, 480, 3))
    return model


def model_with_sigmoid_and_softmax():
    """
    Functional model with sigmoid and softmax activations
    """
    input1 = tf.keras.layers.Input(shape=(1,))
    dense1 = tf.keras.layers.Dense(
        2, input_dim=2, use_bias=True
    )(input1)
    dense1_1 = tf.keras.layers.Activation('sigmoid')(dense1)
    dense2 = tf.keras.layers.Dense(
        1, activation=tf.keras.activations.relu, use_bias=True
    )(dense1_1)
    output = tf.keras.layers.Dense(
        1, use_bias=True
    )(dense2)
    output_1 = tf.keras.layers.Activation('softmax')(output)
    model = tf.keras.Model(inputs=[input1], outputs=[output_1])

    return model
