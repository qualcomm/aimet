# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=ungrouped-imports
# Including above pylint disables since pylint complains about certain module members not found, when they actually
# are there.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten, AvgPool2D, MaxPool2D
from packaging import version

if version.parse(tf.version.VERSION) < version.parse("2.00"):
    from tensorflow.contrib import slim


def transposed_conv2d_model():
    """ Trasposed Conv2D model"""
    inputs = tf.keras.Input(shape=(7, 7, 3))
    x = tf.keras.layers.Conv2DTranspose(3, (4, 4), use_bias=True)(inputs)
    return x


def instance_norm_model():
    """
    Function for Instance Norms
    """
    inputs = tf.keras.Input(shape=(16, 16, 3,))
    x = tf.keras.layers.Conv2D(16, (3, 3))(inputs)
    x = tf.contrib.layers.instance_norm(x)
    return x


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


def single_residual_for_tf2():
    """ Function for returning single residual model for TF2 framework, set trainable=False for BNs """

    inputs = tf.keras.Input(shape=(16, 16, 3,))
    x = tf.keras.layers.Conv2D(16, (3, 3))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65, trainable=False)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.MaxPool2D()(x)
    residual = x
    residual = tf.keras.layers.Conv2D(8, (1, 1))(residual)
    residual = tf.nn.relu(residual)
    x = tf.keras.layers.Conv2D(8, (1, 1))(x)
    x = tf.keras.layers.Conv2D(8, (1, 1))(x)
    x = tf.keras.layers.BatchNormalization(momentum=.4, epsilon=.25, trainable=False)(x)
    x = tf.add(x, residual)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(8, (3, 3))(x)
    x = tf.keras.layers.AvgPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="single_residual")(x)
    return outputs


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


def keras_model_functional():
    """ Function for returning basic keras model defined functionally """
    is_training = tf.compat.v1.placeholder_with_default(tf.constant(True), shape=(), name='is_training')
    inputs = tf.keras.Input(shape=(32, 32, 3,))
    x = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(x, training=True)
    with tf.compat.v1.variable_scope("scope_1"):
        x = tf.keras.layers.Conv2D(16, (2, 2), activation=tf.nn.tanh)(x)
        x = tf.keras.layers.BatchNormalization(momentum=.4, epsilon=.25)(x, training=is_training)
        x = tf.keras.layers.Conv2D(8, (2, 2), activation=tf.nn.tanh)(x)
        x = tf.keras.layers.BatchNormalization(momentum=.5, epsilon=.35)(x, training=False)
        x = tf.keras.layers.Conv2D(4, (2, 2), activation=tf.nn.relu6)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="keras_model_functional")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def keras_model_functional_for_tf2():
    """ Function for returning basic keras model defined functionally """
    inputs = tf.keras.Input(shape=(32, 32, 3,))
    x = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(x, training=True)
    with tf.compat.v1.variable_scope("scope_1"):
        x = tf.keras.layers.Conv2D(16, (2, 2), activation=tf.nn.tanh)(x)
        x = tf.keras.layers.BatchNormalization(momentum=.4, epsilon=.25)(x, training=False)
        x = tf.keras.layers.Conv2D(8, (2, 2), activation=tf.nn.tanh)(x)
        x = tf.keras.layers.BatchNormalization(momentum=.5, epsilon=.35)(x, training=False)
        x = tf.keras.layers.Conv2D(4, (2, 2), activation=tf.nn.relu6)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="keras_model_functional")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def keras_model_functional_with_non_fused_batchnorms():
    """ Function for returning basic keras model defined functionally using non fused batchnorms"""
    is_training = tf.compat.v1.placeholder_with_default(tf.constant(True), shape=(), name='is_training')
    inputs = tf.keras.Input(shape=(32, 32, 3,))
    x = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65, fused=False)(x, training=True)
    with tf.compat.v1.variable_scope("scope_1"):
        x = tf.keras.layers.Conv2D(16, (2, 2), activation=tf.nn.tanh)(x)
        x = tf.keras.layers.BatchNormalization(momentum=.4, epsilon=.25, fused=False)(x, training=is_training)
        x = tf.keras.layers.Conv2D(8, (2, 2), activation=tf.nn.tanh)(x)
        x = tf.keras.layers.BatchNormalization(momentum=.5, epsilon=.35, fused=False)(x, training=False)
        x = tf.keras.layers.Conv2D(4, (2, 2), activation=tf.nn.relu6)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                    name="keras_model_functional_with_non_fused_batchnorms")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def keras_model_functional_with_non_fused_batchnorms_for_tf2():
    """ Function for returning basic keras model defined functionally using non fused batchnorms"""
    inputs = tf.keras.Input(shape=(32, 32, 3,))
    x = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65, fused=False)(x, training=True)
    with tf.compat.v1.variable_scope("scope_1"):
        x = tf.keras.layers.Conv2D(16, (2, 2), activation=tf.nn.tanh)(x)
        x = tf.keras.layers.BatchNormalization(momentum=.4, epsilon=.25, fused=False)(x, training=False)
        x = tf.keras.layers.Conv2D(8, (2, 2), activation=tf.nn.tanh)(x)
        x = tf.keras.layers.BatchNormalization(momentum=.5, epsilon=.35, fused=False)(x, training=False)
        x = tf.keras.layers.Conv2D(4, (2, 2), activation=tf.nn.relu6)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                    name="keras_model_functional_with_non_fused_batchnorms")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def keras_functional_conv_net():
    """ Function for returning basic keras functional conv net """
    inputs = tf.keras.layers.Input(shape=(28, 28, 3))
    x = tf.keras.layers.Conv2D(4, kernel_size=3, activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=3, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=3, activation=None)(x)
    outputs = tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def keras_sequential_conv_net():
    """ Function for returning basic keras sequential conv net """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 3)),
        tf.keras.layers.Conv2D(4, kernel_size=3, activation=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.AvgPool2D(),
        tf.keras.layers.Dense(10)
    ])
    return model


def tf_slim_basic_model(inp):
    """ Function for returning basic tf slim model """
    is_training = tf.compat.v1.placeholder_with_default(tf.constant(True), shape=(), name='is_training')
    net = slim.conv2d(inp, 32, [3, 3])
    net = slim.batch_norm(net, decay=.7, epsilon=.65, is_training=True)
    net = slim.conv2d(net, 16, [2, 2])
    net = slim.batch_norm(net, decay=.6, epsilon=.25, scale=True, is_training=is_training)
    net = slim.conv2d(net, 8, [2, 2])
    net = slim.batch_norm(net, decay=.5, epsilon=.35, scale=True, is_training=False)
    net = slim.conv2d(net, 4, [2, 2])
    net = slim.flatten(net)
    net = slim.fully_connected(net, num_outputs=10, activation_fn=tf.nn.softmax, scope="tf_slim_model")
    return net


def tf_slim_with_softmax(inp):
    """ Function for returning tf slim model ending in softmax """
    net = slim.conv2d(inp, 32, [3, 3])
    net = slim.batch_norm(net, decay=.7, epsilon=.65, is_training=False)
    net = slim.conv2d(net, 16, [2, 2])
    net = slim.batch_norm(net, decay=.6, epsilon=.25, is_training=False)
    net = slim.softmax(net)
    return net


def split_and_concat_model():
    """ Function for returning keras model with splits and concats """
    x = tf.keras.Input(shape=[224, 224, 3, ])
    # TODO: implement split for the following commented out method of splitting
    # y1 = x[:, :100, :, :]
    # y2 = x[:, 101:, :, :]
    y1, y2 = tf.split(x, [100, 124], 1)
    y1 = tf.nn.relu(y1)
    y2 = tf.keras.layers.BatchNormalization()(y2)
    z = tf.keras.layers.concatenate([y1, y2], axis=1)
    z = tf.keras.layers.Flatten()(z)
    output = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="split_and_concat_model")(z)
    return output


def concat_model():
    """ Function for returning model with concat """
    x = tf.keras.Input(shape=[10, 10, 3, ])
    x1 = tf.keras.layers.Conv2D(5, (2, 2))(x)
    x2 = tf.keras.layers.Conv2D(6, (2, 2))(x)
    x3 = tf.keras.layers.Conv2D(7, (2, 2))(x)
    z = tf.keras.layers.concatenate([x2, x1, x3], axis=-1)
    z1 = tf.keras.layers.Conv2D(10, (2, 2))(z)
    z2 = tf.keras.layers.Conv2D(10, (2, 2))(z)
    z = tf.add(z1, z2)
    z = tf.keras.layers.Flatten()(z)
    output = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="concat_model")(z)
    return output


def upsample_model():
    """ Returns a model that can be used to test inserting upsample ops """

    inputs = tf.keras.Input(shape=(16, 16, 3,))
    x = tf.keras.layers.Conv2D(8, (2, 2))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.MaxPool2D()(x)
    residual = x

    x = tf.keras.layers.Conv2D(8, (1, 1))(x)
    x = tf.keras.layers.Conv2D(8, (1, 1))(x)
    x = tf.keras.layers.BatchNormalization(momentum=.4, epsilon=.25)(x)
    x = tf.add(x, residual)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Conv2D(4, (1, 1))(x)
    x = tf.keras.layers.AvgPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="upsample_model")(x)
    return outputs


def upsample_model_for_tf2():
    """ Returns a model that can be used to test inserting upsample ops """

    inputs = tf.keras.Input(shape=(16, 16, 3,))
    x = tf.keras.layers.Conv2D(8, (2, 2))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65, trainable=False)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.MaxPool2D()(x)
    residual = x

    x = tf.keras.layers.Conv2D(8, (1, 1))(x)
    x = tf.keras.layers.Conv2D(8, (1, 1))(x)
    x = tf.keras.layers.BatchNormalization(momentum=.4, epsilon=.25, trainable=False)(x)
    x = tf.add(x, residual)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Conv2D(4, (1, 1))(x)
    x = tf.keras.layers.AvgPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="upsample_model")(x)
    return outputs


def dropout_keras_model():
    """ Returns a keras model with a dropout module (also tests identity op, which is how dropout is represented when
    model is in inference mode """

    inputs = tf.keras.Input(shape=(10, 10, 3,))
    x = tf.keras.layers.Conv2D(16, (3, 3))(inputs)
    x = tf.keras.layers.Dropout(rate=.4)(x)
    x = tf.identity(x)
    x = tf.keras.layers.Conv2D(8, (2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="dropout_keras_model")(x)
    return outputs


def dropout_slim_model():
    """ Returns a slim model with a dropout module (also tests identity op, which is how dropout is represented when
    model is in inference mode """

    inputs = tf.keras.Input(shape=(10, 10, 3,))
    x = slim.conv2d(inputs, 16, [3, 3])
    x = slim.dropout(x, keep_prob=.6)
    x = tf.identity(x)
    x = slim.conv2d(x, 8, [2, 2])
    x = slim.flatten(x)
    outputs = slim.fully_connected(x, num_outputs=10, activation_fn=tf.nn.softmax, scope="dropout_slim_model")
    return outputs


def model_with_postprocessing_nodes():
    """ Returns a model with postprocessing nodes, that is expected to break mask propagation if included in
    connected graph """

    inputs = tf.keras.Input(shape=(10, 10, 3,))
    x = tf.keras.layers.Conv2D(16, (3, 3))(inputs)
    _ = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(8, (2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="model_with_postprocessing_nodes")(x)

    labels_placeholder = tf.compat.v1.placeholder_with_default(tf.constant(1, shape=(1, 10)),
                                                               shape=[None, 10],
                                                               name='labels')

    confidences = tf.nn.softmax(x, axis=1, name='confidences')
    categorical_preds = tf.argmax(confidences, axis=1, name='categorical_preds')
    categorical_labels = tf.argmax(labels_placeholder, axis=1, name='categorical_labels')
    correct_predictions = tf.equal(categorical_labels, categorical_preds)
    _ = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='top1-acc')
    _ = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=confidences,
                                              targets=tf.cast(categorical_labels, tf.int32),
                                              k=5), tf.float32), name='top5-acc')


def pad_model():
    """ Returns a model with various pad modules """

    inputs = tf.keras.Input(shape=(10, 10, 3,))
    x = tf.keras.layers.Conv2D(16, (1, 1))(inputs)
    x = tf.pad(x, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]))  # pylint: disable=no-value-for-parameter
    x = tf.keras.layers.Conv2D(8, (2, 2))(x)
    x = tf.pad(x, tf.constant([[0, 0], [1, 1], [1, 1], [1, 1]]))  # pylint: disable=no-value-for-parameter
    x = tf.keras.layers.Conv2D(8, (2, 2))(x)
    x = tf.pad(x, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), constant_values=2)
    x = tf.keras.layers.Conv2D(8, (2, 2))(x)
    x = tf.pad(x, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), mode='SYMMETRIC')  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    x = tf.keras.layers.Conv2D(8, (2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="pad_model")(x)
    return outputs


def depthwise_conv2d_model():
    """ Returns a model with depthwise conv2d """

    inputs = tf.keras.Input(shape=(10, 10, 3,))
    x = tf.keras.layers.Conv2D(16, (1, 1))(inputs)
    x = tf.keras.layers.SeparableConv2D(10, (2, 2))(x)
    x = tf.keras.layers.DepthwiseConv2D(3, (1, 1))(x)
    x = tf.keras.layers.Conv2D(8, (1, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="depthwise_conv2d_model")(x)
    return outputs


def multiple_input_model():
    """ Returns a model with multiple inputs """

    input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
    input2 = tf.keras.Input(name='input2', shape=(12, 12, 3))
    x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a')(input1)
    x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b')(input2)
    x = tf.keras.layers.add([x1, x2])
    x = tf.keras.layers.Conv2D(4, (1, 1), name='conv2')(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="multiple_input_model")(x)

    return outputs


def minimum_maximum_model():
    """ Returns a model with minimum and maximum operations """
    inputs = tf.keras.Input(shape=(32, 32, 3,))
    x = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(x, training=False)
    x = tf.minimum(x, .2)
    x = tf.maximum(x, .5)
    x = tf.keras.layers.Conv2D(16, (2, 2), activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="minimum_maximum_model")(x)
    return outputs


def model_with_upsample_already_present():
    """ Function for returning basic keras model defined functionally """
    inputs = tf.keras.Input(shape=(8, 8, 3,))
    x = tf.keras.layers.Conv2D(8, (3, 3))(inputs)
    with tf.name_scope("upsample"):
        unstacked = tf.unstack(x, axis=-1)
        zeros = tf.zeros_like(unstacked[0])
        current_index = 0

        for index in [0, 3, 4, 6, 7, 8, 9, 11]:
            while current_index < index:
                unstacked.insert(current_index, zeros)
                current_index += 1
            current_index += 1

        stack = tf.stack(unstacked, axis=-1)
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(stack, training=False)
    x = tf.keras.layers.Conv2D(4, (2, 2), activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="model_with_upsample_already_present")(x)
    return outputs


def model_with_multiple_downsamples():
    """ Function returning model containing multiple downsamples, in different name scopes """
    inputs = tf.keras.Input(shape=(8, 8, 10,))
    with tf.name_scope('downsample'):
        gather_1 = tf.gather(inputs, [1, 2, 3, 4, 5, 6, 7, 8], axis=-1)
        conv2d = tf.keras.layers.Conv2D(16, [2, 2])(gather_1)
        # gather_2 will be in same downsample namescope, but named GatherV2_1
        gather_2 = tf.gather(conv2d, [1, 2, 3, 4, 5, 6], axis=-1)
    with tf.name_scope('downsample'):
        conv2d_1 = tf.keras.layers.Conv2D(16, [2, 2])(gather_2)
        # gather_3 will be in a different namescope, downsample_1
        gather_3 = tf.gather(conv2d_1, [1, 2, 3, 4], axis=-1)
        conv2d_2 = tf.keras.layers.Conv2D(16, [2, 2])(gather_3)
    x = tf.keras.layers.Flatten()(conv2d_2)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="multiple_downsamples")(x)
    return outputs


def model_with_multiple_training_tensors():
    """ Return model with multiple batchnorms using different training tensors """

    inputs = tf.keras.Input(shape=(32, 32, 3,))
    # Should create a keras learning phase tensor
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(inputs)
    is_training = tf.compat.v1.placeholder_with_default(False, [], name='is_training')
    # Should attach to second is_training tensor
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(x, training=is_training)
    is_training_2 = tf.compat.v1.placeholder_with_default(False, [], name='is_training_2')
    # Should attach to third is_training_2 tensor
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(x, training=is_training_2)
    # Should not have a training tensor
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(x, training=True)
    # Should not have a training tensor
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(x, training=False)
    return x


def model_with_three_convs():
    """ Return model with three conv layers """

    inputs = tf.keras.Input(shape=(8, 8, 3,))
    x = tf.keras.layers.Conv2D(8, (2, 2))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65, trainable=False)(x)
    x = tf.keras.layers.Conv2D(6, (2, 2))(x)
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65, trainable=False)(x)
    x = tf.keras.layers.Conv2D(4, (2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="three_convs")(x)
    return outputs


def model_with_upsample2d():
    """ Return model with upsample2D op """
    inputs = tf.keras.Input(shape=(8, 8, 3,))
    x = tf.keras.layers.Conv2D(8, (2, 2))(inputs)
    x = tf.keras.layers.UpSampling2D(size=(2, 3))(x)
    x = tf.keras.layers.Conv2D(4, (2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="model_with_upsample2d")(x)
    return outputs


def model_with_leaky_relu():
    """ Return model with leaky relu op """
    inputs = tf.keras.Input(shape=(8, 8, 3,))
    x = tf.keras.layers.Conv2D(8, (2, 2))(inputs)
    x = tf.nn.leaky_relu(x, alpha=.4)
    x = tf.keras.layers.Conv2D(4, (2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="model_with_leaky_relu")(x)
    return outputs


def model_with_global_max_pool2d():
    """ Return model with global max pool op """
    inputs = tf.keras.Input(shape=(8, 8, 3,))
    x = tf.keras.layers.Conv2D(8, (2, 2))(inputs)
    x = tf.keras.layers.GlobalMaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="model_with_global_max_pool2d")(x)
    return outputs


def model_to_test_downstream_masks():
    """ Return model to test setting downstream masks for """
    inputs = tf.keras.Input(shape=(8, 8, 3,))
    x = tf.keras.layers.Conv2D(8, (2, 2), activation=tf.nn.relu)(inputs)
    residual = x
    x = tf.keras.layers.Conv2D(8, (1, 1))(x)
    x = x + residual
    x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(x)
    x1 = tf.nn.relu(x)
    x2 = tf.nn.relu(x)
    x1 = tf.keras.layers.Conv2D(4, (2, 2))(x1)
    x2 = tf.keras.layers.Conv2D(4, (2, 2))(x2)
    x = x1 + x2
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="model_to_test_downstream_masks")(x)
    return outputs


def model_with_dtype_int():
    """ Function containing dtype int32 op """

    input_1 = tf.keras.Input(shape=(8, 8, 3,), dtype=tf.int32)
    input_2 = tf.keras.Input(shape=(8, 8, 3,), dtype=tf.float32)
    x = tf.cast(input_1, tf.float32)
    x = tf.add(x, input_2)
    x = tf.keras.layers.Conv2D(8, (2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="model_with_dtype_int")(x)
    return outputs
