# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
This file contains unit tests for testing batch norm folding
"""
import pytest
import copy
import json
import os
import tempfile
from pathlib import Path
from packaging import version
import tensorflow as tf
import numpy as np
from test_models_keras import transposed_conv_model
import aimet_common.libpymo as libpymo
from aimet_tensorflow.keras.utils import common
from aimet_tensorflow.keras.batch_norm_fold import _delete_all_bns_from_model, _find_possible_convs_linears_bn, \
    _get_ordered_conv_linears, _find_all_batch_norms_to_fold, fold_all_batch_norms, fold_all_batch_norms_to_scale, fold_given_batch_norms
from aimet_tensorflow.keras.utils.op.batchnorm import BNUtils
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme
from aimet_tensorflow.keras.utils.quantizer_utils import get_wrappers_weight_quantizer

class TestBatchNormFold():
    """ Test methods for BatchNormFold"""
    @pytest.fixture(autouse=True)
    def set_random_seed(self):
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(43)
        if version.parse(tf.version.VERSION) >= version.parse("2.10"):
            tf.keras.utils.set_random_seed(43)
        else:
            np.random.seed(43)
        yield


    def test_bn_replacement_model(self):

        class Block2(tf.keras.Model):
            def __init__(self):
                super(Block2, self).__init__()
                # define all layers in init
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.relu = tf.keras.layers.ReLU()

            def call(self, x):
                # forward pass
                x = self.dn1(x)
                x = self.bn1(x)
                x = self.relu(x)
                return x

        class Block1(tf.keras.Model):
            def __init__(self):
                super(Block1, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block2 = Block2()

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block2(x)
                return x

        class MyModel(tf.keras.Model):
            def __init__(self, kernel_num, kernel_size, strides):
                super(MyModel, self).__init__()
                # define all layers in init
                self.conv1 = tf.keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.block1 = Block1()
                self.block2 = Block2()

            def call(self, input_tensor, training=False):
                # forward pass
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.block1(x)
                x = self.block2(x)
                return x

        model = MyModel(3, (3, 3), 1)

        bn_layers = [model.bn1, model.block1.block2.bn1, model.block2.bn1]

        _delete_all_bns_from_model(model, bn_layers)

        assert not isinstance(model.bn1, tf.keras.layers.BatchNormalization)
        assert not isinstance(model.block1.block2.bn1, tf.keras.layers.BatchNormalization)
        assert not isinstance(model.block2.bn1, tf.keras.layers.BatchNormalization)
        assert isinstance(model.block1.bn1, tf.keras.layers.BatchNormalization)

    def test_bn_replacement_layers(self):

        class Block2(tf.keras.Model):
            def __init__(self):
                super(Block2, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.relu = tf.keras.layers.ReLU()

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.relu(x)
                return x

        class Block1(tf.keras.Model):
            def __init__(self):
                super(Block1, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.block2 = Block2()

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.block2(x)
                return x

        class MyModel(tf.keras.Model):
            def __init__(self, kernel_num, kernel_size, strides):
                super(MyModel, self).__init__()
                # define all layers in init
                self.conv1 = tf.keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block1 = Block1()
                self.block2 = Block2()

            def call(self, input_tensor, training=False):
                # forward pass
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block1(x)
                x = self.block2(x)
                return x

        model = MyModel(3, (3, 3), 1)
        model.build((1, 6, 6, 3))

        bn_layers = [model.bn1, model.block1.block2.bn1, model.block2.bn1]

        # ref_name = {}
        # module_name_reference(ref_name, model)

        _delete_all_bns_from_model(model, bn_layers)

        assert not isinstance(model.bn1, tf.keras.layers.BatchNormalization)
        assert not isinstance(model.block1.block2.bn1, tf.keras.layers.BatchNormalization)
        assert not isinstance(model.block2.bn1, tf.keras.layers.BatchNormalization)
        assert isinstance(model.block1.bn1, tf.keras.layers.BatchNormalization)

    def test_bn_removal_functional(self):
        inp = tf.keras.Input(shape=(6, 6, 3))
        x = tf.keras.layers.Conv2D(3, 3)(inp)
        x = tf.keras.layers.BatchNormalization(fused=True)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(3, 3)(x)
        x = tf.keras.layers.BatchNormalization(fused=True)(x)
        x = tf.keras.layers.ReLU()(x)

        model = tf.keras.Model(inputs=inp, outputs=x)

        bn_layers = [model.layers[2], model.layers[5]]

        new_model = _delete_all_bns_from_model(model, bn_layers)

        for layer in new_model.layers:
            assert not isinstance(layer, tf.keras.layers.BatchNormalization)
        assert len(new_model.layers) == len(model.layers) - 2

    def test_bn_removal_functional_with_sequantial_bns(self):
        inp = tf.keras.Input(shape=(6, 6, 3))
        x = tf.keras.layers.Conv2D(3, 3)(inp)
        x = tf.keras.layers.BatchNormalization(fused=True)(x)
        x = tf.keras.layers.BatchNormalization(fused=True)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(3, 3)(x)
        x = tf.keras.layers.BatchNormalization(fused=True)(x)
        x = tf.keras.layers.ReLU()(x)

        model = tf.keras.Model(inputs=inp, outputs=x)

        bn_layers = [model.layers[2], model.layers[3], model.layers[6]]

        new_model = _delete_all_bns_from_model(model, bn_layers)

        for layer in new_model.layers:
            assert not isinstance(layer, tf.keras.layers.BatchNormalization)
        assert len(new_model.layers) == len(model.layers) - 3

    @pytest.mark.parametrize('use_scale', [True, False])
    @pytest.mark.parametrize('is_standalone', [True, False])
    def test_bn_without_gamma_for_standalone_bn(self, use_scale, is_standalone):
        inp = tf.keras.Input(shape=(28, 28, 3))
        x = (tf.keras.layers.MaxPool2D() if is_standalone else tf.keras.layers.Conv2D(3, 3))(inp)
        bn = tf.keras.layers.BatchNormalization(scale=use_scale)
        x = bn(x)
        x = tf.keras.layers.ReLU()(x)

        if not use_scale:
            assert not bn.scale, "scale should be False"

        model = tf.keras.Model(inputs=[inp], outputs=[x])

        if use_scale:
            bn.gamma = None
        assert bn.gamma is None, "gamma should be None."

        conv_bns, model = fold_all_batch_norms(model)

        if is_standalone:
            assert bn.gamma is not None
        else:
            for layer in model.layers:
                assert not isinstance(layer, tf.keras.layers.BatchNormalization)

        assert len(conv_bns) == 0 if is_standalone else 1

    def test_bn_removal_functional_two_paths(self):
        inp = tf.keras.Input(shape=(6, 6, 3))

        left = tf.keras.layers.Conv2D(3, 3)(inp)
        left = tf.keras.layers.BatchNormalization(fused=True)(left)
        left = tf.keras.layers.ReLU()(left)
        left = tf.keras.layers.Conv2D(3, 3)(left)
        left = tf.keras.layers.BatchNormalization(fused=True)(left)
        left = tf.keras.layers.ReLU()(left)

        right = tf.keras.layers.Conv2D(3, 3)(inp)
        right = tf.keras.layers.BatchNormalization(fused=True)(right)
        right = tf.keras.layers.ReLU()(right)
        right = tf.keras.layers.Conv2D(3, 3)(right)
        right = tf.keras.layers.BatchNormalization(fused=True)(right)
        right = tf.keras.layers.ReLU()(right)

        output = tf.keras.layers.concatenate([left, right])

        model = tf.keras.Model(inputs=inp, outputs=output)

        bn_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]

        new_model = _delete_all_bns_from_model(model, bn_layers)

        for layer in new_model.layers:
            assert not isinstance(layer, tf.keras.layers.BatchNormalization)
        assert len(new_model.layers) == len(model.layers) - len(bn_layers)

    def test_bn_removal_functional_lambda(self):

        inp = tf.keras.Input(shape=(6, 6, 3))

        left = tf.keras.layers.Conv2D(3, 3)(inp)
        left = tf.keras.layers.BatchNormalization(fused=True)(left)
        left = tf.keras.layers.ReLU()(left)
        left = tf.keras.layers.Conv2D(3, 3)(left)
        left = tf.keras.layers.BatchNormalization(fused=True)(left)
        left = tf.keras.layers.ReLU()(left)

        right = tf.keras.layers.Conv2D(3, 3)(inp)
        right = tf.keras.layers.BatchNormalization(fused=True)(right)
        right = tf.keras.layers.ReLU()(right)
        right = tf.keras.layers.Conv2D(3, 3)(right)
        right = tf.keras.layers.BatchNormalization(fused=True)(right)
        right = tf.keras.layers.ReLU()(right)

        joined = left + right  # lamda connection

        main = tf.keras.layers.Conv2D(2, 2)(joined)
        main = tf.keras.layers.BatchNormalization(fused=True)(main)
        main = tf.keras.layers.ReLU()(main)

        model = tf.keras.Model(inputs=inp, outputs=main)

        bn_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]

        new_model = _delete_all_bns_from_model(model, bn_layers)

        for layer in new_model.layers:
            assert not isinstance(layer, tf.keras.layers.BatchNormalization)
        assert len(new_model.layers) == len(model.layers) - len(bn_layers)

    def test_bn_replacement_sequential(self):

        Block3 = tf.keras.Sequential()
        Block3.add(tf.keras.layers.BatchNormalization(fused=True))
        Block3.add(tf.keras.layers.ReLU())

        Block1 = tf.keras.Sequential()
        Block1.add(tf.keras.layers.BatchNormalization(fused=True))
        Block1.add(Block3)

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(Block1)

        chain_model = tf.keras.Sequential()
        chain_model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1))
        chain_model.add(tf.keras.layers.BatchNormalization(fused=True))
        chain_model.add(Block2)

        # model.build()

        bn_layers = []
        len_layers = []
        len_layers.append(len(chain_model.layers[2].layers))
        len_layers.append(len(chain_model.layers[2].layers[3].layers[1].layers))
        bn_layers.append(chain_model.layers[2].layers[3].layers[1].layers[0])
        bn_layers.append(chain_model.layers[2].layers[0])

        _delete_all_bns_from_model(chain_model, bn_layers)

        assert len(chain_model.layers[2].layers) == len_layers[0]-1
        assert len(chain_model.layers[2].layers[2].layers[1].layers) == len_layers[1]-1

    def test_bn_replacement_combined_seq_model(self):

        class Block3(tf.keras.Model):
            def __init__(self):
                super(Block3, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.relu = tf.keras.layers.ReLU()

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.relu(x)
                return x

        class Block1(tf.keras.Model):
            def __init__(self):
                super(Block1, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.block3 = Block3()

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.block3(x)
                return x

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(Block1())

        class chain_model(tf.keras.Model):
            def __init__(self, kernel_num, kernel_size, strides):
                super(chain_model, self).__init__()
                # define all layers in init
                self.conv1 = tf.keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block2 = Block2

            def call(self, input_tensor, training=False):
                # forward pass
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block2(x)
                return x

        model = chain_model(3, (3, 3), 1)

        bn_layers = []
        bn_layers.append(model.block2.get_layer(index=0))
        bn_layers.append(model.block2.get_layer(index=3).bn1)
        bn_layers.append(model.block2.get_layer(index=3).block3.bn1)

        len_layers = []
        len_layers.append(len(model.block2.layers))

        _delete_all_bns_from_model(model, bn_layers)

        assert len(model.block2.layers) == len_layers[0]-1
        assert not isinstance(model.block2.get_layer(index=2).bn1, tf.keras.layers.BatchNormalization)
        assert not isinstance(model.block2.get_layer(index=2).block3.bn1, tf.keras.layers.BatchNormalization)

    def test_bn_replacement_combined2_seq_model(self):

        Block3 = tf.keras.Sequential()
        Block3.add(tf.keras.layers.BatchNormalization(fused=True))
        Block3.add(tf.keras.layers.ReLU())
        Block3.add(tf.keras.layers.Conv2D(3, 3))

        class Block1(tf.keras.Model):
            def __init__(self):
                super(Block1, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.block3 = Block3

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.block3(x)
                return x

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(Block1())

        class chain_model(tf.keras.Model):
            def __init__(self, kernel_num, kernel_size, strides):
                super(chain_model, self).__init__()
                # define all layers in init
                self.conv1 = tf.keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block2 = Block2

            def call(self, input_tensor, training=False):
                # forward pass
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block2(x)
                return x

        model = chain_model(3, (3, 3), 1)

        bn_layers = []
        bn = model.block2.get_layer(index=3).block3.get_layer(index=0)
        bn_layers.append(bn)
        bn = model.bn1
        bn_layers.append(bn)

        len_layers = []
        len_layers.append(len(model.block2.get_layer(index=3).block3.layers))

        _delete_all_bns_from_model(model, bn_layers)

        assert len(model.block2.get_layer(index=3).block3.layers) == len_layers[0]-1
        assert not isinstance(model.bn1, tf.keras.layers.BatchNormalization)

    def test_bn_replacement_combined3_seq_model(self):
        Block3 = tf.keras.Sequential()
        Block3.add(tf.keras.layers.BatchNormalization(fused=True))
        Block3.add(tf.keras.layers.ReLU())
        Block3.add(tf.keras.layers.Conv2D(3, 3))

        class Block1(tf.keras.Model):
            def __init__(self):
                super(Block1, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.block3 = Block3

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.block3(x)
                return x

        class chain_model(tf.keras.Model):
            def __init__(self, kernel_num, kernel_size, strides):
                super(chain_model, self).__init__()
                # define all layers in init
                self.conv1 = tf.keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block1 = Block1()

            def call(self, input_tensor, training=False):
                # forward pass
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block1(x)
                return x

        model = chain_model(3, (3, 3), 1)

        bn_layers = []
        bn = model.block1.block3.get_layer(index=0)
        bn_layers.append(bn)
        bn = model.block1.bn1
        bn_layers.append(bn)

        len_layers = []
        len_layers.append(len(model.block1.block3.layers))

        _delete_all_bns_from_model(model, bn_layers)

        assert len(model.block1.block3.layers) == len_layers[0]-1
        assert not isinstance(model.block1.bn1, tf.keras.layers.BatchNormalization)

    def test_bn_replacement_combined_all(self):

        Block3 = tf.keras.Sequential()
        Block3.add(tf.keras.layers.BatchNormalization(fused=True))
        Block3.add(tf.keras.layers.ReLU())
        Block3.add(tf.keras.layers.Conv2D(3, 3))

        inputs = tf.keras.Input((28, 28, 64))
        conv2d = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(inputs)
        block3 = Block3(conv2d)
        outputs = tf.keras.layers.BatchNormalization(fused=True)(block3)
        Block1 = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model3")

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(Block1)

        class chain_model(tf.keras.Model):
            def __init__(self, kernel_num, kernel_size, strides):
                super(chain_model, self).__init__()
                # define all layers in init
                self.conv1 = tf.keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block2 = Block2

            def call(self, input_tensor, training=False):
                # forward pass
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block2(x)
                return x

        model = chain_model(3, (3, 3), 1)

        bn_layers = []
        bn = model.block2.get_layer(index=3).get_layer(index=2).get_layer(index=0)
        bn_layers.append(bn)
        bn = model.block2.get_layer(index=3).get_layer(index=3)
        bn_layers.append(bn)
        bn = model.bn1
        bn_layers.append(bn)

        len_layers = []
        len_layers.append(len(model.block2.get_layer(index=3).get_layer(index=2).layers))

        _delete_all_bns_from_model(model, bn_layers)

        assert len(model.block2.get_layer(index=3).get_layer(index=2).layers) == len_layers[0]-1

        assert isinstance(model.block2.get_layer(index=3).get_layer(index=3), tf.keras.layers.BatchNormalization)
        assert not isinstance(model.bn1, tf.keras.layers.BatchNormalization)

    def test_modify_bn_params_to_make_as_passthrough(self):
        inputs = tf.keras.Input(shape=(1,))
        add_ = tf.keras.layers.Lambda(lambda x: x + 10)(inputs)
        outputs = tf.keras.layers.BatchNormalization(epsilon=0, beta_initializer='random_uniform',
                                                     gamma_initializer='random_uniform',
                                                     moving_mean_initializer='random_uniform',
                                                     moving_variance_initializer='ones')(add_)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_model")

        BNUtils.modify_bn_params_to_make_as_passthrough(model.layers[2])

        var = tf.constant([[5.0]])
        out = model(var)
        assert out.numpy() == 15.0

    def test_find_conv_bn_pairs_combined_model(self):
        Block3 = tf.keras.Sequential()
        Block3.add(tf.keras.layers.BatchNormalization(fused=True))
        Block3.add(tf.keras.layers.Conv2D(3, 3))

        inputs = tf.keras.Input((60, 60, 3))
        conv2d = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(inputs)
        block3 = Block3(conv2d)
        outputs = tf.keras.layers.BatchNormalization(fused=True)(block3)
        Block1 = tf.keras.Model(inputs=inputs, outputs=outputs)

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(Block1)

        class MyModelMLP(tf.keras.Model):
            def __init__(self, input_shape):
                super(MyModelMLP, self).__init__()

                self.conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=1, activation='relu')
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block2 = Block2
                self.relu = tf.keras.layers.ReLU()
                # # Parameters of the model
                self.input_layer = tf.keras.layers.Input(input_shape)
                self.out = self.call(self.input_layer)
                super().__init__(inputs=self.input_layer, outputs=self.out)

            # Define forward passing of model
            def call(self, input_tensor):
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block2(x)
                x = self.relu(x)
                return x

        model = MyModelMLP((64, 64, 64))

        node_layer_map = common.create_node_to_layer_map(model)
        layer_out_node_map = common.create_layer_to_out_node_map(model)
        conv_linear_with_bn_dict = _find_possible_convs_linears_bn(node_layer_map, layer_out_node_map)

        assert not model.conv1 in conv_linear_with_bn_dict
        assert not model.bn1 in conv_linear_with_bn_dict
        assert len(conv_linear_with_bn_dict) == 3

    def test_find_conv_bn_pairs_functional(self):

        tf.compat.v1.reset_default_graph()
        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        input2 = tf.keras.Input(name='input2', shape=(12, 12, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a')(input1)
        y = tf.keras.layers.BatchNormalization()(x1)
        x2 = tf.keras.layers.BatchNormalization()(input2)
        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b')(x2)
        x = tf.keras.layers.add([y, x2])
        x = tf.keras.layers.Conv2D(4, (1, 1), name='conv2')(x)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(x)
        output = tf.nn.relu(bn_op)
        model = tf.keras.Model([input1, input2], output)

        node_layer_map = common.create_node_to_layer_map(model)
        layer_out_node_map = common.create_layer_to_out_node_map(model)
        conv_linear_with_bn_dict = _find_possible_convs_linears_bn(node_layer_map, layer_out_node_map)

        assert 3 == len(conv_linear_with_bn_dict)

    def test_find_conv_bn_pairs_sequential(self):
        Block1 = tf.keras.Sequential()
        Block1.add(tf.keras.layers.ReLU())
        Block1.add(tf.keras.layers.BatchNormalization())
        Block1.add(tf.keras.layers.Conv2D(3, 3))

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.Input((28, 28, 64)))
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(Block1)
        Block2.add(tf.keras.layers.ReLU())

        node_layer_map = common.create_node_to_layer_map(Block2)
        layer_out_node_map = common.create_layer_to_out_node_map(Block2)
        conv_linear_with_bn_dict = _find_possible_convs_linears_bn(node_layer_map, layer_out_node_map)

        assert Block1.layers[2] in conv_linear_with_bn_dict
        assert 1, len(conv_linear_with_bn_dict)

    def test_ordered_conv_linears_functional(self):
        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a')(input1)
        y = tf.keras.layers.BatchNormalization()(x1)
        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b')(y)
        x3 = tf.keras.layers.Conv2D(8, (3, 3), name='conv2b')(y)
        y2 = tf.keras.layers.BatchNormalization()(x2)
        y3 = tf.keras.layers.BatchNormalization()(x3)
        x = tf.keras.layers.add([y3, y2])
        x = tf.keras.layers.Conv2D(4, (1, 1), name='conv3')(x)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(x)
        output = tf.nn.relu(bn_op)
        model = tf.keras.Model(input1, output)

        node_layer_map = common.create_node_to_layer_map(model)
        layer_out_node_map = common.create_layer_to_out_node_map(model)
        ordered_conv_linears = _get_ordered_conv_linears(node_layer_map, layer_out_node_map)

        for layer in model.layers:
            if layer.name == 'conv1a':
                assert layer == ordered_conv_linears[0]
            if layer.name == 'conv3':
                assert layer == ordered_conv_linears[3]

    def test_ordered_conv_linears_sequential(self):
        Block1 = tf.keras.Sequential()
        Block1.add(tf.keras.layers.ReLU())
        Block1.add(tf.keras.layers.BatchNormalization())
        Block1.add(tf.keras.layers.Conv2D(3, 3))

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.Input((28, 28, 64)))
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(Block1)
        Block2.add(tf.keras.layers.ReLU())

        node_layer_map = common.create_node_to_layer_map(Block2)
        layer_out_node_map = common.create_layer_to_out_node_map(Block2)
        ordered_conv_linears = _get_ordered_conv_linears(node_layer_map, layer_out_node_map)
        assert Block2.layers[2] == ordered_conv_linears[0]

    def test_ordered_conv_linears_combined_model(self):
        Block3 = tf.keras.Sequential()
        Block3.add(tf.keras.layers.BatchNormalization(fused=True))
        Block3.add(tf.keras.layers.Conv2D(3, 3))

        inputs = tf.keras.Input((60, 60, 3))
        conv2d = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(inputs)
        block3 = Block3(conv2d)
        outputs = tf.keras.layers.BatchNormalization(fused=True)(block3)
        Block1 = tf.keras.Model(inputs=inputs, outputs=outputs)

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(Block1)

        class MyModelMLP(tf.keras.Model):
            def __init__(self, input_shape):
                super(MyModelMLP, self).__init__()

                self.conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=1, activation='linear')
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block2 = Block2
                self.relu = tf.keras.layers.ReLU()
                # # Parameters of the model
                self.input_layer = tf.keras.layers.Input(input_shape)
                self.out = self.call(self.input_layer)
                super().__init__(inputs=self.input_layer, outputs=self.out)

            # Define forward passing of model
            def call(self, input_tensor):
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block2(x)
                x = self.relu(x)
                return x

        model = MyModelMLP((64, 64, 64))

        node_layer_map = common.create_node_to_layer_map(model)
        layer_out_node_map = common.create_layer_to_out_node_map(model)
        ordered_conv_linears = _get_ordered_conv_linears(node_layer_map, layer_out_node_map)

        assert model.conv1 == ordered_conv_linears[0]
        assert model.dn1 == ordered_conv_linears[1]
        assert model.block2.layers[1] == ordered_conv_linears[2]
        assert model.block2.layers[3].layers[1] == ordered_conv_linears[3]
        assert model.block2.layers[3].layers[2].layers[1] == ordered_conv_linears[4]

    def test_find_all_bns_to_fold_sequential(self):

        Block1 = tf.keras.Sequential()
        Block1.add(tf.keras.Input((60, 60, 3)))
        Block1.add(tf.keras.layers.ReLU())
        Block1.add(tf.keras.layers.BatchNormalization())
        Block1.add(tf.keras.layers.Conv2D(3, 3))
        Block1.add(tf.keras.layers.BatchNormalization())
        Block1.add(tf.keras.layers.Conv2D(6, 6))
        Block1.add(tf.keras.layers.ReLU())

        conv_bn_pairs, bn_conv_pairs, _ = _find_all_batch_norms_to_fold(Block1)
        assert 2 == len(conv_bn_pairs) + len(bn_conv_pairs)
        assert (Block1.layers[2], Block1.layers[3]) == conv_bn_pairs[0]
        assert (Block1.layers[1], Block1.layers[2]) == bn_conv_pairs[0]

    def test_find_all_bns_to_fold_functional(self):
        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a')(input1)
        y = tf.keras.layers.BatchNormalization()(x1)
        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b')(y)
        x3 = tf.keras.layers.Conv2D(8, (3, 3), name='conv2b')(y)
        y2 = tf.keras.layers.BatchNormalization()(x2)
        y3 = tf.keras.layers.BatchNormalization()(x3)
        x = tf.keras.layers.add([y3, y2])
        x = tf.keras.layers.Conv2D(4, (1, 1), name='conv3')(x)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(x)
        output = tf.keras.layers.ReLU()(bn_op)
        model = tf.keras.Model(input1, output)

        conv_bn_pairs, bn_conv_pairs, _ = _find_all_batch_norms_to_fold(model)
        assert 4 == len(conv_bn_pairs) + len(bn_conv_pairs)

    def test_find_all_bns_to_fold_combined_model(self):
        Block3 = tf.keras.Sequential()
        Block3.add(tf.keras.layers.BatchNormalization(fused=True))
        Block3.add(tf.keras.layers.Conv2D(3, 3))

        inputs = tf.keras.Input((60, 60, 3))
        conv2d = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(inputs)
        block3 = Block3(conv2d)
        outputs = tf.keras.layers.BatchNormalization(fused=True)(block3)
        Block1 = tf.keras.Model(inputs=inputs, outputs=outputs)

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(Block1)

        class MyModelMLP(tf.keras.Model):
            def __init__(self, input_shape):
                super(MyModelMLP, self).__init__()

                self.conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=1, activation='relu')
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block2 = Block2
                self.relu = tf.keras.layers.ReLU()
                # # Parameters of the model
                self.input_layer = tf.keras.layers.Input(input_shape)
                self.out = self.call(self.input_layer)
                super().__init__(inputs=self.input_layer, outputs=self.out)

            # Define forward passing of model
            def call(self, input_tensor):
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block2(x)
                x = self.relu(x)
                return x

        model = MyModelMLP((64, 64, 64))

        conv_bn_pairs, bn_conv_pairs, _ = _find_all_batch_norms_to_fold(model)

        assert 3, len(conv_bn_pairs) + len(bn_conv_pairs)
        assert (model.layers[4].layers[0], model.layers[4].layers[1]) == bn_conv_pairs[0]
        assert (model.layers[4].layers[3].layers[1],
                model.layers[4].layers[3].layers[2].layers[0]) == conv_bn_pairs[0]
        assert (model.layers[4].layers[3].layers[2].layers[1],
                model.layers[4].layers[3].layers[3]) == conv_bn_pairs[1]

    def test_bn_fold_auto_rules_bn_after_conv(self):
        """
        Test batch norm fold layer selection when conv layer is followed by a BN layer
        """
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(conv_op)
        relu = tf.nn.relu(bn_op)
        model = tf.keras.Model(inputs=inputs, outputs=relu)

        conv_bn_pairs, bn_conv_pairs, _ = _find_all_batch_norms_to_fold(model)
        assert 1 == len(conv_bn_pairs) + len(bn_conv_pairs)

    def test_bn_fold_layer_selection_looped_network(self):
        """
        Test layer selection with looped network
        """
        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a',
                                    kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                    bias_initializer='random_uniform')(input1)

        bn_op_1 = tf.keras.layers.BatchNormalization(fused=True)(x1)
        bn_op_2 = tf.keras.layers.BatchNormalization(fused=True)(x1)

        add = tf.keras.layers.add([bn_op_1, bn_op_2])

        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b',
                                    kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                    bias_initializer='random_uniform')(add)

        model = tf.keras.Model(inputs=input1, outputs=x2)

        conv_bn_pairs, bn_conv_pairs, _ = _find_all_batch_norms_to_fold(model)

        assert 0 == len(conv_bn_pairs) + len(bn_conv_pairs)

    def test_bn_fold_auto_rules_bn_before_conv(self):
        """
        Test batch norm fold layer selection when BN layer is followed by a conv layer
        """
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(inputs)
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(bn_op)
        relu = tf.nn.relu(conv_op)
        model = tf.keras.Model(inputs=inputs, outputs=relu)

        conv_bn_pairs, bn_conv_pairs, _ = _find_all_batch_norms_to_fold(model)
        assert 1 == len(conv_bn_pairs) + len(bn_conv_pairs)

    def test_bn_fold_find_layers_model_with_multi_input(self):
        """
        Test bn fold with multiple input nodes
        """

        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        input2 = tf.keras.Input(name='input2', shape=(12, 12, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a')(input1)
        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b')(input2)
        x = tf.keras.layers.add([x1, x2])
        x = tf.keras.layers.Conv2D(4, (1, 1), name='conv2')(x)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(x)
        relu = tf.nn.relu(bn_op)
        model = tf.keras.Model(inputs=[input1, input2], outputs=relu)

        conv_bn_pairs, bn_conv_pairs, _ = _find_all_batch_norms_to_fold(model)
        assert 1 == len(conv_bn_pairs) + len(bn_conv_pairs)

    def test_bn_fold_auto_rules_conv_bn_conv(self):
        """
        Test batch norm fold layer selection with pattern conv1 - bn - conv2
        bn folds into conv1
        """
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv = tf.keras.layers.Conv2D(32, (3, 3), name='conv1')(inputs)
        bn = tf.keras.layers.BatchNormalization(fused=True, name="bn")(conv)
        conv2 = tf.keras.layers.Conv2D(32, (3, 3), name='conv2')(bn)
        relu = tf.nn.relu(conv2)
        model = tf.keras.Model(inputs=inputs, outputs=relu)

        conv_bn_pairs, bn_conv_pairs, _ = _find_all_batch_norms_to_fold(model)
        assert 1 == len(conv_bn_pairs) + len(bn_conv_pairs)
        conv_linear, batchnorm = conv_bn_pairs[0]
        assert 'conv1' == conv_linear.name
        assert 'bn' == batchnorm.name
        # add additional check to verify backward fold is picked over forward in case both are available

    def test_batch_norm_fold(self):
        """
        Test batch norm fold custom model
        """
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn = tf.keras.layers.BatchNormalization(fused=True)(conv, training=False)
        relu = tf.nn.relu(bn)
        model = tf.keras.Model(inputs=inputs, outputs=relu)

        np.random.seed(0)
        w_shape = model.layers[0].input.shape
        numpy_data = np.random.rand(1, w_shape[1], w_shape[2], w_shape[3]).astype(np.float32)

        baseline_output = model(numpy_data)

        _, model = fold_all_batch_norms(model)
        output_after_fold = model(numpy_data)

        assert np.allclose(baseline_output, output_after_fold, atol=1.e-4)

    def test_batch_norm_fold_with_random_data(self):
        """
        Test batch norm fold custom model with randomly initialized kernel, bias and bn params,
        """
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv = tf.keras.layers.Conv2D(32, (3, 3),
                                      kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                      bias_initializer='random_uniform')(inputs)
        bn = tf.keras.layers.BatchNormalization(fused=True,
                                                beta_initializer='random_uniform',
                                                gamma_initializer='random_uniform',
                                                moving_mean_initializer='random_uniform',
                                                moving_variance_initializer='ones')(conv, training=False)
        relu = tf.nn.relu(bn)

        model = tf.keras.Model(inputs=inputs, outputs=relu)

        np.random.seed(0)
        w_shape = model.layers[0].input.shape
        numpy_data = np.random.rand(1, w_shape[1], w_shape[2], w_shape[3]).astype(np.float32)
        baseline_output = model(numpy_data)

        _, model = fold_all_batch_norms(model)

        output_after_fold = model(numpy_data)

        assert not np.allclose(baseline_output, output_after_fold, atol=0)
        assert np.allclose(baseline_output, output_after_fold, atol=1e-4)

    def test_bn_fold_with_linear_layer(self):
        """
        Custom Model where BN layer is followed by Dense layer
        :return:
        """
        inputs = tf.keras.Input(shape=(1, 1, 4,))
        bn = tf.keras.layers.BatchNormalization(fused=True)(inputs, training=False)
        x = tf.keras.layers.Flatten()(bn)
        dense = tf.keras.layers.Dense(2, activation=tf.nn.relu, name="linear_layer")(x)
        model = tf.keras.Model(inputs=inputs, outputs=dense)

        # get baseline output
        np.random.seed(0)
        w_shape = model.layers[0].input.shape
        numpy_data = np.random.rand(1, w_shape[1], w_shape[2], w_shape[3]).astype(np.float32)
        baseline_output = model(numpy_data)
        weight_before_fold = model.layers[3].kernel.numpy()

        _, model = fold_all_batch_norms(model)
        after_fold_output = model(numpy_data)
        weight_after_fold = model.layers[2].kernel.numpy()

        # check that weight got updated
        assert not np.allclose(weight_before_fold, weight_after_fold, atol=1e-4)

        # check outputs are close
        assert np.allclose(baseline_output, after_fold_output, atol=1e-3)

    def test_find_conv_bn_pairs_functional_nested(self):
        """
        Test case for finding conv-bn pairs when the inner model has 2 layers connected to the input

        """
        inputs = tf.keras.Input((26, 26, 3))
        conv2d_1 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(inputs)
        bn = tf.keras.layers.BatchNormalization(fused=True)(inputs)
        conv2d_2 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(bn)
        outputs = tf.keras.layers.add([conv2d_1, conv2d_2])
        Block1 = tf.keras.Model(inputs=inputs, outputs=outputs)

        inputs2 = tf.keras.Input((28, 28, 64))
        bn1 = tf.keras.layers.BatchNormalization(fused=True)(inputs2)
        relu = tf.keras.layers.ReLU()(bn1)
        conv2d_0 = tf.keras.layers.Conv2D(3, 3)(relu)
        block1 = Block1(conv2d_0)
        outputs = tf.keras.layers.ReLU()(block1)
        model = tf.keras.Model(inputs=inputs2, outputs=outputs)

        node_layer_map = common.create_node_to_layer_map(model)
        layer_out_node_map = common.create_layer_to_out_node_map(model)
        conv_linear_with_bn_dict = _find_possible_convs_linears_bn(node_layer_map, layer_out_node_map)

        assert 10 == len(node_layer_map)
        assert 9 == len(layer_out_node_map)
        assert 1 == len(conv_linear_with_bn_dict)

    def test_bn_fold_with_no_bias(self):
        inputs = tf.keras.Input((32, 32, 3))
        x = tf.keras.layers.Conv2D(16, 3, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization(beta_initializer="normal", gamma_initializer="normal")(x)
        outputs = tf.keras.layers.ReLU()(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        mock_input = np.random.randn(1, 32, 32, 3)
        output_before_batchnorm_folding = model(mock_input)
        _, model = fold_all_batch_norms(model)
        output_after_batchnorm_folding = model(mock_input)

        assert np.allclose(output_before_batchnorm_folding, output_after_batchnorm_folding, atol=1e-4)

    def test_bn_fold_depthwise_convolution(self):
        inputs = tf.keras.Input((32, 32, 3))
        x = tf.keras.layers.DepthwiseConv2D(16, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization(beta_initializer="normal", gamma_initializer="normal")(x)
        outputs = tf.keras.layers.ReLU()(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        mock_input = np.random.randn(1, 32, 32, 3)
        output_before_batchnorm_folding = model(mock_input)
        _, model = fold_all_batch_norms(model)
        output_after_batchnorm_folding = model(mock_input)

        assert np.allclose(output_before_batchnorm_folding, output_after_batchnorm_folding, atol=1e-4)

    def test_bn_conversion(self):
        inputs = tf.keras.Input((26, 26, 3))
        x = tf.keras.layers.MaxPool2D()(inputs)

        # Standalone Batch Norm which will get converted.
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(x)
        x = tf.keras.layers.BatchNormalization(fused=True)(x)
        x = tf.keras.layers.ReLU()(x)
        outputs = tf.keras.layers.Dense(units = 10)(x)
        model = tf.keras.Model(inputs, outputs)

        mock_input = np.random.randn(1, 26, 26, 3)
        output_before_batchnorm_folding = model(mock_input)
        folded_pairs, model = fold_all_batch_norms(model)
        output_after_batchnorm_folding = model(mock_input)
        assert 1 == len(folded_pairs)
        assert np.allclose(output_before_batchnorm_folding, output_after_batchnorm_folding, atol=1e-2)


symmetric_quantsim_config = {
    "defaults": {
        "ops": {"is_output_quantized": "True"},
        "params": {"is_quantized": "True", "is_symmetric": "True"},
        "strict_symmetric": "False",
        "unsigned_symmetric": "True",
        "per_channel_quantization": "True"
    },
    "params": {
        "bias": {"is_quantized": "False"}
    },
    "op_type": {},
    "supergroups": [
        {"op_list": ["Conv", "Relu"]},
        {"op_list": ["Conv", "Clip"]},
        {"op_list": ["Add", "Relu"]},
        {"op_list": ["Gemm", "Relu"]},
    ],
    "model_input": {"is_input_quantized": "True"},
    "model_output": {}
}

asymmetric_quantsim_config = copy.deepcopy(symmetric_quantsim_config)
asymmetric_quantsim_config["defaults"]["params"]["is_symmetric"] = "False"

strict_symmetric_quantsim_config = copy.deepcopy(symmetric_quantsim_config)
strict_symmetric_quantsim_config["defaults"]["strict_symmetric"] = "True"

quantsim_config_map = {
    "symmetric": symmetric_quantsim_config,
    "asymmetric": asymmetric_quantsim_config,
    # "strict_symmetric": strict_symmetric_quantsim_config,
}


def create_quantsim_model_and_compute_encodings(model, dummy_input, quantsim_config=None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_file_path = Path(tmp_dir, "quantsim_config.json")

        quantsim_config = quantsim_config or symmetric_quantsim_config
        with open(config_file_path, 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(model,
                                quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                config_file=config_file_path)

    def forward_pass_callback(model, _):
        model(dummy_input)

    sim.compute_encodings(forward_pass_callback, None)
    return sim

class TestBatchNormFoldToScale:
    @pytest.fixture(autouse=True)
    def clear_sessions(self):
        tf.keras.backend.clear_session()
        yield
    
    @pytest.fixture(autouse=True)
    def set_random_seed(self):
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(43)
        if version.parse(tf.version.VERSION) >= version.parse("2.10"):
            tf.keras.utils.set_random_seed(43)
        else:
            np.random.seed(43)
        yield

    def _test_output_quantizers_enabled_and_quant_mode_is_quantize_dequantize(self, model, layer_nums_to_check):
        for layer_num in layer_nums_to_check:
            is_enabled = model.layers[layer_num].output_quantizers[0].is_enabled()
            output_quantizers_quant_mode = model.layers[layer_num].output_quantizers[0].quant_mode
            assert is_enabled and output_quantizers_quant_mode == int(libpymo.TensorQuantizerOpMode.quantizeDequantize), \
            "Quantizer layer num {} -> Enabled: {}, quantMode: {}".format(layer_num, is_enabled, output_quantizers_quant_mode)

    def test_fold_bn_before_conv_no_bias(self):
        input_shape = (20, 4, 4, 10)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.Conv2D(20, 2, use_bias=False)(inp)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(40, 2, use_bias=False)(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])

        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert model.layers[3].output_quantizers[0].is_enabled()
        assert np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[3].param_quantizers)))

        layer_list = [(model.layers[3], model.layers[-1])]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    def test_fold_bn_before_conv_with_bias(self):
        input_shape = (2, 24, 24, 10)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.Conv2D(20, 3)(inp)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(30, 3)(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])

        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert model.layers[3].output_quantizers[0].is_enabled()
        assert np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[3].param_quantizers)))

        layer_list = [(model.layers[3], model.layers[-1])]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    def test_fold_bn_after_conv_no_bias(self):
        input_shape = (2, 24, 24, 10)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.Conv2D(20, 3)(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])

        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.layers[1].output_quantizers[0].is_enabled()
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()
        assert not model.layers[2].output_quantizers[0].is_enabled()
        assert not np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[2].param_quantizers)))
        assert model.layers[-1].output_quantizers[0].is_enabled()

        baseline_output = model(random_input)

        layer_list = [(model.layers[1], model.layers[2])]

        model = fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        # Check bn is deleted
        for wrapper in model.layers[1:]:
            assert not isinstance(wrapper._layer_to_wrap, tf.keras.layers.BatchNormalization)

        # Check quantizers
        assert not model.layers[1].output_quantizers[0].is_enabled()
        self._test_output_quantizers_enabled_and_quant_mode_is_quantize_dequantize(model, layer_nums_to_check=[-1])
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()

        relu_output_encoding = model.layers[-1].output_quantizers[0].encoding
        delta = float((relu_output_encoding.max - relu_output_encoding.min)/255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference

    def test_fold_bn_after_conv_depthwise(self):
        input_shape = (2, 24, 24, 10)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.DepthwiseConv2D(3)(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])

        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.layers[1].output_quantizers[0].is_enabled()
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()
        assert not model.layers[2].output_quantizers[0].is_enabled()
        assert not np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[2].param_quantizers)))
        assert model.layers[-1].output_quantizers[0].is_enabled()

        baseline_output = model(random_input)

        _ = fold_all_batch_norms_to_scale(sim)
        model = sim.model

        output_after_fold = model(random_input)

        # Check to make sure rebuild Quantization Sim Model working properly
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(path=tmp_dir, filename_prefix="temp_bn_fold_to_scale")

        # Check bn is deleted
        for wrapper in model.layers[1:]:
            assert not isinstance(wrapper._layer_to_wrap, tf.keras.layers.BatchNormalization)

        # Check quantizers
        assert not model.layers[1].output_quantizers[0].is_enabled()
        self._test_output_quantizers_enabled_and_quant_mode_is_quantize_dequantize(model, layer_nums_to_check=[-1])
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()

        relu_output_encoding = model.layers[-1].output_quantizers[0].encoding
        delta = float((relu_output_encoding.max - relu_output_encoding.min)/255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference

    def test_fold_bn_after_conv_with_bias(self):
        input_shape = (2, 24, 24, 10)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.Conv2D(20, 3)(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])

        random_input = np.random.rand(*input_shape)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.layers[1].output_quantizers[0].is_enabled()
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()
        assert not model.layers[2].output_quantizers[0].is_enabled()
        assert not np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[2].param_quantizers)))
        assert model.layers[-1].output_quantizers[0].is_enabled()

        baseline_output = model(random_input)

        layer_list = [(model.layers[1], model.layers[2])]

        model = fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        # Check bn is deleted
        for wrapper in model.layers[1:]:
            assert not isinstance(wrapper._layer_to_wrap, tf.keras.layers.BatchNormalization)

        # Check quantizers
        assert not model.layers[1].output_quantizers[0].is_enabled()
        self._test_output_quantizers_enabled_and_quant_mode_is_quantize_dequantize(model, layer_nums_to_check=[-1])
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()

        relu_output_encoding = model.layers[-1].output_quantizers[0].encoding
        delta = float((relu_output_encoding.max - relu_output_encoding.min)/255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference

    def test_fold_bn_before_linear_layer_no_bias(self):
        input_shape = (32, 10)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.BatchNormalization()(inp)
        x = tf.keras.layers.Dense(20, use_bias=False)(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])

        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim

        # Check quantizers are enabled/disabled properly
        assert model.layers[2].output_quantizers[0].is_enabled()
        assert np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[2].param_quantizers)))
        assert model.layers[1].output_quantizers[0].is_enabled()
        assert np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[1].param_quantizers)))

        layer_list = [(model.layers[1], model.layers[2])]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    def test_fold_bn_before_linear_layer_with_bias(self):
        input_shape = (32, 10)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.BatchNormalization()(inp)
        x = tf.keras.layers.Dense(20)(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])

        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim

        # Check quantizers are enabled/disabled properly
        assert model.layers[2].output_quantizers[0].is_enabled()
        assert np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[2].param_quantizers)))
        assert model.layers[1].output_quantizers[0].is_enabled()
        assert np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[1].param_quantizers)))

        layer_list = [(model.layers[1], model.layers[2])]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    def test_fold_bn_after_linear_layer_no_bias(self):
        input_shape = (32, 10)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.Dense(20, use_bias=False)(inp)
        x = tf.keras.layers.BatchNormalization()(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])

        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.layers[1].output_quantizers[0].is_enabled()
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()
        assert model.layers[2].output_quantizers[0].is_enabled()
        assert not np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[2].param_quantizers)))

        baseline_output = model(random_input)

        layer_list = [(model.layers[1], model.layers[2])]

        model = fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        # Check bn is deleted
        for wrapper in model.layers[1:]:
            assert not isinstance(wrapper._layer_to_wrap, tf.keras.layers.BatchNormalization)

        self._test_output_quantizers_enabled_and_quant_mode_is_quantize_dequantize(model, layer_nums_to_check=[1])
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()

        # Check batchnorm's output encoding is coped to fc's output encoding
        fc_output_encoding = model.layers[1].output_quantizers[0].encoding
        delta = float((fc_output_encoding.max - fc_output_encoding.min)/255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference

    def test_fold_bn_after_linear_layer_with_bias(self):
        input_shape = (32, 10)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.Dense(20)(inp)
        x = tf.keras.layers.BatchNormalization()(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])

        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.layers[1].output_quantizers[0].is_enabled()
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()
        assert model.layers[2].output_quantizers[0].is_enabled()
        assert not np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[2].param_quantizers)))

        baseline_output = model(random_input)

        layer_list = [(model.layers[1], model.layers[2])]

        model = fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        # Check bn is deleted
        for wrapper in model.layers[1:]:
            assert not isinstance(wrapper._layer_to_wrap, tf.keras.layers.BatchNormalization)

        self._test_output_quantizers_enabled_and_quant_mode_is_quantize_dequantize(model, layer_nums_to_check=[1])
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()

        # Check batchnorm's output encoding is coped to fc's output encoding
        fc_output_encoding = model.layers[1].output_quantizers[0].encoding
        delta = float((fc_output_encoding.max - fc_output_encoding.min)/255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference

    def test_bn_fold_auto_mode_transposed_conv2d(self):
        model = transposed_conv_model()
        random_input = np.random.rand(10, *model.input_shape[1:])
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim.model

        baseline_output = model(random_input)
        folded_pairs = fold_all_batch_norms_to_scale(sim)
        model = sim.model
        output_after_fold = model(random_input)
        # Check to make sure rebuild Quantization Sim Model working properly
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(path=tmp_dir, filename_prefix="temp_bn_fold_to_scale")

        # Check bn is deleted
        for wrapper in model.layers[1:]:
            assert not isinstance(wrapper._layer_to_wrap, tf.keras.layers.BatchNormalization)

        # Check quantizers are enabled/disabled proeprly
        self._test_output_quantizers_enabled_and_quant_mode_is_quantize_dequantize(model, layer_nums_to_check=[1, 3])

        conv2_output_encoding = model.layers[3].output_quantizers[0].encoding
        delta = float((conv2_output_encoding.max - conv2_output_encoding.min)/255)
        # Moved to 2 ticks because the test would fail randomly based on different random inputs.
        # Looks to have the trivial numerical error explained by Kyunggeun
        assert np.allclose(baseline_output, output_after_fold, atol=delta*2)  # Allow 2-tick difference
        assert len(folded_pairs) == 2

    def test_bn_fold_auto_mode(self):
        input_shape = (2, 24, 24, 10)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.Conv2D(20, 3)(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(15, 3)(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(20, 3)(x)

        x = tf.keras.layers.Conv2D(20, 3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(10)(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])
        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)

        with pytest.raises(RuntimeError):
            fold_all_batch_norms_to_scale(sim)

    @pytest.mark.parametrize("layer_type", ["GRU", "RNN"])
    def test_bn_fold_with_gru_layer(self, layer_type):
        def get_gru_model(layer_type):
            _input = tf.keras.layers.Input(shape=(32, 32, 3), name="input")
            x = tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', activation=None)(_input)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(x)
            x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation=None)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Reshape(target_shape=(2, -1))(x)
            if layer_type == "GRU":
                x = tf.keras.layers.GRU(32, return_sequences=True, unroll=True)(x)
            elif layer_type == "RNN":
                x = tf.keras.layers.RNN(tf.keras.layers.GRUCell(32))(x)
            else:
                raise AttributeError(f"Found invalid layer_type: {layer_type}")
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(128, activation=None)(x)
            x = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(x)
            x = tf.keras.layers.Dense(10, activation='softmax')(x)
            gru_model = tf.keras.Model(inputs=_input, outputs=x)
            return gru_model

        model = get_gru_model(layer_type)
        mock_input = np.random.randn(1, 32, 32, 3)

        node_to_layer_map = common.create_node_to_layer_map(model)
        for node, connection in node_to_layer_map.items():
            assert not isinstance(node.layer, tf.keras.layers.GRUCell)
            if connection[0] is not None:
                for inp_layer in connection[0]:
                    assert not isinstance(inp_layer, tf.keras.layers.GRUCell)
            assert not isinstance(connection[1], tf.keras.layers.GRUCell)

        node_layer_types = {type(node_ref.layer) for node_ref in node_to_layer_map.keys()}
        if layer_type == "GRU":
            assert tf.keras.layers.GRU in node_layer_types
        elif layer_type == "RNN":
            assert tf.keras.layers.RNN in node_layer_types

        output_before_batchnorm_folding = model(mock_input)
        _, model = fold_all_batch_norms(model)
        output_after_batchnorm_folding = model(mock_input)

        assert np.allclose(output_before_batchnorm_folding, output_after_batchnorm_folding, atol=1e-4)

    def test_bn_fold_with_kwargs_from_layer(self):
        indices_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='indices')
        updates_input = tf.keras.layers.Input(shape=(None,), dtype=tf.float32, name='updates')
        shape_1 = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='shape_first_value')
        conv_inp_layer = tf.keras.layers.Input(shape=(224, 224, 3), dtype=tf.float32)

        conv_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="valid")(conv_inp_layer)
        bn_layer = tf.keras.layers.BatchNormalization()(conv_layer)
        relu = tf.keras.layers.ReLU()(bn_layer)
        scatter_nd_layer = tf.scatter_nd(indices_input, updates_input, shape=[shape_1[0], 4, 4])

        scatter_model = tf.keras.models.Model(inputs=[conv_inp_layer, indices_input, updates_input, shape_1],
                                              outputs=[relu, scatter_nd_layer])

        conv_inp = np.random.randn(1, 224, 224, 3)
        indices = tf.constant([[1], [3]], dtype=tf.int32)
        updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                               [7, 7, 7, 7], [8, 8, 8, 8]],
                              [[5, 5, 5, 5], [6, 6, 6, 6],
                               [7, 7, 7, 7], [8, 8, 8, 8]]], dtype=tf.float32)
        shape_first = tf.constant((4,), dtype=tf.int32)

        fp32_outs = scatter_model([conv_inp, indices, updates, shape_first])

        _, bn_folded_model = fold_all_batch_norms(scatter_model)

        bn_outs = bn_folded_model([conv_inp, indices, updates, shape_first])

        np.testing.assert_allclose(fp32_outs[0], bn_outs[0], atol=1e-4)
        np.testing.assert_allclose(fp32_outs[1], bn_outs[1], atol=1e-4)

    @pytest.mark.parametrize('set_dtype', [tf.float32, tf.float16, tf.int64, tf.int32])
    def test_cast_op_with_same_dtypes(self, set_dtype):
        input = tf.keras.layers.Input(shape=(7, 7, 3))
        conv1 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="valid")(input)
        bn1 = tf.keras.layers.BatchNormalization()(conv1)
        conv2 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="valid")(input)
        bn2 = tf.keras.layers.BatchNormalization()(conv2)
        cast1 = tf.cast(bn1, dtype=set_dtype)
        cast2 = tf.cast(bn2, dtype=set_dtype)
        concat = tf.concat([cast1, cast2], axis=-1)
        model = tf.keras.models.Model(inputs=[input], outputs=[concat])

        dummy_inp = np.random.randn(1, 7, 7, 3)
        outs = model(dummy_inp)

        _, bn_folded_model = fold_all_batch_norms(model)
        bn_outs = bn_folded_model(dummy_inp)

        assert np.allclose(outs, bn_outs, atol=1e-04)

        # Number of cast ops present in BatchNorm folded model
        num_cast_ops = ["cast" in layer.name for layer in bn_folded_model.layers]
        if set_dtype == tf.float32:
            assert not any(num_cast_ops)
        else:
            # Only two cast ops should be present as the original model has two cast ops
            assert sum(num_cast_ops) == 2

    def test_split_op_model(self):
        inp = tf.keras.layers.Input(shape=(3, 224, 224))
        conv = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="valid")(inp)
        bn = tf.keras.layers.BatchNormalization()(conv)
        split, split2 = tf.split(bn, num_or_size_splits=2, axis=2)
        split_3, split4 = tf.split(bn, num_or_size_splits=2, axis=2)
        concat = tf.concat([split2, split_3], axis=1)
        concat_2 = tf.concat([split, split4, split_3], axis=1)
        model = tf.keras.models.Model(inputs=inp, outputs=[concat_2, concat])

        dummy_inp = np.random.randn(1, 3, 224, 224)

        fp32_outs = model(dummy_inp)

        folded_pairs, bn_folded_model = fold_all_batch_norms(model)

        bn_folded_outs = bn_folded_model(dummy_inp)

        for idx, fp32_out in enumerate(fp32_outs):
            assert np.allclose(fp32_out, bn_folded_outs[idx], atol=1e-04)

        split_0_output_tensor_names = [x.name for x in bn_folded_model.layers[2].output]
        split_1_output_tensor_names = [x.name for x in bn_folded_model.layers[3].output]

        assert bn_folded_model.layers[4].input[0].name == split_0_output_tensor_names[0]
        assert bn_folded_model.layers[4].input[1].name == split_1_output_tensor_names[1]
        assert bn_folded_model.layers[4].input[2].name == split_1_output_tensor_names[0]

        assert bn_folded_model.layers[5].input[0].name == split_0_output_tensor_names[1]
        assert bn_folded_model.layers[5].input[1].name == split_1_output_tensor_names[0]

    @pytest.mark.skip("Possible Batch norms to fold is returning None?")
    def test_fold_auto_mode_with_bn_after_Conv1d_layer(self):
        input_shape = (2, 10, 32)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.Conv1D(20, 2)(inp)
        x = tf.keras.layers.BatchNormalization()(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])
        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.layers[1].output_quantizers[0].is_enabled()
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()
        assert model.layers[2].output_quantizers[0].is_enabled()
        assert not np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[2].param_quantizers)))

        baseline_output = model(random_input)
        bn_pairs = fold_all_batch_norms_to_scale(sim)
        model = sim.model
        output_after_fold = model(random_input)

        # Check to make sure rebuild Quantization Sim Model working properly
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(path=tmp_dir, filename_prefix="temp_bn_fold_to_scale")

        # Check bn is deleted
        for wrapper in model.layers[1:]:
            assert not isinstance(wrapper._layer_to_wrap, tf.keras.layers.BatchNormalization)

        # Check quantizers are enabled/disabled proeprly
        self._test_output_quantizers_enabled_and_quant_mode_is_quantize_dequantize(model, layer_nums_to_check=[1])
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()

        conv_output_encoding = model.layers[1].output_quantizers[0].encoding
        delta = float((conv_output_encoding.max - conv_output_encoding.min)/255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference
        assert len(bn_pairs) == 1

    @pytest.mark.skip("Conv1D not in _supported_layers")
    def test_fold_manual_with_bn_after_Conv1d_layer_no_bias(self):
        input_shape = (2, 10, 32)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.Conv1D(20, 2, use_bias=False)(inp)
        x = tf.keras.layers.BatchNormalization()(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])
        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert not model.layers[1].output_quantizers[0].is_enabled()
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()
        assert model.layers[2].output_quantizers[0].is_enabled()
        assert not np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[2].param_quantizers)))

        baseline_output = model(random_input)
        layer_list = [(model.layers[1], model.layers[2])]
        model = fold_given_batch_norms(model, layer_list)
        output_after_fold = model(random_input)

        # Check bn is deleted
        for wrapper in model.layers[1:]:
            assert not isinstance(wrapper._layer_to_wrap, tf.keras.layers.BatchNormalization)

        # Check quantizers are enabled/disabled proeprly
        self._test_output_quantizers_enabled_and_quant_mode_is_quantize_dequantize(model, layer_nums_to_check=[1])
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()

        conv_output_encoding = model.layers[1].output_quantizers[0].encoding
        delta = float((conv_output_encoding.max - conv_output_encoding.min)/255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference

    @pytest.mark.skip("Conv1D not found in potential BN layers")
    def test_fold_bn_before_Conv1d_with_bias(self):
        input_shape = (2, 10, 32)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.BatchNormalization()(inp)
        x = tf.keras.layers.Conv1D(20, 2)(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])
        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert model.layers[2].output_quantizers[0].is_enabled()
        assert np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[2].param_quantizers)))

        assert model.layers[1].output_quantizers[0].is_enabled()
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()

        with pytest.raises(RuntimeError):
            fold_all_batch_norms_to_scale(sim)

    @pytest.mark.skip("Conv1D not in supported layers")
    def test_fold_bn_before_Conv1d_no_bias(self):
        input_shape = (2, 10, 32)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.BatchNormalization()(inp)
        x = tf.keras.layers.Conv1D(20, 2, use_bias=False)(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])
        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        sim = create_quantsim_model_and_compute_encodings(model, random_input)
        model = sim.model

        # Check quantizers are enabled/disabled properly
        assert model.layers[2].output_quantizers[0].is_enabled()
        assert np.all(np.vectorize(lambda x: x.is_enabled())(
            get_wrappers_weight_quantizer(model.layers[2].param_quantizers)))

        assert model.layers[1].output_quantizers[0].is_enabled()
        assert get_wrappers_weight_quantizer(model.layers[1].param_quantizers).is_enabled()

        layer_list = [(model.layers[1], model.layers[2])]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    @pytest.mark.skip("Conv3D not found in possible layers")
    def test_bn_fold_conv3d_fold_backward(self):
        input_shape = (1, 24, 24, 24, 3)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.Conv3D(6, 3)(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv3D(8, 3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])
        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        baseline_output = model(random_input)
        _ = fold_all_batch_norms(model)
        output_after_fold = model(random_input)

        assert np.allclose(baseline_output, output_after_fold, atol=1e-5)

        # Check bn is deleted
        for wrapper in model.layers[1:]:
            assert not isinstance(wrapper._layer_to_wrap, tf.keras.layers.BatchNormalization)

    @pytest.mark.skip("Conv3D not found in possible layers")
    def test_bn_fold_conv3d_fold_forward(self):
        input_shape = (1, 24, 24, 24, 3)

        inp = tf.keras.Input(input_shape[1:])
        x = tf.keras.layers.Conv3D(6, 3)(inp)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv3D(8, 3)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        model = tf.keras.Model(inputs=[inp], outputs=[x])
        random_input = np.random.rand(*input_shape)
        _ = model(random_input)

        baseline_output = model(random_input)
        _, model = fold_all_batch_norms(model)
        output_after_fold = model(random_input)

        assert np.allclose(baseline_output, output_after_fold, atol=1e-5)

        # Check bn is deleted
        for wrapper in model.layers[1:]:
            assert not isinstance(wrapper._layer_to_wrap, tf.keras.layers.BatchNormalization)
