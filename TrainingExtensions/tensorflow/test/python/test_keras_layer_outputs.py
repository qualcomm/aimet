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

import re
import json
import shutil
from glob import glob
from datetime import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Conv2D, Flatten, AvgPool2D, MaxPool2D
from aimet_tensorflow.keras.quantsim import QuantizationSimModel, QcQuantizeWrapper
from aimet_tensorflow.keras.model_preparer import prepare_model
from aimet_tensorflow.keras.layer_output_utils import LayerOutputUtil

class DummyDataLoader(tf.keras.utils.Sequence):

    def __init__(self, batch_size, data_count):
        self.batch_size = batch_size
        self.data_count = data_count
        self.data = [np.random.rand(16, 16, 3).astype(np.float32) for _ in range(data_count)]

    def on_epoch_end(self):
        # if anything needs to be done in between two epochs
        pass

    def __getitem__(self, index):
        return tf.convert_to_tensor(self.data[index*self.batch_size: (index+1)*self.batch_size], dtype=np.float32), tf.convert_to_tensor([np.random.choice([0, 1]) for _ in range(self.batch_size)], dtype=np.int32)

    def __len__(self):
        return int(np.ceil(self.data_count / self.batch_size))

def dummy_forward_pass(model, input_batch):
    _ = model.predict(input_batch)

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
        Dense(2, activation='softmax')])
    return model

def get_quantsim_artifacts(base_model):
    # Using model preparer
    base_model = prepare_model(base_model)
    dummy_input = np.random.rand(1, 16, 16, 3)

    sim = QuantizationSimModel(
        model=base_model,
        quant_scheme='tf_enhanced',
        rounding_mode="nearest",
        default_output_bw=8,
        default_param_bw=8,
        in_place=False,
        config_file=None
    )

    sim.trainable = False

    sim.compute_encodings(dummy_forward_pass,
                          forward_pass_callback_args=dummy_input
                          )
    return sim

class TestLayerOutputUtil:
    def test_generate_layer_output(self):
        """ Test whether layer-output files are correctly saved and are getting correctly loaded """

        # Load the baseline model
        base_model = keras_model()

        # Get the QuantSim artifacts
        qs_obj = get_quantsim_artifacts(base_model)

        # Temporary output path to store outputs temporarily
        temp_folder_name = f"temp_keras_layer_output_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        save_dir = os.path.join(os.getcwd(), temp_folder_name)

        # Required params for the layer output generation function
        data_points = 4
        batch_size = 3

        # Get the dataLoader
        dataloader = DummyDataLoader(data_count=data_points, batch_size=batch_size)

        layer_output_util_obj = LayerOutputUtil(model=qs_obj.model, save_dir=save_dir)
        for batch_num, inp_batch in enumerate(dataloader):
            batch_x, _ = inp_batch
            for single_input in batch_x:
                layer_output_util_obj.generate_layer_outputs(np.expand_dims(single_input, axis=0))

        # Verify number of inputs
        assert data_points == len(glob(save_dir+"/inputs/*.raw")) ## Check #Inputs

        # Verify number of layer outputs
        assert data_points == len(glob(save_dir+"/outputs/*")) ## Check #Outputs

        # Getting the actual layer output names
        actual_layer_output_names = list()
        unmodified_actual_layer_output_names = list()
        for each_layer in qs_obj.model.layers:
            layer_output_name = each_layer.output.name
            if isinstance(each_layer, QcQuantizeWrapper):
                layer_output_name = each_layer.original_layer.output.name
            unmodified_actual_layer_output_names.append(layer_output_name)
            layer_output_name = re.sub(r'\W+', "_", layer_output_name)
            actual_layer_output_names.append(layer_output_name)

        # Getting the saved layer output names
        saved_layer_output_list = list()
        for each_layer_output_name in glob(save_dir+"/outputs/layer_outputs_0/*.raw"):
            each_layer_output_name = each_layer_output_name.split("/")[-1][:-4]
            saved_layer_output_list.append(each_layer_output_name)

        # Verify layer-output names
        np.testing.assert_array_equal(np.array(sorted(actual_layer_output_names)), np.array(sorted(saved_layer_output_list)))

        # Verify final layer output for all data points
        cnt = 0
        n_iterations = np.ceil(data_points/batch_size)
        for batch_num, input_batch in enumerate(dataloader):
            batch_x, _ = input_batch
            for inp_batch in batch_x:
                actual_output = qs_obj.model.predict(np.expand_dims(inp_batch, axis=0))
                last_layer_name = qs_obj.model.layers[-1].original_layer.output.name
                last_layer_name = re.sub(r'\W+', "_", last_layer_name)
                last_layer_file_name = f"{save_dir}/outputs/layer_outputs_{cnt}/{last_layer_name}.raw"
                saved_last_layer_output = np.fromfile(last_layer_file_name, dtype=np.float32)
                np.testing.assert_array_equal(actual_output[0], saved_last_layer_output)
                cnt += 1

            if (batch_num+1) >= n_iterations:
                break

        # Test the layer output name mapper file and dict
        saved_layer_output_name_mapper = json.load(open(temp_folder_name+"/LayerOutputNameMapper.json", "r"))

        # Verify number of layers
        np.testing.assert_array_equal(np.array(unmodified_actual_layer_output_names),
                                      np.array(list(saved_layer_output_name_mapper.keys())))

        # Verify modified layer name for each of the layers
        for layer_idx, unmodified_actual_layer_name in enumerate(unmodified_actual_layer_output_names):

            # Test saved file layer output
            assert actual_layer_output_names[layer_idx] == \
                   saved_layer_output_name_mapper[unmodified_actual_layer_name]

            # Test dict layer output
            assert actual_layer_output_names[layer_idx] == \
                   layer_output_util_obj.original_name_to_modified_name_mapper[unmodified_actual_layer_name]

        # Removing the temporary output that was created (if all tests are passed)
        shutil.rmtree(save_dir)

    def test_get_quantsim_outputs(self):
        base_model = keras_model()
        qs_obj = get_quantsim_artifacts(base_model)

        qs_model_actual_output_names = list()

        # Getting the actual layer output names
        for each_layer in qs_obj.model.layers:
            layer_output_name = each_layer.output.name
            if isinstance(each_layer, QcQuantizeWrapper):
                layer_output_name = re.sub(r'\W+', "_", each_layer.original_layer.output.name)
            qs_model_actual_output_names.append(layer_output_name)

        layer_output_obj = LayerOutputUtil(qs_obj.model)
        batch_size = 3
        data_count = 4
        dataloader = DummyDataLoader(batch_size, data_count)
        layer_output_dict = layer_output_obj.get_outputs(dataloader[0][0])

        # Verify whether outputs are generated for all the layers
        for each_actual_output_name in qs_model_actual_output_names:
            assert each_actual_output_name in layer_output_dict.keys(), f"Output not generated for " \
                                                                        f"{each_actual_output_name}"

        # Verify the final layer output
        actual_output = qs_obj.model.predict(dataloader[0][0])
        np.testing.assert_array_equal(actual_output, layer_output_dict[qs_model_actual_output_names[-1]])

    def test_get_original_model_outputs(self):
        base_model = keras_model()

        base_model_actual_output_names = list()

        for each_layer in base_model.layers:
            layer_output_name = re.sub(r'\W+', "_", each_layer.output.name)
            base_model_actual_output_names.append(layer_output_name)

        layer_output_obj = LayerOutputUtil(base_model)
        batch_size = 3
        data_count = 4
        dataloader = DummyDataLoader(batch_size, data_count)
        layer_output_dict = layer_output_obj.get_outputs(dataloader[0][0])

        # Verify whether outputs are generated for all the layers
        for each_actual_output_name in base_model_actual_output_names:
            assert each_actual_output_name in layer_output_dict.keys(), f"Output not generated for " \
                                                                        f"{each_actual_output_name}"

        # Verify the final layer output
        actual_output = base_model.predict(dataloader[0][0])
        np.testing.assert_array_equal(actual_output, layer_output_dict[base_model_actual_output_names[-1]])
