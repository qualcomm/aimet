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

""" This module contains a common utility class for saving outputs of intermediate layers to disk """

import os
from typing import Union, List, Tuple
import json
import numpy as np


class SaveInputOutput:
    """ This class saves the input instance and corresponding layer-outputs to the disk. """

    def __init__(self, dir_path: str, axis_layout: str):
        """
        Constructor
        :param dir_path: Directory to save input and output.
        :param axis_layout: Axis-layout of the tensor in source framework. Eg: NCHW or NHWC
        """
        self.dir_path = dir_path
        self.input_cntr = 0
        self.axis_layout = axis_layout

    @staticmethod
    def transform_axis(tensor: np.ndarray, old_layout: str, new_layout: str) -> np.ndarray:
        """
        This function transforms the axis-layout of given tensor from old format to new format.
        :param tensor: Four dimensional numpy array.
        :param old_layout: Axis-layout (eg: 'NCHW')
        :param new_layout: Axis-layout (eg: 'NHWC')
        :return: Transformed tensor
        """
        new_layout_idx = [old_layout.index(ax) for ax in new_layout]
        return np.transpose(tensor, new_layout_idx)

    @staticmethod
    def save_raw_tensor(numpy_tensor: np.ndarray, axis_layout: str, file_name: str, dir_path: str):
        """
        This function saves the tensor into a raw file. The axis-layout of the saved tensor is NHWC.
        :param numpy_tensor: Tensor to save.
        :param axis_layout: Axis-layout of the tensor.
        :param file_name: Name to be given to the raw tensor file.
        :param dir_path: Directory wherein the file has to be stored.
        :return:
        """
        file_path = os.path.join(dir_path, file_name + '.raw')

        # Convert axis-layout of the tensor to NHWC if not already
        if numpy_tensor.ndim == 4 and axis_layout == 'NCHW':
            numpy_tensor = SaveInputOutput.transform_axis(tensor=numpy_tensor, old_layout='NCHW', new_layout='NHWC')

        with open(file_path, 'wb') as fptr:
            numpy_tensor.tofile(fptr)

    def save(self, input_instance: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]], layer_output: dict):
        """
        This function saves the input and layer-outputs in the form of raw files. Separate directories are used to store inputs
        and outputs. The correspondence between input and output is obtained using an identical number which is used for naming.

        :param input_instance: Input instance for which we want to obtain layer-outputs.
        :param layer_output: Dictionary where key is output-name and value is output(s).
        :return:
        """
        input_dir = os.path.join(self.dir_path, 'inputs')
        output_dir = os.path.join(self.dir_path, 'outputs')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(input_instance, (List, Tuple)):
            multi_input_dir = os.path.join(input_dir, 'input_' + str(self.input_cntr))
            os.makedirs(multi_input_dir, exist_ok=True)
            for i, ith_input in enumerate(input_instance):
                SaveInputOutput.save_raw_tensor(ith_input, self.axis_layout, str(i), multi_input_dir)
        else:
            input_file_name = 'input_' + str(self.input_cntr)
            SaveInputOutput.save_raw_tensor(input_instance, self.axis_layout, input_file_name, input_dir)

        layer_output_dir = os.path.join(output_dir, 'layer_outputs_' + str(self.input_cntr))
        os.makedirs(layer_output_dir, exist_ok=True)
        for layer_output_name in layer_output:
            SaveInputOutput.save_raw_tensor(layer_output[layer_output_name], self.axis_layout, layer_output_name, layer_output_dir)

        self.input_cntr += 1


def save_layer_output_names(layer_output_names: list, dir_path: str):
    """
    This function saves layer-output names into a json file.
    :param layer_output_names: Dictionary of layer-name to layer-output name
    :param dir_path: Directory to save json file
    :return:
    """
    os.makedirs(dir_path, exist_ok=True)
    json_file_path = os.path.join(dir_path, 'layer_output_name_order.json')
    with open(json_file_path, 'w') as fptr:
        json.dump({'layer_output_names': layer_output_names}, fptr, indent=4)
