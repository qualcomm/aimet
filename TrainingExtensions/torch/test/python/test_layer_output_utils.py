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
import pytest
import os
import re
import shutil
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import tempfile

from aimet_torch.model_preparer import prepare_model
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.v1.quantsim import QuantizationSimModel as QuantizationSimModelV1
from aimet_torch.v2.quantsim import QuantizationSimModel as QuantizationSimModelV2
from aimet_torch.layer_output_utils import NamingScheme, LayerOutputUtil, LayerOutput
from aimet_torch.utils import is_leaf_module
from aimet_torch.onnx_utils import OnnxExportApiArgs
from .models import test_models


def dummy_forward_pass(model: torch.nn.Module, input_batch: torch.Tensor):
    model.eval()
    with torch.no_grad():
        _ = model(input_batch)


def get_original_model_artifacts():
    # Load resnet18 model
    model = models.resnet18()
    model.eval()
    dummy_input = torch.rand(1, 3, 224, 224)

    # Prepare model for quantization simulation
    model = prepare_model(model)
    ModelValidator.validate_model(model, dummy_input)

    # Obtain layer-names of original model
    layer_names = []
    for name, module in model.named_modules():
        if is_leaf_module(module):
            layer_names.append(name)

    return model, layer_names, dummy_input


def get_quantsim_artifacts(_QuantizationSimModel, model_and_input=None):
    if model_and_input is not None:
        model, dummy_input = model_and_input
    else:
        # Load resnet18 model
        model = models.resnet18()
        model.eval()
        dummy_input = torch.rand(1, 3, 224, 224)

    # Prepare model for quantization simulation
    model = prepare_model(model)
    ModelValidator.validate_model(model, dummy_input)

    # Obtain layer-names of original model
    layer_names = []
    for name, module in model.named_modules():
        if is_leaf_module(module):
            layer_names.append(name)

    # Obtain quantsim model
    quantsim = _QuantizationSimModel(model=model, quant_scheme='tf_enhanced',
                                     dummy_input=dummy_input, rounding_mode='nearest',
                                     default_output_bw=8, default_param_bw=8, in_place=False)

    quantsim.compute_encodings(forward_pass_callback=dummy_forward_pass,
                               forward_pass_callback_args=dummy_input)

    return quantsim, layer_names, dummy_input


class TestLayerOutput:
    def test_get_original_model_outputs(self):
        """ Test whether outputs are generated for all the layers of an original model """

        # Get original model artifacts
        original_model, layer_names, dummy_input = get_original_model_artifacts()
        layer_names = [re.sub(r'\W+', "_", name) for name in layer_names]

        temp_dir_path = os.path.dirname(os.path.abspath(__file__))
        temp_dir_path = os.path.join(temp_dir_path, 'temp_dir')

        # Obtain layer-outputs of quantsim model
        layer_output = LayerOutput(model=original_model, dir_path=temp_dir_path, naming_scheme=NamingScheme.PYTORCH)
        name_to_output_dict = layer_output.get_outputs(dummy_input)

        # Verify whether outputs are generated for all the layers
        for layer_name in layer_names:
            assert layer_name in name_to_output_dict, \
                "Output not generated for layer " + layer_name

        # Verify whether outputs are correct. This can only be checked for final output of the model
        assert torch.equal(original_model(dummy_input), name_to_output_dict['fc']), \
            "Output of last layer of original model doesn't match with captured layer-output"

        # Delete temp_dir
        shutil.rmtree(temp_dir_path, ignore_errors=False, onerror=None)

    @pytest.mark.parametrize('_QuantizationSimModel', [QuantizationSimModelV1, QuantizationSimModelV2])
    def test_get_quantsim_outputs(self, _QuantizationSimModel):
        """ Test whether outputs are generated for all the layers of a quantsim model """

        # Get quantsim artifacts
        quantsim, layer_names, dummy_input = get_quantsim_artifacts(_QuantizationSimModel)
        layer_names = [re.sub(r'\W+', "_", name) for name in layer_names]

        temp_dir_path = os.path.dirname(os.path.abspath(__file__))
        temp_dir_path = os.path.join(temp_dir_path, 'temp_dir')

        # Obtain layer-outputs of quantsim model
        layer_output = LayerOutput(model=quantsim.model, dir_path=temp_dir_path, naming_scheme=NamingScheme.PYTORCH)
        name_to_output_dict = layer_output.get_outputs(dummy_input)

        # Verify whether outputs are generated for all the layers
        for layer_name in layer_names:
            assert layer_name in name_to_output_dict, \
                "Output not generated for layer " + layer_name

        # Verify whether outputs are quantized outputs. This can only be checked for final output of the model
        assert torch.equal(quantsim.model(dummy_input), name_to_output_dict['fc']), \
            "Output of last layer of quantsim model doesn't match with captured layer-output"

        # Delete temp_dir
        shutil.rmtree(temp_dir_path, ignore_errors=False, onerror=None)

    @pytest.mark.parametrize('_QuantizationSimModel', [QuantizationSimModelV1, QuantizationSimModelV2])
    def test_layer_name_to_onnx_layer_output_name_dict(self, _QuantizationSimModel):
        """ Test whether every layer-name has corresponding onnx layer-output-name """

        # Get quantsim artifacts
        quantsim, layer_names, dummy_input = get_quantsim_artifacts(_QuantizationSimModel)

        temp_dir_path = os.path.dirname(os.path.abspath(__file__))
        temp_dir_path = os.path.join(temp_dir_path, 'temp_dir')

        # Obtain layer-output names of the onnx model used in layer-output utils api
        layer_output = LayerOutput(model=quantsim.model, dir_path=temp_dir_path, naming_scheme=NamingScheme.ONNX,
                                   dummy_input=dummy_input, onnx_export_args=OnnxExportApiArgs())

        # Check if every layer-name has corresponding onnx layer-output-name
        for layer_name in layer_names:
            assert layer_name in layer_output.layer_name_to_layer_output_name_dict, \
                "Missing onnx output name for layer "+layer_name

        # Delete temp_dir
        shutil.rmtree(temp_dir_path, ignore_errors=False, onerror=None)

    def test_layer_name_to_torchscript_layer_output_name_dict(self):
        """
        Test whether every layer-name has corresponding torchscript layer-output-name

        TODO: Implement this unit-test once the bug in 'quantsim.QuantizationSimModel.export_torch_script_model_and_encodings()' is fixed.
            Bug: The number of items in 'torch_script_node_io_tensor_map' is less than the number of layers in the pytorch model.
            Because not all layers have corresponding torchscript layer-output names, this unit-test will fail if implemented before bug-fix.
        """
        pass


def get_dataset_artifacts():
    class DummyDataset(Dataset):
        def __init__(self, count):
            Dataset.__init__(self)
            self.random_img = []
            while count:
                self.random_img.append(torch.rand(3, 224, 224))
                count -= 1

        def __len__(self):
            return len(self.random_img)

        def __getitem__(self, idx):
            return self.random_img[idx]

    data_count = 4
    dummy_dataset = DummyDataset(data_count)
    dummy_dataloader = DataLoader(dataset=dummy_dataset, batch_size=2)

    return dummy_dataset, dummy_dataloader, data_count


class TestLayerOutputUtil:
    @pytest.mark.parametrize('_QuantizationSimModel', [QuantizationSimModelV1, QuantizationSimModelV2])
    def test_generate_layer_outputs(self, _QuantizationSimModel):
        """ Test whether input files and corresponding layer-output files are generated """

        # Get quantsim artifacts
        quantsim, layer_output_names, dummy_input = get_quantsim_artifacts(_QuantizationSimModel)
        layer_output_names = [re.sub(r'\W+', "_", name) for name in layer_output_names]

        # Get dataset artifacts
        dummy_dataset, dummy_data_loader, data_count = get_dataset_artifacts()

        temp_dir_path = os.path.dirname(os.path.abspath(__file__))
        temp_dir_path = os.path.join(temp_dir_path, 'temp_dir')

        # Generate layer-outputs
        layer_output_util = LayerOutputUtil(model=quantsim.model, dir_path=temp_dir_path)
        for input_batch in dummy_data_loader:
            for single_input in input_batch:
                layer_output_util.generate_layer_outputs(single_input.unsqueeze(0))

        # Verify number of inputs
        assert data_count == len(os.listdir(os.path.join(temp_dir_path, 'inputs')))

        # Verify number of layer-output folders
        assert data_count == len(os.listdir(os.path.join(temp_dir_path, 'outputs')))

        # Verify number of layer-outputs
        saved_layer_outputs = os.listdir(os.path.join(temp_dir_path, 'outputs', 'layer_outputs_0'))
        saved_layer_outputs = [i[:-len('.raw')] for i in saved_layer_outputs]
        for name in layer_output_names:
            assert name in saved_layer_outputs

        # Ensure generated layer-outputs can be correctly loaded for layer-output comparison
        saved_last_layer_output = np.fromfile(os.path.join(temp_dir_path, 'outputs', 'layer_outputs_0', 'fc.raw'), dtype=np.float32).reshape((1, 1000))
        saved_last_layer_output = torch.from_numpy(saved_last_layer_output)
        last_layer_output = quantsim.model(torch.unsqueeze(dummy_dataset.__getitem__(0), dim=0))
        assert torch.equal(last_layer_output, saved_last_layer_output)

        # Delete temp_dir
        shutil.rmtree(temp_dir_path, ignore_errors=False, onerror=None)

    @pytest.mark.parametrize('_QuantizationSimModel', [QuantizationSimModelV1, QuantizationSimModelV2])
    def test_layer_output_axis_transform(self, _QuantizationSimModel):
        """ Test to ensure the layer outputs are saved with axis-transform """

        model = test_models.ModelWithOneSplit().to(device='cpu')
        input_shapes = (1, 1, 10, 10)
        dummy_input = torch.randn(input_shapes)
        qsim, layer_names, _ = get_quantsim_artifacts(_QuantizationSimModel, model_and_input=(model, dummy_input))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate layer-outputs
            layer_output_util = LayerOutputUtil(model=qsim.model, dir_path=tmpdir)
            for single_input in dummy_input:
                layer_output_util.generate_layer_outputs(single_input.unsqueeze(0))

            for conv_layer_name in layer_names:
                # Output in NCHW format
                layer_output_from_dict = layer_output_util.layer_output.layer_name_to_layer_output_dict[conv_layer_name]

                # Convert from NCHW to NHWC
                layer_output_from_dict = layer_output_from_dict.detach().permute(0, 2, 3, 1)

                # Saved output should already be in NHWC format
                saved_output = np.fromfile(tmpdir+f"/outputs/layer_outputs_0/{conv_layer_name}.raw",
                                           dtype=np.float32).reshape(layer_output_from_dict.shape)
                saved_output = torch.from_numpy(saved_output)

                assert torch.equal(layer_output_from_dict, saved_output)
