# /usr/bin/env python
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

import tempfile
import pytest
import torch
from torch.utils.data import Dataset, DataLoader
import copy
import itertools
import json
import unittest.mock
import numpy as np
import os
import shutil
import onnx

from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.sequential_mse.dependency_graph_utils import DependencyGraphUtils
from aimet_onnx.sequential_mse.dependency_graph import DependencyGraph
from aimet_onnx.sequential_mse.dependency_graph import DependencyNode

from aimet_onnx.sequential_mse.seq_mse import SeqMseParams
from aimet_onnx.sequential_mse.seq_mse import SequentialMse
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel

from models.test_models import single_linear_layer_model
from models.test_models import single_conv_layer_model
from models.test_models import model_with_split
from models.test_models import single_residual_model
from models.test_models_onnx import model_with_multiple_inputs
from models.test_models_onnx import model_with_multiple_outputs


torch.manual_seed(42)

def unlabeled_data_loader(dummy_input):
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return tuple(d[index] for d in self.data)

        def __len__(self):
            return len(self.data[0])
    dataset = MyDataset(dummy_input)
    return DataLoader(dataset)


def dummy_input_for_linear_layer():
    return [torch.randn((1, 100, 100))]


def dummy_input_for_conv_layer():
    return [torch.randn((1, 5, 5, 5))]


def dummy_input_for_dependency_graph():
    return [torch.randn((1, 1, 10, 10))]


def dummy_input_for_residual_model():
    return [torch.randn((1, 3, 32, 32))]


def dummy_input_for_model_with_multiple_input():
    return [torch.randn((1, 3, 32, 32)), torch.randn((1, 3, 32, 32))]


def get_single_linear_layer_model():
    return single_linear_layer_model()


def get_single_conv_layer_model():
    return single_conv_layer_model()


def get_model_with_split():
    return model_with_split()


def get_model_with_multiple_inputs():
    return model_with_multiple_inputs()


def get_model_with_multiple_outputs():
    return model_with_multiple_outputs()

@staticmethod
def _get_config_file(is_symmetric: bool, strict_symmetric: bool, unsigned_symmetric:bool, pcq: bool) -> str:
    """ Temporary fix until the config file can be read from beq_config directory"""

    def get_bool_str(in_bool: bool) -> str:
        if in_bool:
            return "True"
        else:
            return "False"

    beq_per_channel_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": get_bool_str(is_symmetric)
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": get_bool_str(is_symmetric)
            },
            "strict_symmetric": get_bool_str(strict_symmetric),
            "unsigned_symmetric": get_bool_str(unsigned_symmetric),
            "per_channel_quantization": get_bool_str(pcq)
        },
        "params": {
            "bias": {
                "is_quantized": "True"
            }
        },
        "op_type": {
            "PRelu": {
                "params": {
                    "weight": {
                        "is_quantized": "False"
                    }
                }
            }
        },
        "supergroups": [
            {
                "op_list": [
                    "Gemm",
                    "PRelu"
                ]
            },
            {
                "op_list": [
                    "Gemm",
                    "Sigmoid"
                ]
            },
            {
                "op_list": [
                    "Conv",
                    "PRelu"
                ]
            },
            {
                "op_list": [
                    "Conv",
                    "Sigmoid"
                ]
            }
        ],
        "model_input": {
            "is_input_quantized": "True"
        },
        "model_output": {}
    }

    if not os.path.exists("data"):
        os.mkdir("data")
    file_name = './data/beq_per_channel_config.json'
    with open(file_name, 'w') as f:
        json.dump(beq_per_channel_config, f)

    return file_name


@pytest.mark.parametrize("inp_symmetry", ['asym', 'symfp', 'symqt'])
@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ['mse', 'l1'])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_do_seq_mse_for_conv(inp_symmetry, param_bw, loss_fn, enable_pcq):
    model = single_conv_layer_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=param_bw,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=enable_pcq))
    seq_params = SeqMseParams()
    seq_params.loss_fn = loss_fn
    seq_params.inp_symmetry = inp_symmetry
    dataloader = unlabeled_data_loader(dummy_input_for_conv_layer())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    seq_mse.dependency_graph = seq_mse.dependency_graph_utils.create_dependency_graph(dataloader, seq_params.num_batches)
    conv_node = seq_mse.dependency_graph.node_by_name['/conv/Conv']
    seq_mse._do_seq_mse(conv_node)
    _, per_channel_max = seq_mse._get_min_max_from_weights(conv_node)
    if not enable_pcq:
        per_channel_max = max(per_channel_max)

    weight_name = seq_mse.node_name_to_input_names[conv_node.op_name][1]
    quantize_op = seq_mse.sim.qc_quantize_op_dict[weight_name]
    encodings = quantize_op.encodings
    encodings_max = [encoding.max for encoding in encodings]
    if param_bw == 31:
        assert np.all(np.isclose(encodings_max, per_channel_max))
    else:
        assert not np.all(np.isclose(encodings_max, per_channel_max))


@pytest.mark.parametrize("inp_symmetry", ['asym', 'symfp', 'symqt'])
@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_do_seq_mse_for_linear(inp_symmetry, param_bw, loss_fn, enable_pcq):
    model = get_single_linear_layer_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=param_bw,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams()
    seq_params.loss_fn = loss_fn
    seq_params.inp_symmetry = inp_symmetry
    dataloader = unlabeled_data_loader(dummy_input_for_linear_layer())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    seq_mse.dependency_graph = seq_mse.dependency_graph_utils.create_dependency_graph(dataloader, seq_params.num_batches)
    fc_node = seq_mse.dependency_graph.node_by_name['/fc/MatMul']
    seq_mse._do_seq_mse(fc_node)
    _, per_channel_max = seq_mse._get_min_max_from_weights(fc_node)
    weight_name = seq_mse.node_name_to_input_names[fc_node.op_name][1]
    quantize_op = seq_mse.sim.qc_quantize_op_dict[weight_name]
    encodings = quantize_op.encodings
    encodings_max = [encoding.max for encoding in encodings]
    if param_bw == 31:
        assert np.all(np.isclose(encodings_max, per_channel_max))
    else:
        assert not np.all(np.isclose(encodings_max, per_channel_max))


@pytest.mark.parametrize("inp_symmetry", ['asym', 'symfp', 'symqt'])
@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_apply_seq_mse_for_conv(inp_symmetry, param_bw, loss_fn, enable_pcq):
    model = get_single_conv_layer_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=param_bw,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams()
    seq_params.loss_fn = loss_fn
    seq_params.inp_symmetry = inp_symmetry
    dataloader = unlabeled_data_loader(dummy_input_for_conv_layer())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    seq_mse.apply_seq_mse_algo()
    weight_quantizer = seq_mse.sim.qc_quantize_op_dict['conv.weight']
    assert weight_quantizer._is_encoding_frozen


@pytest.mark.parametrize("inp_symmetry", ['asym', 'symfp', 'symqt'])
@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_static_apply_seq_mse(inp_symmetry, param_bw, loss_fn, enable_pcq):
    model = get_single_conv_layer_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=param_bw,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams()
    seq_params.loss_fn = loss_fn
    seq_params.inp_symmetry = inp_symmetry
    dataloader = unlabeled_data_loader(dummy_input_for_conv_layer())
    SequentialMse.apply_seq_mse(model, sim, seq_params, dataloader)


@pytest.mark.parametrize("inp_symmetry", ['asym', 'symfp', 'symqt'])
@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_apply_seq_mse_for_split(inp_symmetry, param_bw, loss_fn, enable_pcq):
    model = get_model_with_split()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=param_bw,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams()
    seq_params.loss_fn = loss_fn
    seq_params.inp_symmetry = inp_symmetry
    dataloader = unlabeled_data_loader(dummy_input_for_dependency_graph())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    seq_mse.apply_seq_mse_algo()

    weight_quantizer_conv_1 = seq_mse.sim.qc_quantize_op_dict['conv1.weight']
    weight_quantizer_conv_2 = seq_mse.sim.qc_quantize_op_dict['conv2.weight']
    weight_quantizer_conv_3 = seq_mse.sim.qc_quantize_op_dict['conv3.weight']

    assert weight_quantizer_conv_1.is_encoding_frozen()
    assert weight_quantizer_conv_2.is_encoding_frozen()
    assert weight_quantizer_conv_3.is_encoding_frozen()


def test_dependency_graph():
    model = get_model_with_split()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=4,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams()
    dataloader = unlabeled_data_loader(dummy_input_for_dependency_graph())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    seq_mse.dependency_graph = seq_mse.dependency_graph_utils.create_dependency_graph(dataloader, seq_params.num_batches)

    assert seq_mse.dependency_graph.node_by_name['/conv1/Conv'].indegree == 0
    assert seq_mse.dependency_graph.node_by_name['/conv1/Conv'].outdegree == 2
    assert seq_mse.dependency_graph.node_by_name['/conv1/Conv'].op_input_names == ['input']
    assert seq_mse.dependency_graph.node_by_name['/conv1/Conv'].op_output_names == ['/conv1/Conv_output_0']

    assert seq_mse.dependency_graph.node_by_name['/conv2/Conv'].indegree == 1
    assert seq_mse.dependency_graph.node_by_name['/conv2/Conv'].outdegree == 0
    assert seq_mse.dependency_graph.node_by_name['/conv2/Conv'].op_input_names == ['/conv1/Conv_output_0']
    assert seq_mse.dependency_graph.node_by_name['/conv2/Conv'].op_output_names == ['/conv2/Conv_output_0']

    assert seq_mse.dependency_graph.node_by_name['/conv3/Conv'].indegree == 1
    assert seq_mse.dependency_graph.node_by_name['/conv3/Conv'].outdegree == 0
    assert seq_mse.dependency_graph.node_by_name['/conv3/Conv'].op_input_names == ['/conv1/Conv_output_0']
    assert seq_mse.dependency_graph.node_by_name['/conv3/Conv'].op_output_names == ['/conv3/Conv_output_0']


def test_residual_model_dependency_graph():
    model = single_residual_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=4,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams()
    dataloader = unlabeled_data_loader(dummy_input_for_residual_model())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    seq_mse.dependency_graph = seq_mse.dependency_graph_utils.create_dependency_graph(dataloader, seq_params.num_batches)

    assert seq_mse.dependency_graph.node_by_name['/conv1/Conv'].indegree == 0
    assert seq_mse.dependency_graph.node_by_name['/conv1/Conv'].outdegree == 2
    assert seq_mse.dependency_graph.node_by_name['/conv1/Conv'].op_input_names == ['input']
    assert seq_mse.dependency_graph.node_by_name['/conv1/Conv'].op_output_names == ['/conv1/Conv_output_0']

    assert seq_mse.dependency_graph.node_by_name['/conv4/Conv'].indegree == 1
    assert seq_mse.dependency_graph.node_by_name['/conv4/Conv'].outdegree == 1
    assert seq_mse.dependency_graph.node_by_name['/conv4/Conv'].op_input_names == ['/maxpool/MaxPool_output_0']
    assert seq_mse.dependency_graph.node_by_name['/conv4/Conv'].op_output_names == ['/conv4/Conv_output_0']

    assert seq_mse.dependency_graph.node_by_name['/conv2/Conv'].indegree == 1
    assert seq_mse.dependency_graph.node_by_name['/conv2/Conv'].outdegree == 1
    assert seq_mse.dependency_graph.node_by_name['/conv2/Conv'].op_input_names == ['/maxpool/MaxPool_output_0']
    assert seq_mse.dependency_graph.node_by_name['/conv2/Conv'].op_output_names == ['/conv2/Conv_output_0']

    assert seq_mse.dependency_graph.node_by_name['/conv3/Conv'].indegree == 1
    assert seq_mse.dependency_graph.node_by_name['/conv3/Conv'].outdegree == 1
    assert seq_mse.dependency_graph.node_by_name['/conv3/Conv'].op_input_names == ['/relu2/Relu_output_0']
    assert seq_mse.dependency_graph.node_by_name['/conv3/Conv'].op_output_names == ['/conv3/Conv_output_0']

    assert seq_mse.dependency_graph.node_by_name['/Add'].indegree == 2
    assert seq_mse.dependency_graph.node_by_name['/Add'].outdegree == 0
    assert seq_mse.dependency_graph.node_by_name['/Add'].op_input_names == ['/conv3/Conv_output_0', '/ada/AveragePool_output_0']
    assert seq_mse.dependency_graph.node_by_name['/Add'].op_output_names == ['/Add_output_0']


@pytest.mark.skip  # TODO: check why its failing on CD but not locally
@pytest.mark.parametrize("inp_symmetry", ['asym', 'symfp', 'symqt'])
@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_apply_seq_mse_for_residual_model(inp_symmetry, param_bw, loss_fn, enable_pcq):
    model = single_residual_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=param_bw,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=enable_pcq))
    seq_params = SeqMseParams()
    seq_params.loss_fn = loss_fn
    seq_params.inp_symmetry = inp_symmetry
    dataloader = unlabeled_data_loader(dummy_input_for_residual_model())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    seq_mse.apply_seq_mse_algo()

    weight_quantizer_conv_1 = seq_mse.sim.qc_quantize_op_dict['onnx::Conv_45']
    weight_quantizer_conv_2 = seq_mse.sim.qc_quantize_op_dict['onnx::Conv_48']
    weight_quantizer_conv_3 = seq_mse.sim.qc_quantize_op_dict['conv3.weight']
    weight_quantizer_conv_4 = seq_mse.sim.qc_quantize_op_dict['conv4.weight']
    weight_quantizer_fc = seq_mse.sim.qc_quantize_op_dict['fc.weight']

    assert weight_quantizer_conv_1.is_encoding_frozen()
    assert weight_quantizer_conv_2.is_encoding_frozen()
    assert weight_quantizer_conv_3.is_encoding_frozen()
    assert weight_quantizer_conv_4.is_encoding_frozen()
    assert weight_quantizer_fc.is_encoding_frozen() == False


def test_model_with_multiple_inputs_dependency_graph_utils():

    model = get_model_with_multiple_inputs()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=4,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams()
    dataloader = unlabeled_data_loader(dummy_input_for_model_with_multiple_input())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)

    starting_ops_names = [op.name_op for op in seq_mse.dependency_graph_utils.starting_ops]

    assert starting_ops_names == ["Conv1"]
    assert seq_mse.dependency_graph_utils.indegree == {"Conv1": 0, "ADD_0": 1, "ADD_1": 1, "Conv2": 1}
    assert seq_mse.dependency_graph_utils.input_ops_name == ["Conv1", "ADD_0", "ADD_1"]

def test_model_with_multiple_outputs_value_info():

    model = get_model_with_multiple_outputs()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=4,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams()
    dataloader = unlabeled_data_loader(dummy_input_for_model_with_multiple_input())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)

    assert 'Conv1_Y' in seq_mse._sim_extractor.vimap