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
"""Test GPTVQ weight"""
import itertools
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Union, Tuple

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from aimet_common import quantsim
from aimet_torch.gptvq.defs import GPTVQSupportedModules
from aimet_torch.gptvq.gptvq_weight import GPTVQ, GPTVQParameters
from aimet_torch.utils import is_vector_encoding
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantization.affine import VectorEncoding
from aimet_torch.v2.quantsim import QuantizationSimModel
from .models import test_models

QUANTSIM_CONFIG = {
    "defaults": {
        "ops": {"is_output_quantized": "True"},
        "params": {"is_quantized": "True", "is_symmetric": "True"},
        "strict_symmetric": "False",
        "per_channel_quantization": "True",
    },
    "params": {"bias": {"is_quantized": "False"}},
    "op_type": {
        "Squeeze": {"is_output_quantized": "False"},
        "Pad": {"is_output_quantized": "False"},
        "Mean": {"is_output_quantized": "False"},
        # Enable per-channel quantization for Gemm to validate GPTVQ algorithm
        "Gemm": {"per_channel_quantization": "True"},
        "LayerNorm": {"per_channel_quantization": "False"},
        "Gather": {"is_output_quantized": "False"},
    },
    "supergroups": [
        {"op_list": ["Conv", "Relu"]},
        {"op_list": ["Conv", "Clip"]},
        {"op_list": ["ConvTranspose", "Relu"]},
        {"op_list": ["Add", "Relu"]},
        {"op_list": ["Gemm", "Relu"]},
    ],
    "model_input": {"is_input_quantized": "True"},
    "model_output": {},
}


@contextmanager
def swap_encoding_version(version='1.0.0'):
    old_version = quantsim.encoding_version
    quantsim.encoding_version = version

    yield

    quantsim.encoding_version = old_version


class RandomDataset(Dataset):
    def __init__(self, data_size = 32, input_dim: Union[int, Tuple] = 10):
        self.data_size = data_size
        self.input_dim = input_dim

        # generate random data and store it in lists
        input_dim = input_dim if isinstance(input_dim, tuple) else (input_dim,)
        self.data_x = [torch.rand(*input_dim) for _ in range(data_size)]
        self.data_y = [torch.rand(1) for _ in range(data_size)]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y


class TestGPTVQWeight:
    @pytest.mark.parametrize("vector_bw", [4, 8, 16])
    @pytest.mark.parametrize("rows_per_block", [32, 64])
    def test_quant_sim_initialization_in_gptvq(self, vector_bw, rows_per_block):
        model = test_models.ModelWithThreeLinears()

        data_loader = DataLoader(RandomDataset(data_size=2, input_dim=768), batch_size=1, shuffle=False)
        gptvq_parameters = GPTVQParameters(
            data_loader=data_loader,
            forward_fn=lambda m, d: m(d[0]),
            vector_bw=vector_bw, rows_per_block=rows_per_block
        )
        dummy_input = torch.randn(1, 768)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/quantsim_config.json"
            with open(config_path, "w") as f:
                json.dump(QUANTSIM_CONFIG, f)

            module_name_set = GPTVQ._get_candidate_module_name_set(model, module_names_to_exclude=None)
            quant_sim = GPTVQ._get_quantsim(
                model, dummy_input, gptvq_parameters, config_file_path=config_path, module_name_set=module_name_set
            )

        with GPTVQ._disable_quantizers_for_gptvq_optimization(quant_sim, module_name_set):
            for module in quant_sim.model.modules():
                if isinstance(module, BaseQuantizationMixin):
                    # Input/Output quantizers should be disabled
                    assert all((x is None for x in module.input_quantizers))
                    assert all((x is None for x in module.output_quantizers))

                    if isinstance(module.get_original_module(), GPTVQSupportedModules):
                        weight_shape = module.weight.shape
                        weight_quantizer = module.param_quantizers["weight"]
                        # Bitwidth, shape and block_size should be matched with GPTVQ parameters
                        assert weight_quantizer.bitwidth == gptvq_parameters.vector_bw
                        assert weight_quantizer.shape == (weight_shape[0] // gptvq_parameters.rows_per_block, 1)
                        assert weight_quantizer.block_size == (gptvq_parameters.rows_per_block, weight_shape[1])
                        assert not weight_quantizer.is_initialized(), "Weight quantizer should be initialized during GPTVQ optimization"

    def test_gptvq_weight_update(self):
        model = test_models.ModelWithThreeLinears()
        data_loader = DataLoader(RandomDataset(data_size=1, input_dim=768), batch_size=1, shuffle=False)
        gptvq_parameters = GPTVQParameters(data_loader, forward_fn=lambda m, d: m(d[0]), num_of_kmeans_iterations=1)
        dummy_input = torch.randn(1, 768)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/quantsim_config.json"
            with open(config_path, "w") as f:
                json.dump(QUANTSIM_CONFIG, f)

            rounded_model = GPTVQ.apply_gptvq(
                model,
                dummy_input,
                gptvq_parameters,
                param_encoding_path=temp_dir,
                config_file_path=config_path,
            )

        assert not torch.allclose(model.linear1.weight, rounded_model.linear1.weight)
        assert not torch.allclose(model.linear2.weight, rounded_model.linear2.weight)
        assert not torch.allclose(model.linear3.weight, rounded_model.linear3.weight)

    def test_exported_param_encodings_after_gptvq(self):
        model = test_models.ModelWithThreeLinears()
        data_loader = DataLoader(RandomDataset(data_size=1, input_dim=768), batch_size=1, shuffle=False)
        gptvq_parameters = GPTVQParameters(data_loader, forward_fn=lambda m, d: m(d[0]), num_of_kmeans_iterations=1)
        dummy_input = torch.randn(1, 768)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/quantsim_config.json"
            with open(config_path, "w") as f:
                json.dump(QUANTSIM_CONFIG, f)

            _ = GPTVQ.apply_gptvq(
                model,
                dummy_input,
                gptvq_parameters,
                param_encoding_path=temp_dir,
                config_file_path=config_path,
            )

            with open(f"{temp_dir}/gptvq.encodings") as f:
                encodings = json.load(f)
            param_encodings = encodings["param_encodings"]

        for name, module in model.named_modules():
            if isinstance(module, GPTVQSupportedModules):
                num_of_channels = module.weight.shape[0]
                weight_encodings = param_encodings[f"{name}.weight"]
                # The number of encodings should be same with the number of channels
                assert num_of_channels == len(weight_encodings)
                # Encodings in same block should have same encodings parameters
                # e.g., 0 to 31 channel (First block) should have same min/max encodings
                for i in range(0, num_of_channels, gptvq_parameters.rows_per_block):
                    assert len({x["min"] for x in weight_encodings[i : i + gptvq_parameters.rows_per_block]}) == 1
                    assert len({x["max"] for x in weight_encodings[i : i + gptvq_parameters.rows_per_block]}) == 1

    def test_gptvq_weight_update_with_block_level_modules(self):
        model = test_models.ModelWithThreeLinears()
        data_loader = DataLoader(RandomDataset(data_size=1, input_dim=768), batch_size=1, shuffle=False)
        gptvq_parameters = GPTVQParameters(data_loader, forward_fn=lambda m, d: m(d[0]), num_of_kmeans_iterations=1)
        dummy_input = torch.randn(1, 768)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/quantsim_config.json"
            with open(config_path, "w") as f:
                json.dump(QUANTSIM_CONFIG, f)

            leaf_level_rounded_model = GPTVQ.apply_gptvq(
                model,
                dummy_input,
                gptvq_parameters,
                param_encoding_path=temp_dir,
                config_file_path=config_path,
                block_level_module_names=[["linear1"], ["linear2"]]
            )
            block_level_rounded_model = GPTVQ.apply_gptvq(
                model,
                dummy_input,
                gptvq_parameters,
                param_encoding_path=temp_dir,
                config_file_path=config_path,
                block_level_module_names=[["linear1", "linear2"]]
            )

        # Updated weight of first module should be same both leaf level and block level
        assert torch.allclose(leaf_level_rounded_model.linear1.weight, block_level_rounded_model.linear1.weight)
        # After first module optimization, Hessian of next module is affected by previous module if leaf level optimization
        assert not torch.allclose(leaf_level_rounded_model.linear2.weight, block_level_rounded_model.linear2.weight)

    def test_gptvq_module_name_validation(self):
        model = test_models.ModelWithThreeLinears()
        with pytest.raises(ValueError):
            module_names_to_exclude = ["foo", "bar"]
            GPTVQ._validate_module_names(model, module_names_to_exclude, "module_names_to_exclude")

        with pytest.raises(ValueError):
            block_level_module_names = itertools.chain.from_iterable([["linear1"], ["linear2", "foo"]])
            GPTVQ._validate_module_names(model, block_level_module_names, "block_level_module_names")

        GPTVQ._validate_module_names(model, ["linear1", "linear2"], "module_names_to_exclude")
        GPTVQ._validate_module_names(model, itertools.chain.from_iterable([["linear1", "linear2"], ["linear3"]]), "block_level_module_names")

    def test_gptvq_block_level_module_names(self):
        model = test_models.ModelWithThreeLinears()
        dummy_input = torch.randn(1, 768)

        block_level_module_names = [["linear2"]]
        assert GPTVQ._get_block_level_module_names(
            model, dummy_input, block_level_module_names, {"linear1"}
        ) == [["linear2"], ["linear3"]]

        block_level_module_names = [["linear3", "linear2"]]
        assert GPTVQ._get_block_level_module_names(
            model, dummy_input, block_level_module_names, set()
        ) == [["linear1"], ["linear2", "linear3"]]

        block_level_module_names = [["linear3"], ["linear1"], ["linear2"]]
        assert GPTVQ._get_block_level_module_names(
            model, dummy_input, block_level_module_names, set()
        ) == [["linear1"], ["linear2"], ["linear3"]]

        block_level_module_names = [["linear3", "linear2"], ["linear1"]]
        assert GPTVQ._get_block_level_module_names(
            model, dummy_input, block_level_module_names, set()
        ) == [["linear1"], ["linear2", "linear3"]]

        block_level_module_names = [["linear2"], ["linear3", "linear1"]]
        assert GPTVQ._get_block_level_module_names(
            model, dummy_input, block_level_module_names, set()
        ) == [["linear1", "linear3"], ["linear2"]]

        block_level_module_names = [["linear3", "linear1", "linear2"]]
        assert GPTVQ._get_block_level_module_names(
            model, dummy_input, block_level_module_names, set()
        ) == [["linear1", "linear2", "linear3"]]

        block_level_module_names = None
        assert GPTVQ._get_block_level_module_names(
            model, dummy_input, block_level_module_names, set()
        ) == [["linear1"], ["linear2"], ["linear3"]]

    def test_gptvq_and_load_encodings(self):
        model = test_models.ModelWithThreeLinears()
        data_loader = DataLoader(RandomDataset(data_size=1, input_dim=768), batch_size=1, shuffle=False)
        gptvq_parameters = GPTVQParameters(data_loader, forward_fn=lambda m, d: m(d[0]), num_of_kmeans_iterations=1)
        dummy_input = torch.randn(1, 768)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/quantsim_config.json"
            with open(config_path, "w") as f:
                json.dump(QUANTSIM_CONFIG, f)

            rounded_model = GPTVQ.apply_gptvq(
                model,
                dummy_input,
                gptvq_parameters,
                param_encoding_path=temp_dir,
                config_file_path=config_path,
                module_names_to_exclude=["linear2"]
            )

            sim = QuantizationSimModel(
                rounded_model,
                dummy_input=dummy_input,
                default_param_bw=gptvq_parameters.vector_bw,
                config_file=config_path
            )
            sim.load_encodings(f"{temp_dir}/gptvq.encodings", allow_overwrite=False)
            assert hasattr(sim.model.linear1.weight, "encoding")
            assert isinstance(sim.model.linear1.weight.encoding, VectorEncoding)
            assert sim.model.linear1.param_quantizers["weight"] is None

            # linear2 was excluded in GPTVQ
            assert not hasattr(sim.model.linear2.weight, "encoding")
            assert sim.model.linear2.param_quantizers["weight"] is not None

            assert hasattr(sim.model.linear3.weight, "encoding")
            assert isinstance(sim.model.linear3.weight.encoding, VectorEncoding)
            assert sim.model.linear3.param_quantizers["weight"] is None

    def test_gptvq_export(self):
        model = test_models.ModelWithThreeLinears()
        data_loader = DataLoader(RandomDataset(data_size=1, input_dim=768), batch_size=1, shuffle=False)
        gptvq_parameters = GPTVQParameters(data_loader, forward_fn=lambda m, d: m(d[0]), num_of_kmeans_iterations=1)
        dummy_input = torch.randn(1, 768)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/quantsim_config.json"
            with open(config_path, "w") as f:
                json.dump(QUANTSIM_CONFIG, f)

            rounded_model = GPTVQ.apply_gptvq(
                model,
                dummy_input,
                gptvq_parameters,
                param_encoding_path=temp_dir,
                config_file_path=config_path,
                module_names_to_exclude=["linear2"]
            )

            sim = QuantizationSimModel(
                rounded_model,
                dummy_input=dummy_input,
                default_param_bw=gptvq_parameters.vector_bw,
                config_file=config_path
            )
            sim.load_encodings(f"{temp_dir}/gptvq.encodings", allow_overwrite=False)
            sim.compute_encodings(lambda m, _: m(dummy_input), None)
            with swap_encoding_version():
                sim.export(temp_dir, "vq_with_activation", dummy_input)

            with open(os.path.join(temp_dir, "vq_with_activation.encodings")) as f:
                encodings = json.load(f)

        assert len(encodings["activation_encodings"]) == 1 + 4  # One input, four output encodings

        param_encodings = encodings["param_encodings"]
        linear1_encoding, linear2_encoding, linear3_encoding = param_encodings
        assert linear1_encoding["enc_type"] == "VECTOR"
        num_scales = len(linear1_encoding["scale"])
        rows_per_block = linear1_encoding["rows_per_block"]
        assert num_scales == sim.model.linear1.weight.shape[0]
        for i in range(0, num_scales, rows_per_block):
            # per-channel scales should be same within a block
            assert len(set(linear1_encoding["scale"][i:i + rows_per_block])) == 1

        assert linear2_encoding["enc_type"] == "PER_CHANNEL"
        assert len(linear2_encoding["scale"]) == sim.model.linear2.weight.shape[0]

        assert linear3_encoding["enc_type"] == "VECTOR"
        num_scales = len(linear3_encoding["scale"])
        rows_per_block = linear3_encoding["rows_per_block"]
        assert num_scales == sim.model.linear3.weight.shape[0]
        for i in range(0, num_scales, rows_per_block):
            # per-channel scales should be same within a block
            assert len(set(linear3_encoding["scale"][i:i + rows_per_block])) == 1

    def test_gptvq_conv_model(self):
        model = test_models.BasicConv2d(kernel_size=3)
        dataset = RandomDataset(data_size=4, input_dim=(64, 8, 8))
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        gptvq_parameters = GPTVQParameters(data_loader, forward_fn=lambda m, d: m(d[0]), num_of_kmeans_iterations=1)
        dummy_input = torch.randn(1, 64, 8, 8)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/quantsim_config.json"
            with open(config_path, "w") as f:
                json.dump(QUANTSIM_CONFIG, f)

            rounded_model = GPTVQ.apply_gptvq(
                model,
                dummy_input,
                gptvq_parameters,
                param_encoding_path=temp_dir,
                config_file_path=config_path,
            )

            assert not torch.allclose(model.conv.weight, rounded_model.conv.weight)
            sim = QuantizationSimModel(
                rounded_model,
                dummy_input=dummy_input,
                default_param_bw=gptvq_parameters.vector_bw,
                config_file=config_path
            )
            sim.load_encodings(f"{temp_dir}/gptvq.encodings", allow_overwrite=False)
            sim.compute_encodings(lambda m, _: m(dummy_input), None)
            with swap_encoding_version():
                sim.export(temp_dir, "vq_conv", dummy_input)

            with open(os.path.join(temp_dir, "vq_conv.encodings")) as f:
                encodings = json.load(f)

        assert len(encodings["activation_encodings"]) == 1 + 3  # One input, three (Conv, BN, Relu) output encodings
        param_encodings = encodings["param_encodings"]
        conv_encoding, *_ = param_encodings
        assert is_vector_encoding([conv_encoding])
        assert conv_encoding["enc_type"] == "VECTOR"
        assert conv_encoding["bw"] == gptvq_parameters.vector_bw
        assert conv_encoding["rows_per_block"] == gptvq_parameters.rows_per_block
        assert conv_encoding["cols_per_block"] == gptvq_parameters.cols_per_block
        assert conv_encoding["index_bw"] == gptvq_parameters.index_bw
        assert conv_encoding["vector_dim"] == gptvq_parameters.vector_dim
        assert conv_encoding["vector_stride"] == gptvq_parameters.vector_stride

        num_scales = len(conv_encoding["scale"])
        rows_per_block = conv_encoding["rows_per_block"]
        assert num_scales == sim.model.conv.weight.shape[0]
        for i in range(0, num_scales, rows_per_block):
            # per-channel scales should be same within a block
            assert len(set(conv_encoding["scale"][i:i + rows_per_block])) == 1
