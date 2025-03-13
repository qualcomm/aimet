# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
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
from aimet_common.connected_graph.operation import Op
from aimet_onnx.graph_passes.pass_registry import register_pass
from aimet_onnx.graph_passes.graph_pass import SupergroupGraphPass
from aimet_onnx.graph_passes.utils import get_const_input_names, get_output_names
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.utils import ModelProto
from aimet_onnx.quantsim import QuantizationSimModel, QuantScheme

import numpy as np
import json
import pytest
import tempfile

from ..models.models_for_tests import build_dummy_model


def _generate_quantsim_config(supergroup_pass_name: str, file_path: str) -> dict:
    """
    Writes QuantSim config with provided supergroup pass name to provided file path

    Args:
        supergroup_pass_name (str): supergroup pass name to set
        file_path (str): path to write json config file to
    """
    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "False"
            },
            "params": {
                "is_quantized": "False",
                "is_symmetric": "False"
            }
        },
        "params": {},
        "op_type": {},
        "supergroup_pass_list": [supergroup_pass_name],
        "supergroups": [
            {
                "op_list": ["Conv", "Relu"]
            },
            {
                "op_list": ["Relu", "MaxPool"]
            },
        ],
        "model_input": {
            "is_input_quantized": "True"
        },
        "model_output": {}
    }
    with open(file_path, 'w') as f:
        json.dump(quantsim_config, f)

@register_pass("DummyTestGraphPass")
class DummyTestGraphPass(SupergroupGraphPass):
    def match_pattern(self, op: Op, _: ModelProto):
        self.disable_quantizers = get_const_input_names(op_list=[op]) + get_output_names(op_list=[op])
        return True

def test_register_and_apply_graph_pass():
    model = build_dummy_model()
    input_data = { "x" : np.random.rand(1, 3, 32, 32).astype(np.float32) }

    with tempfile.NamedTemporaryFile(prefix="quantsim_config", suffix=".json") as config_file:
        _generate_quantsim_config("DummyTestGraphPass", config_file.name)
        sim = QuantizationSimModel(model, input_data, quant_scheme=QuantScheme.post_training_tf, default_param_bw=8,
            default_activation_bw=8, config_file=config_file.name)

        graph = ConnectedGraph(model)
        disable_quantizers = set(get_const_input_names(graph.ordered_ops) + get_output_names(graph.ordered_ops))
        for name, quantizer in sim.qc_quantize_op_dict.items():
            # Ensure quantizers are disabled if they are in disable_quantizers set
            assert quantizer.enabled ^ (name in disable_quantizers)

def test_error_on_unregistered_graph_pass():
    model = build_dummy_model()

    with pytest.raises(
        ValueError, match="Graph pass requested but not found:"
    ):
        with tempfile.NamedTemporaryFile(prefix="quantsim_config", suffix=".json") as config_file:
            _generate_quantsim_config("UnsupportedGraphPass", config_file.name)
            input_data = { "x" : np.random.rand(1, 3, 32, 32).astype(np.float32) }
            _ = QuantizationSimModel(model, input_data, quant_scheme=QuantScheme.post_training_tf, default_param_bw=8,
                                default_activation_bw=8, config_file=config_file.name)
