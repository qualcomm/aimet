# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import os
from aimet_onnx.common.connected_graph.operation import Op
from aimet_onnx.graph_passes.pass_registry import register_pass
from aimet_onnx.graph_passes.graph_pass import SupergroupGraphPass
from aimet_onnx.graph_passes.utils import get_const_input_names, get_output_names
import onnx_ir
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.utils import ModelProto
from aimet_onnx.quantsim import QuantizationSimModel, QuantScheme

import numpy as np
import json
import pytest
import tempfile

from ..models.models_for_tests import build_dummy_model
from ..utils import tmp_dir, patch_fuse_supergroups


def _generate_quantsim_config(supergroup_pass_name: str, file_path: str) -> dict:
    """
    Writes QuantSim config with provided supergroup pass name to provided file path

    Args:
        supergroup_pass_name (str): supergroup pass name to set
        file_path (str): path to write json config file to
    """
    quantsim_config = {
        "defaults": {
            "ops": {"is_output_quantized": "True", "is_symmetric": "False"},
            "params": {"is_quantized": "False", "is_symmetric": "False"},
        },
        "params": {},
        "op_type": {},
        "supergroup_pass_list": [supergroup_pass_name],
        "supergroups": [
            {"op_list": ["Conv", "Relu"]},
            {"op_list": ["Relu", "MaxPool"]},
        ],
        "model_input": {"is_input_quantized": "True"},
        "model_output": {},
    }
    with open(file_path, "w") as f:
        json.dump(quantsim_config, f)


@register_pass("DummyTestGraphPass")
class DummyTestGraphPass(SupergroupGraphPass):
    def match_pattern(self, op: Op, _: ModelProto):
        self.disable_quantizers = get_const_input_names(
            op_list=[op]
        ) + get_output_names(op_list=[op])
        return [op]


def test_register_and_apply_graph_pass(tmp_dir):
    model = build_dummy_model()
    input_data = {"x": np.random.rand(1, 3, 32, 32).astype(np.float32)}

    config_file = str(os.path.join(tmp_dir, "quantsim_config.json"))
    _generate_quantsim_config("DummyTestGraphPass", config_file)
    with patch_fuse_supergroups(False):
        sim = QuantizationSimModel(
            model,
            input_data,
            quant_scheme=QuantScheme.post_training_tf,
            default_param_bw=8,
            default_activation_bw=8,
            config_file=config_file,
        )

    graph = ConnectedGraph(onnx_ir.from_proto(model))
    disable_quantizers = set(
        get_const_input_names(graph.ordered_ops) + get_output_names(graph.ordered_ops)
    )
    for name, quantizer in sim.qc_quantize_op_dict.items():
        # Ensure quantizers are disabled if they are in disable_quantizers set
        assert quantizer.enabled ^ (name in disable_quantizers)


def test_error_on_unregistered_graph_pass(tmp_dir):
    model = build_dummy_model()

    with pytest.raises(ValueError, match="Graph pass requested but not found:"):
        config_file = str(os.path.join(tmp_dir, "quantsim_config.json"))
        _generate_quantsim_config("UnsupportedGraphPass", config_file)
        input_data = {"x": np.random.rand(1, 3, 32, 32).astype(np.float32)}
        _ = QuantizationSimModel(
            model,
            input_data,
            quant_scheme=QuantScheme.post_training_tf,
            default_param_bw=8,
            default_activation_bw=8,
            config_file=config_file,
        )
