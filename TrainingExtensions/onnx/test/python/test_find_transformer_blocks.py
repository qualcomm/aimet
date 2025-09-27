# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
from aimet_onnx.experimental.adascale.find_blocks import (
    get_decoder_blocks_end_points,
    get_conv_linear_layers_decoder_block,
)


def test_get_decoder_blocks(monkeypatch):
    path = os.path.abspath(os.path.join("../../../../GenAITests"))
    monkeypatch.syspath_prepend(path)
    from GenAITests.onnx.models.qwen import Qwen_25_ONNX

    sim, _ = Qwen_25_ONNX.instantiate_quantsim(
        "Qwen/Qwen2-0.5B", 32, 16, small_model=True
    )
    end_points = get_decoder_blocks_end_points(sim)
    end_points_names = [(op1.name, op2.name) for op1, op2 in end_points]
    assert end_points_names == [
        (
            "/model/model/layers.0/input_layernorm/Pow",
            "/model/model/layers.1/input_layernorm/Pow",
        ),
        ("/model/model/layers.1/input_layernorm/Pow", "/model/model/norm/Pow"),
    ]
    conv_linear_blocks = get_conv_linear_layers_decoder_block(sim, end_points)
    conv_linear_blocks_names = []
    for ops in conv_linear_blocks:
        res = []
        for op in ops:
            res.append(op.name)
        conv_linear_blocks_names.append(res)

    assert conv_linear_blocks_names == [
        [
            "/model/model/layers.0/self_attn/v_proj/MatMul",
            "/model/model/layers.0/self_attn/k_proj/MatMul",
            "/model/model/layers.0/self_attn/q_proj/MatMul",
            "/model/model/layers.0/self_attn/MatMul",
            "/model/model/layers.0/self_attn/MatMul_1",
            "/model/model/layers.0/self_attn/o_proj/MatMul",
            "/model/model/layers.0/mlp/up_proj/MatMul",
            "/model/model/layers.0/mlp/gate_proj/MatMul",
            "/model/model/layers.0/mlp/down_proj/MatMul",
        ],
        [
            "/model/model/layers.1/self_attn/v_proj/MatMul",
            "/model/model/layers.1/self_attn/k_proj/MatMul",
            "/model/model/layers.1/self_attn/q_proj/MatMul",
            "/model/model/layers.1/self_attn/MatMul",
            "/model/model/layers.1/self_attn/MatMul_1",
            "/model/model/layers.1/self_attn/o_proj/MatMul",
            "/model/model/layers.1/mlp/up_proj/MatMul",
            "/model/model/layers.1/mlp/gate_proj/MatMul",
            "/model/model/layers.1/mlp/down_proj/MatMul",
        ],
    ]
