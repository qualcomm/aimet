# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import io
import os
import json
import numpy as np
import pytest
import tempfile
import torch
from onnx import load_model
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from torchvision import models

from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx import apply_adaround
from aimet_onnx.adaround.utils import AdaroundSupportedModules
import copy

image_size = 32
batch_size = 64
num_workers = 4

EXECUTION_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class TestAdaroundAcceptance:
    """Acceptance test for AIMET ONNX"""

    @pytest.mark.cuda
    def test_adaround(self):
        np.random.seed(0)
        torch.manual_seed(0)
        model = get_model()
        dummy_input = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}

        sim = QuantizationSimModel(
            copy.deepcopy(model),
            dummy_input,
            quant_scheme=QuantScheme.post_training_tf,
            default_param_bw=8,
            default_activation_bw=8,
            providers=EXECUTION_PROVIDERS,
        )
        sim.compute_encodings([dummy_input])
        out_before_ada = sim.session.run(None, dummy_input)
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)
        out_after_ada = sim.session.run(None, dummy_input)
        assert not np.array_equal(out_before_ada[0], out_after_ada[0])

        fp_model = sim.remove_quantizers(sim.model.model)
        for node in fp_model.graph.node:
            if node.op_type in AdaroundSupportedModules:
                assert sim.qc_quantize_op_dict[node.input[1]]._is_encoding_frozen


def get_model():
    model = models.resnet18(pretrained=False, num_classes=10)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.to(device)

    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        torch.rand(batch_size, 3, 32, 32).cuda(),
        buffer,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    buffer.seek(0)
    onnx_model = ONNXModel(load_model(buffer))
    return onnx_model
