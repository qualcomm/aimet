# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

import os
import json
import numpy as np
import pytest
import tempfile
import torch
from onnx import load_model
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from torchvision import models

from aimet_common.defs import QuantScheme
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

        sim.remove_quantizers(sim.model.model)
        for node in sim.model.nodes():
            if node.op_type in AdaroundSupportedModules:
                assert sim.qc_quantize_op_dict[node.input[1]]._is_encoding_frozen


def get_model():
    model = models.resnet18(pretrained=False, num_classes=10)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.to(device)

    torch.onnx.export(
        model,
        torch.rand(batch_size, 3, 32, 32).cuda(),
        "./resnet18.onnx",
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    onnx_model = ONNXModel(load_model("./resnet18.onnx"))
    return onnx_model
