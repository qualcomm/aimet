# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

import io
import os
import numpy as np
import pytest
import tempfile
import torch
from onnx import load_model
from torchvision import models

from aimet_onnx.utils import make_dummy_input
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.common.quantsim_config.utils import get_path_for_per_channel_config

try:
    from torch_utils import get_cifar10_data_loaders, train_cifar10
except (ImportError, OSError):
    pass
    # TODO (hitameht): For onnx-cpu variant, fix OSError: libtorch_hip.so: cannot open shared object file: No such file or directory

image_size = 32
batch_size = 64
num_workers = 4
CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def model_eval_onnx(session, val_loader):
    """
    :param model: model to be evaluated
    :param early_stopping_iterations: if None, data loader will iterate over entire validation data
    :return: top_1_accuracy on validation data
    """

    corr = 0
    total = 0
    for i, batch in enumerate(val_loader):
        x, y = batch[0].numpy(), batch[1].numpy()
        in_tensor = {"input": x}
        out = session.run(None, in_tensor)[0]
        corr += np.sum(np.argmax(out, axis=1) == y)
        total += x.shape[0]
    print(f"Accuracy: {corr / total}")
    return corr / total


class TestQuantizeAcceptance:
    """Acceptance test for AIMET ONNX"""

    @pytest.mark.skip("Disable test temporarily to unblock pipeline")
    @pytest.mark.parametrize("config_file", [None, get_path_for_per_channel_config()])
    @pytest.mark.cuda
    def test_quantized_accuracy(self, config_file):
        np.random.seed(0)
        torch.manual_seed(0)
        model = models.resnet18(pretrained=False, num_classes=10)
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model.to(device)

        train_cifar10(model, 2)
        train_loader, val_loader = get_cifar10_data_loaders(drop_last=False)

        buffer = io.BytesIO()
        torch.onnx.export(
            model,
            torch.rand(batch_size, 3, 32, 32).cuda(),
            buffer,
            training=torch.onnx.TrainingMode.PRESERVE,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=12,
            dynamo=False,
        )

        buffer.seek(0)
        onnx_model = load_model(buffer)
        dummy_input = make_dummy_input(onnx_model)
        sim = QuantizationSimModel(
            onnx_model,
            dummy_input,
            quant_scheme=QuantScheme.post_training_tf,
            default_param_bw=8,
            default_activation_bw=8,
            providers=CUDA_PROVIDERS,
            config_file=config_file,
        )

        def onnx_callback(session, iters):
            for i, batch in enumerate(train_loader):
                x = batch[0].detach().cpu().numpy()
                in_tensor = {"input": x}
                session.run(None, in_tensor)
                if i >= iters:
                    break

        sim.compute_encodings(onnx_callback, 10)
        onnx_qs_acc = model_eval_onnx(sim.session, val_loader)
        assert onnx_qs_acc > 0.5

    def test_dummy(self):
        # pytest has a 'feature' that returns an error code when all tests for a given suite are not selected
        # to be executed, So adding a dummy test to satisfy pytest
        pass
