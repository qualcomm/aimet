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

import io
import os
import numpy as np
import pytest
import tempfile
import torch
from onnx import load_model

from aimet_onnx.utils import make_dummy_input
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.common.quantsim_config.utils import get_path_for_per_channel_config

try:
    from torch_utils import get_librispeech_data_loaders, train_librispeech
    from torchaudio import models
except (ImportError, OSError):
    pass
    # TODO (hitameht): For onnx-cpu variant, fix OSError: libtorch_hip.so: cannot open shared object file: No such file or directory

batch_size = 64
n_feature = 128
n_class = 29
CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def model_eval_onnx(session, val_loader, max_batches):
    """
    :param model: model to be evaluated
    :param val_loader: dataloader for validation data
    :return: CTC Loss on validation data
    """

    test_loss = 0
    for i, batch in enumerate(val_loader):
        spectrograms, labels, input_lengths, label_lengths = batch
        x = spectrograms.numpy()

        in_tensor = {"input": x}
        out = session.run(None, in_tensor)[0]

        out = torch.Tensor(out).transpose(0, 1)
        criterion = torch.nn.CTCLoss(blank=28)
        loss = criterion(out, labels, input_lengths, label_lengths)
        test_loss += loss.item() / len(val_loader)

        if i + 1 >= max_batches:
            break

    print(f"Test loss: {test_loss}")
    return test_loss


class TestQuantizeAcceptance:
    """Acceptance test for AIMET ONNX"""

    @pytest.mark.parametrize("config_file", [None, get_path_for_per_channel_config()])
    @pytest.mark.cuda
    @pytest.mark.skip(reason="Figure out a way to download datasets.")
    def test_quantized_accuracy(self, config_file):
        np.random.seed(0)
        torch.manual_seed(0)
        model = models.DeepSpeech(n_feature=n_feature, n_class=n_class)
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model.to(device)

        train_librispeech(model, 1, max_batches=30)

        train_loader, val_loader = get_librispeech_data_loaders(
            batch_size=batch_size, drop_last=False
        )

        buffer = io.BytesIO()
        torch.onnx.export(
            model,
            torch.rand(1, 1, 1, 128).cuda(),
            buffer,
            training=torch.onnx.TrainingMode.PRESERVE,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "time"},
                "output": {0: "batch_size", 1: "time"},
            },
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
                print(i, "/", iters)
                if i + 1 >= iters:
                    break

        sim.compute_encodings(onnx_callback, 1)

        onnx_qs_test_loss = model_eval_onnx(sim.session, val_loader, max_batches=1)

        assert onnx_qs_test_loss < 0.1
