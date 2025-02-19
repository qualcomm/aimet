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
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from aimet_onnx.adaround.adaround_weight import Adaround, AdaroundParameters

image_size = 32
batch_size = 64
num_workers = 4


def model_eval_onnx(session, val_loader):
    """
    :param model: model to be evaluated
    :param early_stopping_iterations: if None, data loader will iterate over entire validation data
    :return: top_1_accuracy on validation data
    """

    corr = 0
    total = 0
    for (i, batch) in enumerate(val_loader):
        x, y = batch[0].numpy(), batch[1].numpy()
        in_tensor = {'input': x}
        out = session.run(None, in_tensor)[0]
        corr += np.sum(np.argmax(out, axis=1) == y)
        total += x.shape[0]
    print(f'Accuracy: {corr / total}')
    return corr / total


class TestAdaroundAcceptance:
    """ Acceptance test for AIMET ONNX """
    @pytest.mark.cuda
    def test_adaround(self):
        np.random.seed(0)
        torch.manual_seed(0)
        model = get_model()
        data_loader = dataloader()
        dummy_input = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
        sess = build_session(model)
        out_before_ada = sess.run(None, dummy_input)
        def callback(session, args):
            in_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5, forward_fn=callback,
                                    forward_pass_callback_args=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            ada_rounded_model = Adaround.apply_adaround(model, params, tmpdir, 'dummy')
            sess = build_session(ada_rounded_model)
            out_after_ada = sess.run(None, dummy_input)
            assert not np.array_equal(out_before_ada[0], out_after_ada[0])

            with open(os.path.join(tmpdir, 'dummy.encodings')) as json_file:
                encoding_data = json.load(json_file)

            sim = QuantizationSimModel(ada_rounded_model, dummy_input, quant_scheme=QuantScheme.post_training_tf, default_param_bw=8,
                                       default_activation_bw=8, use_cuda=True)
            sim.set_and_freeze_param_encodings(os.path.join(tmpdir, 'dummy.encodings'))
            sim.compute_encodings(callback, None)

            param_encodings = {encoding['name']: encoding for encoding in encoding_data}
            assert sim.qc_quantize_op_dict['fc.weight'].encodings[0].delta == param_encodings['fc.weight']['scale'][0]

def get_model():
    model = models.resnet18(pretrained=False, num_classes=10)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model.to(device)

    torch.onnx.export(model, torch.rand(batch_size, 3, 32, 32).cuda(), './resnet18.onnx',
                      training=torch.onnx.TrainingMode.EVAL,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={
                          'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'},
                      }
                      )

    onnx_model = ONNXModel(load_model('./resnet18.onnx'))
    return onnx_model

def dataloader():
    class DataLoader:
        """
        Example of a Dataloader which can be used for running AMPv2
        """
        def __init__(self, batch_size: int):
            """
            :param batch_size: batch size for data loader
            """
            self.batch_size = batch_size

        def __iter__(self):
            """Iterates over dataset"""
            dummy_input = np.random.rand(1, 3, 32, 32).astype(np.float32)
            yield dummy_input

        def __len__(self):
            return 4

    dummy_dataloader = DataLoader(batch_size=2)
    return dummy_dataloader


def build_session(model):
    """
    Build and return onnxruntime inference session
    :param providers: providers to execute onnxruntime
    """
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    session = InferenceSession(
        path_or_bytes=model.model.SerializeToString(),
        sess_options=sess_options,
        providers=['CPUExecutionProvider'],
    )
    return session
