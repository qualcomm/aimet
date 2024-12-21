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
import os
import onnxruntime as ort
import pytest
import numpy as np
import torch
from torch import nn
import onnx
import tempfile
import aimet_torch.v2 as aimet
import aimet_torch.v2.quantization as Q
from aimet_torch.v2.quantsim import QuantizationSimModel
from torchvision.models import resnet18
from aimet_torch.v2.experimental.onnx._export import export as _export



@pytest.fixture(autouse=True, params=range(1))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)


@pytest.mark.parametrize("qtzr_cls", [Q.affine.Quantize, Q.affine.QuantizeDequantize])
@pytest.mark.parametrize("input_shape, scale_shape, block_size", [
                         ([],          [],          None      ), # per-tensor
                         ((100, 100),  (1,),        None      ), # per-tensor
                         ((100, 100),  [],          None      ), # per-tensor
                         ((100, 100),  (100, 1),    None      ), # per-channel
                         ((100, 100),  (100, 1),    (1, 100)  ), # per-channel
                         ((100, 100),  (100, 50),   (1, 2)    ), # blockwise
                         ((100, 100),  (50, 100),   (2, 1)    ), # blockwise
                         ((100, 100),  (50, 50),    (2, 2)    ), # blockwise
                         ((100, 100),  (50, 50),    (-1, -1)  ), # blockwise
])
@pytest.mark.parametrize("symmetric", [True, False])
def test_quantize_torch_ort_equal(qtzr_cls, input_shape, scale_shape, block_size, symmetric):
    """
    When: Export a quantizer with torch.onnx.export
    """
    x = torch.randn(input_shape)
    qtzr = qtzr_cls(scale_shape, 8, symmetric, block_size=block_size)
    with qtzr.compute_encodings():
        _ = qtzr(x)

    with tempfile.TemporaryDirectory() as dirname:
        full_path = os.path.join(dirname, "qtzr.onnx")

        with open(full_path, "wb") as f:
            torch.onnx.export(qtzr, x, f, input_names=['input'], output_names=['output'])

        with torch.no_grad():
            y = qtzr(x)

        """
        Then: The saved onnx model should pass onnx model checker
        """
        model = onnx.load_model(full_path)
        onnx.checker.check_model(model)

        """
        Then: The saved onnx model should contain exactly one graph node in "aimet" domain
              with proper name and attributes
        """
        nodes = [node for node in model.graph.node if node.domain == 'aimet']
        assert len(nodes) == 1
        node, = nodes

        assert node.name == '/quantize' if qtzr_cls is Q.affine.Quantize else '/quantize_dequantize'
        assert node.attribute[0].name == 'block_size'
        assert node.attribute[0].ints == ([1] if block_size is None else list(np.array(input_shape) // np.array(scale_shape)))
        assert node.attribute[1].name == 'qmax'
        assert node.attribute[1].i == (127 if symmetric else 255)
        assert node.attribute[2].name == 'qmin'
        assert node.attribute[2].i == (-128 if symmetric else 0)

        """
        Then: The saved onnx model should produce the same output with the original quantizer
              given the same input
        """
        sess = ort.InferenceSession(full_path, providers=['CPUExecutionProvider'])
        out, = sess.run(None, {'input': x.numpy()})
        assert torch.equal(torch.from_numpy(out), y)


@pytest.mark.parametrize("input_shape, scale_shape, block_size", [
                         ([],          [],          None      ), # per-tensor
                         ((100, 100),  (1,),        None      ), # per-tensor
                         ((100, 100),  [],          None      ), # per-tensor
                         ((100, 100),  (100, 1),    None      ), # per-channel
                         ((100, 100),  (100, 1),    (1, 100)  ), # per-channel
                         ((100, 100),  (100, 50),   (1, 2)    ), # blockwise
                         ((100, 100),  (50, 100),   (2, 1)    ), # blockwise
                         ((100, 100),  (50, 50),    (2, 2)    ), # blockwise
                         ((100, 100),  (50, 50),    (-1, -1)  ), # blockwise
])
@pytest.mark.parametrize("symmetric", [True, False])
def test_dequantize_torch_ort_equal(input_shape, scale_shape, block_size, symmetric):
    """
    When: Export dequantize with torch.onnx.export
    """

    class Dequantize(torch.nn.Module):
        def forward(self, x: Q.QuantizedTensor):
            return x.dequantize()

    x = torch.randn(input_shape)
    qtzr = Q.affine.Quantize(scale_shape, 8, symmetric, block_size=block_size)
    with qtzr.compute_encodings():
        x = qtzr(x)

    with tempfile.TemporaryDirectory() as dirname:
        full_path = os.path.join(dirname, "qtzr.onnx")

        with open(full_path, "wb") as f:
            torch.onnx.export(Dequantize(), x, f, input_names=['input'], output_names=['output'])

        with torch.no_grad():
            y = x.dequantize()

        """
        Then: The saved onnx model should pass onnx model checker
        """
        model = onnx.load_model(full_path)
        onnx.checker.check_model(model)

        """
        Then: The saved onnx model should contain exactly one graph node in "aimet" domain
              with proper name and attributes
        """
        nodes = [node for node in model.graph.node if node.domain == 'aimet']
        assert len(nodes) == 1
        node, = nodes

        assert node.name == '/dequantize'
        assert node.attribute[0].name == 'block_size'
        assert node.attribute[0].ints == ([1] if block_size is None else list(np.array(input_shape) // np.array(scale_shape)))

        """
        Then: The saved onnx model should produce the same output with the original quantizer
              given the same input
        """
        sess = ort.InferenceSession(full_path, providers=['CPUExecutionProvider'])
        out, = sess.run(None, {'input': x.numpy()})
        assert torch.equal(torch.from_numpy(out), y)



@torch.no_grad()
def test_resnet18():
    """
    When: Export quantized resnet18 and run it on onnx runtime
    Then: The onnx model should produce output close enough to the original pytorch model
    """
    x = torch.randn(1, 3, 224, 224)
    model = resnet18(pretrained=False).eval()
    model = QuantizationSimModel(model, x).model

    with aimet.nn.compute_encodings(model):
        model(x)

    y = model(x)

    with tempfile.TemporaryDirectory() as dirname:
        full_path = os.path.join(dirname, "resnet18.onnx")

        with open(full_path, "wb") as f:
            torch.onnx.export(model, x, f, input_names=['input'], output_names=['output'])

        onnx_model = onnx.load_model(full_path)
        onnx.checker.check_model(onnx_model)

        sess = ort.InferenceSession(full_path, providers=['CPUExecutionProvider'])
        out, = sess.run(None, {'input': x.numpy()})

        # Allow off-by-3 error
        atol = 3 * model.fc.output_quantizers[0].get_scale().item()
        assert torch.allclose(torch.from_numpy(out), y, atol=atol)


def test_export(tmpdir):
    model = nn.Sequential(
        nn.Conv2d(3, 3, 3),
        nn.ReLU(),
        nn.Softmax(),
    )
    args = (torch.randn(1, 3, 224, 224),)
    sim = QuantizationSimModel(model, args)
    sim.compute_encodings(lambda model: model(*args))

    with open(os.path.join(tmpdir, 'out.onnx'), 'wb') as f:
        _export(sim.model, args, f)
