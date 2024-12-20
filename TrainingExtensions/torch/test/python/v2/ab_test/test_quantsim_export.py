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

import pytest
import tempfile
import torch.nn
import copy
import os
import json
import onnx
from packaging import version

from torchvision.models import resnet18

import aimet_torch.v2.nn as aimet_nn
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.quantization.affine import Quantize, QuantizeDequantize
from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
from aimet_torch._base.nn.modules.custom import Add
from aimet_torch import onnx_utils
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v1.quantsim import OnnxExportApiArgs
from aimet_torch.v1.qc_quantize_op import QcQuantizeWrapper
from aimet_common.defs import QuantScheme
from aimet_torch.utils import get_layer_by_name

from ..models_.models_to_test import (
    SimpleConditional,
    ModelWithTwoInputs,
    ModelWith5Output,
    SoftMaxAvgPoolModel,
    BasicConv2d,
)


class DummyModel(torch.nn.Module):

    def __init__(self, in_channels):
        super(DummyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(10, 10, 3, padding=1)
        self.add = Add()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x_resid = self.relu(x)
        x = self.conv2(x_resid)
        x = self.add(x, x_resid)
        return self.softmax(x)

export_args = {'opset_version': None, 'input_names': None, 'output_names': None}

class TestQuantsimOnnxExport:

    def test_onnx_export(self):
        export_args = OnnxExportApiArgs(opset_version=10, input_names=["input"], output_names=["output"])
        input_shape = (1, 10, 32, 32)
        fname = "test_model"
        dummy_input = torch.randn(input_shape)
        model = DummyModel(in_channels=input_shape[1])
        sim_model = copy.deepcopy(model)
        for name, module in sim_model.named_children():
            quantized_module = QuantizationMixin.from_module(module)

            if name == "conv1":
                input_quantizer = QuantizeDequantize((),
                                                     bitwidth=8,
                                                     symmetric=False,
                                                     encoding_analyzer=MinMaxEncodingAnalyzer(()))
                quantized_module.input_quantizers[0] = input_quantizer
            else:
                output_quantizer = QuantizeDequantize((),
                                                      bitwidth=8,
                                                      symmetric=False,
                                                      encoding_analyzer=MinMaxEncodingAnalyzer(()))
                quantized_module.output_quantizers[0] = output_quantizer

            if hasattr(module, 'weight'):
                weight_quantizer = QuantizeDequantize((),
                                                      bitwidth=4,
                                                      symmetric=True,
                                                      encoding_analyzer=MinMaxEncodingAnalyzer(()))
                quantized_module.param_quantizers['weight'] = weight_quantizer

            setattr(sim_model, name, quantized_module)

        with aimet_nn.compute_encodings(sim_model):
            _ = sim_model(torch.randn(input_shape))


        with tempfile.TemporaryDirectory() as path:
            QuantizationSimModel.export_onnx_model_and_encodings(path, fname, model, sim_model, dummy_input=dummy_input,
                                                                 onnx_export_args=export_args,
                                                                 propagate_encodings=False, module_marker_map={},
                                                                 is_conditional=False, excluded_layer_names=None,
                                                                 quantizer_args=None)

            file_path = os.path.join(path, fname + '.encodings')

            assert os.path.exists(file_path)

            with open(file_path) as f:
                encoding_dict = json.load(f)

        # Format is "/layer_name/OnnxType_{output/input}_{idx}"
        expected_act_keys = {"input", "/relu/Relu_output_0", "/conv2/Conv_output_0", "/add/Add_output_0", "output"}
        expected_param_keys = {"conv1.weight", "conv2.weight"}

        assert set(encoding_dict["activation_encodings"].keys()) == expected_act_keys
        assert set(encoding_dict["param_encodings"].keys()) == expected_param_keys

    # From: https://github.com/quic/aimet/blob/ce3dafe75d81893cdb8b45ba8abf53a672c28187/TrainingExtensions/torch/test/python/test_quantizer.py#L2731
    def test_export_to_onnx_direct(self):
        model = ModelWithTwoInputs()
        sim_model = copy.deepcopy(model)
        dummy_input = (torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28))
        for name, module in sim_model.named_children():
            quantized_module = QuantizationMixin.from_module(module)

            if name == "conv1_a":
                input_quantizer = QuantizeDequantize((),
                                                     bitwidth=8,
                                                     symmetric=False,
                                                     encoding_analyzer=MinMaxEncodingAnalyzer(()))
                quantized_module.input_quantizers[0] = input_quantizer

            output_quantizer = QuantizeDequantize((),
                                                  bitwidth=8,
                                                  symmetric=False,
                                                  encoding_analyzer=MinMaxEncodingAnalyzer(()))
            quantized_module.output_quantizers[0] = output_quantizer

            if hasattr(module, 'weight'):
                weight_quantizer = QuantizeDequantize((),
                                                      bitwidth=4,
                                                      symmetric=True,
                                                      encoding_analyzer=MinMaxEncodingAnalyzer(()))
                quantized_module.param_quantizers['weight'] = weight_quantizer

            setattr(sim_model, name, quantized_module)

        with aimet_nn.compute_encodings(sim_model):
            _ = sim_model(*dummy_input)

        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_utils.EXPORT_TO_ONNX_DIRECT = True
            QuantizationSimModel.export_onnx_model_and_encodings(temp_dir, "direct_onnx_export", model,
                                                                 sim_model, dummy_input, export_args,
                                                                 propagate_encodings=False)
            onnx_utils.EXPORT_TO_ONNX_DIRECT = False
            QuantizationSimModel.export_onnx_model_and_encodings(temp_dir, "onnxsaver_export", model,
                                                                 sim_model, dummy_input, export_args,
                                                                 propagate_encodings=False)

            with open(os.path.join(temp_dir, 'direct_onnx_export.encodings')) as direct_onnx_json:
                direct_onnx_encodings = json.load(direct_onnx_json)
            with open(os.path.join(temp_dir, 'onnxsaver_export.encodings')) as onnxsaver_json:
                onnxsaver_encodings = json.load(onnxsaver_json)

            assert len(direct_onnx_encodings['activation_encodings']) == \
                   len(onnxsaver_encodings['activation_encodings'])
            assert len(direct_onnx_encodings['param_encodings']) == len(onnxsaver_encodings['param_encodings'])
            direct_onnx_act_names = direct_onnx_encodings['activation_encodings'].keys()
            onnxsaver_act_names = onnxsaver_encodings['activation_encodings'].keys()
            assert direct_onnx_act_names != onnxsaver_act_names


    def test_encodings_propagation(self):
        """
        Test encodings are propagated correctly when more than
        one onnx node maps to the same torch module
        """
        export_args = OnnxExportApiArgs(opset_version=10, input_names=["input"], output_names=["output"])
        pixel_shuffle = torch.nn.PixelShuffle(2)
        model = torch.nn.Sequential(pixel_shuffle)

        quantized_pixel_shuffle = QuantizationMixin.from_module(pixel_shuffle)
        quantized_pixel_shuffle.input_quantizers[0] = QuantizeDequantize((),
                                                                         bitwidth=8,
                                                                         symmetric=False,
                                                                         encoding_analyzer=MinMaxEncodingAnalyzer(()))
        quantized_pixel_shuffle.output_quantizers[0] = QuantizeDequantize((),
                                                                          bitwidth=8,
                                                                          symmetric=False,
                                                                          encoding_analyzer=MinMaxEncodingAnalyzer(()))
        sim_model = torch.nn.Sequential(quantized_pixel_shuffle)
        dummy_input = torch.randn(1, 4, 8, 8)

        with aimet_nn.compute_encodings(sim_model):
            _ = sim_model(dummy_input)

        # Save encodings
        with tempfile.TemporaryDirectory() as path:
            fname_no_prop = "encodings_propagation_false"
            fname_prop = "encodings_propagation_true"
            QuantizationSimModel.export_onnx_model_and_encodings(path, fname_no_prop, model, sim_model,
                                                                 dummy_input=dummy_input,
                                                                 onnx_export_args=export_args,
                                                                 propagate_encodings=False)
            QuantizationSimModel.export_onnx_model_and_encodings(path, fname_prop, model, sim_model,
                                                                 dummy_input=dummy_input,
                                                                 onnx_export_args=export_args,
                                                                 propagate_encodings=True)
            with open(os.path.join(path, fname_no_prop + ".encodings")) as f:
                encoding_dict_no_prop = json.load(f)["activation_encodings"]
            with open(os.path.join(path, fname_prop + ".encodings")) as f:
                encoding_dict_prop = json.load(f)["activation_encodings"]

        assert len(encoding_dict_no_prop) == 2
        assert len(encoding_dict_prop) == 4

        filtered_encoding_dict_prop = [{key: val} for key, val in encoding_dict_prop.items() if 'scale' in val[0]]
        assert len(filtered_encoding_dict_prop) == 2

    # From: https://github.com/quic/aimet/blob/ce3dafe75d81893cdb8b45ba8abf53a672c28187/TrainingExtensions/torch/test/python/test_quantizer.py#L3733
    def test_multi_output_onnx_op(self):
        """
        Test mapping and exporting of output encodings for multiple output onnx op.
        """
        model = ModelWith5Output()
        dummy_input = torch.randn(1, 3, 224, 224)
        sim_model = copy.deepcopy(model)
        sim_model.cust = QuantizationMixin.from_module(sim_model.cust)
        sim_model.cust.input_quantizers[0] = QuantizeDequantize((),
                                                                bitwidth=8,
                                                                symmetric=False,
                                                                encoding_analyzer=MinMaxEncodingAnalyzer(()))
        for i in range(1, 5):
            sim_model.cust.output_quantizers[i] = QuantizeDequantize((),
                                                                     bitwidth=8,
                                                                     symmetric=False,
                                                                     encoding_analyzer=MinMaxEncodingAnalyzer(()))

        with aimet_nn.compute_encodings(sim_model):
            _ = sim_model(dummy_input)

        with tempfile.TemporaryDirectory() as path:
            QuantizationSimModel.export_onnx_model_and_encodings(path, 'module_with_5_output', model, sim_model,
                                                                 dummy_input,
                                                                 onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                                                                 propagate_encodings=False)
            with open(os.path.join(path, "module_with_5_output.encodings")) as json_file:
                activation_encodings = json.load(json_file)['activation_encodings']
                assert '7' not in activation_encodings
                assert set(['8', '9', '10', '11', 't.1']).issubset(activation_encodings.keys())

    # From: https://github.com/quic/aimet/blob/ce3dafe75d81893cdb8b45ba8abf53a672c28187/TrainingExtensions/torch/test/python/test_quantizer.py#L1935
    def test_mapping_encoding_for_torch_module_with_multiple_onnx_ops(self):
        """
        Test the input and output encoding map to input/output at subgraph level when a torch module generates
        multiple onnx ops i.e. a sub-graph
        """
        dummy_input = torch.randn(1, 4, 256, 512)
        model = SoftMaxAvgPoolModel()

        sim_model  = copy.deepcopy(model)
        sim_model.sfmax = QuantizationMixin.from_module(sim_model.sfmax)
        sim_model.sfmax.input_quantizers[0] = QuantizeDequantize((),
                                                                 bitwidth=8,
                                                                 symmetric=False,
                                                                 encoding_analyzer=MinMaxEncodingAnalyzer(()))
        sim_model.sfmax.output_quantizers[0] = QuantizeDequantize((),
                                                                  bitwidth=8,
                                                                  symmetric=False,
                                                                  encoding_analyzer=MinMaxEncodingAnalyzer(()))

        sim_model.avgpool = QuantizationMixin.from_module(sim_model.avgpool)
        sim_model.avgpool.input_quantizers[0] = QuantizeDequantize((),
                                                                  bitwidth=8,
                                                                  symmetric=False,
                                                                  encoding_analyzer=MinMaxEncodingAnalyzer(()))
        sim_model.avgpool.output_quantizers[0] = QuantizeDequantize((),
                                                                    bitwidth=8,
                                                                    symmetric=False,
                                                                    encoding_analyzer=MinMaxEncodingAnalyzer(()))
        with aimet_nn.compute_encodings(sim_model):
            _ = sim_model(dummy_input)

        with tempfile.TemporaryDirectory() as path:
            QuantizationSimModel.export_onnx_model_and_encodings(path, "sfmaxavgpool_model", model, sim_model,
                                                                 dummy_input, export_args, propagate_encodings=False)
            with open(os.path.join(path, "sfmaxavgpool_model" + ".encodings")) as json_file:
                encoding_data = json.load(json_file)

        assert len(encoding_data["activation_encodings"]) == 3

    @torch.no_grad()
    @pytest.mark.skipif(version.parse(torch.__version__) >= version.parse("2.1.2"),
                        reason="Results in RuntimeError when exporting, needs further debugging.")
    def test_conditional_export(self):
        """ Test exporting a model with conditional paths """
        model = SimpleConditional()
        model.eval()
        inp = torch.randn(1, 3)
        true_tensor = torch.tensor([1])
        false_tensor = torch.tensor([0])

        def forward_callback(model, _):
            model(inp, true_tensor)
            model(inp, false_tensor)

        qsim = QuantizationSimModel(model, dummy_input=(inp, true_tensor))
        qsim.compute_encodings(forward_callback, None)

        with aimet_nn.compute_encodings(qsim.model):
            forward_callback(qsim.model, None)

        with tempfile.TemporaryDirectory() as path:
            qsim._export_conditional(path, 'simple_cond', dummy_input=(inp, false_tensor),
                                     forward_pass_callback=forward_callback, forward_pass_callback_args=None)

            with open(os.path.join(path, 'simple_cond.encodings')) as f:
                encodings = json.load(f)
                # verifying the encoding against default eAI HW cfg
                # activation encodings -- input, linear1 out, prelu1 out, linear2 out, prelu2 out, softmax out
                assert 6 == len(encodings['activation_encodings'])
                # param encoding -- linear 1 & 2 weight, prelu 1 & 2 weight
                assert 4 == len(encodings['param_encodings'])

        expected_encoding_keys = {"/linear1/Add_output_0",
                                  "/linear2/Add_output_0",
                                  "/prelu1/CustomMarker_1_output_0",
                                  "/prelu2/PRelu_output_0",
                                  "/softmax/CustomMarker_1_output_0",
                                  "_input.1",
                                  }
        assert encodings["activation_encodings"].keys() == expected_encoding_keys

    @torch.no_grad()
    def test_json_interchangeable(self):
        from aimet_torch.v1.quantsim import QuantizationSimModel as QuantizationSimModelV1
        from aimet_torch.v2.quantsim import QuantizationSimModel

        torch.manual_seed(0)

        model = resnet18().eval()
        dummy_input = torch.randn(1, 3, 224, 224)

        sim_v1 = QuantizationSimModelV1(model, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced)
        sim_v2 = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced)

        sim_v1.compute_encodings(lambda model, _: model(dummy_input), None)
        sim_v2.compute_encodings(lambda model, _: model(dummy_input), None)

        """
        When: export using v1 and v2 quantsim with the same model
        Then: the output encoding json file should have the same structure
              (same list of params/activations, each with same bitwidth/dtype/symmetry)
        """
        with tempfile.TemporaryDirectory() as path:
            sim_v1.export(path, 'v1', dummy_input)
            with open(os.path.join(path, 'v1_torch.encodings')) as f:
                v1_saved_encoding = json.load(f)

        with tempfile.TemporaryDirectory() as path:
            sim_v2.export(path, 'v2', dummy_input)
            with open(os.path.join(path, 'v2_torch.encodings')) as f:
                v2_saved_encoding = json.load(f)

        _assert_same_structure(v1_saved_encoding, v2_saved_encoding)


        """
        When: Import the same encoding to v1 and v2 quantsim
        Then:
            1) All the quantizers enabled/disabled in v1 quantsim should be equally enabled/disabled in v2
            2) The inference results of v1 and v2 should be close enough
        """
        with tempfile.TemporaryDirectory() as path:
            v1_encoding_path = os.path.join(path, 'v1_torch.encodings')
            with open(v1_encoding_path, 'w') as f:
                json.dump(v1_saved_encoding, f)

            sim_v1.load_and_freeze_encodings(v1_encoding_path)
            sim_v2.load_and_freeze_encodings(v1_encoding_path)

        for name, module in sim_v1.model.named_modules():
            if not isinstance(module, QcQuantizeWrapper):
                continue
            wrapper = module
            qmodule = get_layer_by_name(sim_v2.model, name)

            assert wrapper.input_quantizers[0].enabled == (qmodule.input_quantizers[0] is not None)
            assert wrapper.output_quantizers[0].enabled == (qmodule.output_quantizers[0] is not None)
            if hasattr(qmodule, 'weight'):
                assert wrapper.param_quantizers['weight'].enabled == (qmodule.param_quantizers['weight'] is not None)



        v1_logits = sim_v1.model(dummy_input)
        v2_logits = sim_v2.model(dummy_input)

        scale = v2_logits.encoding.scale.item()
        assert torch.allclose(v1_logits, v2_logits, atol=scale * 3) # Allow off-by-3 error

    @pytest.mark.parametrize('quantizer_cls', [QuantizeDequantize, Quantize])
    @pytest.mark.parametrize('rounding_dtype', [torch.float32, torch.float64])
    def test_exported_weight(self, quantizer_cls, rounding_dtype):
        """
        Test to check if the exported weight remains unchanged after quantization,
        regardless of the rounding method used.
        """
        model = BasicConv2d(kernel_size=3)
        dummy_input = torch.randn(2, 64, 8, 8)
        param_bitwidth = 8

        # Set model weight to odd numbers so that quantized weights fall between rounding border
        with torch.no_grad():
            model.conv.weight.data = torch.randint(-127, 127, model.conv.weight.shape, dtype=torch.float32) * 2 + 1

        sim = QuantizationSimModel(model, dummy_input, default_param_bw=param_bitwidth)
        
        # Replace param quantizer with param quantizer of parametrized type
        param_q = sim.model.conv.param_quantizers['weight']
        sim.model.conv.param_quantizers['weight'] = quantizer_cls(param_q.shape, param_q.qmin, param_q.qmax, param_q.symmetric)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        # Adjust delta of the weight quantizer so that quantized weights fall between rounding border
        sim.model.conv.param_quantizers['weight'].set_range(torch.tensor(-255.0), torch.tensor(255.0))

        with tempfile.TemporaryDirectory() as tmp_dir:
            conv_weights = []

            # Gather weights from exported .pth file
            sim.export(tmp_dir, 'model_for_weight_export', dummy_input)
            exported_model = torch.load(os.path.join(tmp_dir, 'model_for_weight_export.pth'))
            conv_weights.append(exported_model.conv.weight.detach())

            # Gather weights from exported .onnx file
            onnx_model = onnx.load(os.path.join(tmp_dir, 'model_for_weight_export.onnx'))
            for tensor in onnx_model.graph.initializer:
                if tensor.name == 'conv.weight':
                    conv_weights.append(torch.tensor(onnx.numpy_helper.to_array(tensor)))
                    break
            else:
                assert False, "Cannot find conv weight inside ONNX model"
            
            # Check exported weight remains unchanged after quantization,
            # regardless of the rounding method used.
            for conv_weight in conv_weights:
                delta = sim.model.conv.param_quantizers['weight'].get_scale()
                assert torch.allclose(conv_weight, model.conv.weight.data, atol=delta.item())

                default_quant_result = torch.round(conv_weight / delta)
                conv_weight = conv_weight.to(rounding_dtype)
                delta = delta.to(rounding_dtype)
                assert torch.equal(default_quant_result,
                                   torch.trunc(conv_weight / delta + torch.sign(conv_weight) * 0.5).to(torch.float32))


def _assert_same_structure(v1_saved_encoding, v2_saved_encoding):
    assert v1_saved_encoding.keys() == v2_saved_encoding.keys()
    assert v1_saved_encoding['quantizer_args'] == v2_saved_encoding['quantizer_args']
    assert v1_saved_encoding['excluded_layers'] == v2_saved_encoding['excluded_layers']
    assert v1_saved_encoding['version'] == v2_saved_encoding['version']

    v1_activation_encodings = v1_saved_encoding['activation_encodings']
    v2_activation_encodings = v2_saved_encoding['activation_encodings']
    assert v1_activation_encodings.keys() == v2_activation_encodings.keys()


    def assert_encoding_equal(encoding, other):
        assert encoding.keys() == other.keys()
        assert encoding['bitwidth'] == other['bitwidth']
        assert encoding['dtype'] == other['dtype']
        assert encoding['is_symmetric'] == other['is_symmetric']

        # NOTE: scale/offset/min/max may not be exactly equal
        # assert encoding['scale'] == other['scale']
        # assert encoding['offset'] == other['offset']
        # assert encoding['min'] == other['min']
        # assert encoding['max'] == other['max']

    for v1_encoding, v2_encoding in zip(v1_activation_encodings.values(), v2_activation_encodings.values()):
        for v1_enc, v2_enc in zip(v1_encoding.values(), v2_encoding.values()):
            for v1, v2 in zip(v1_enc.values(), v2_enc.values()):
                assert_encoding_equal(v1, v2)

    v1_param_encodings = v1_saved_encoding['param_encodings']
    v2_param_encodings = v2_saved_encoding['param_encodings']
    assert v1_param_encodings.keys() == v2_param_encodings.keys()

    for key in v1_param_encodings:
        assert len(v1_param_encodings[key]) == len(v2_param_encodings[key])
        for v1_enc, v2_enc in zip(v1_param_encodings[key], v2_param_encodings[key]):
            assert_encoding_equal(v1_enc, v2_enc)
