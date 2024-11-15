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

import contextlib
import itertools
import json
import os
import tempfile
import onnx.numpy_helper
import torch
import numpy as np
from onnx import load_model
import onnx
import onnxruntime as ort
import pytest
from onnxsim import simplify

from aimet_common import quantsim
from aimet_common import libquant_info
from aimet_common.defs import QuantScheme, QuantizationDataType, EncodingType
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim, set_blockwise_quantization_for_weights, _apply_constraints, clamp_activation_encodings, \
    set_grouped_blockwise_quantization_for_weights
from aimet_onnx.qc_quantize_op import OpMode, GroupedBlockQuantizeDequantize
from aimet_onnx.utils import make_dummy_input
from models.models_for_tests import SingleResidual
from models import models_for_tests, test_models
from models.models_for_tests import build_dummy_model, single_residual_model, BNAfterConv, multi_input_with_constant_model , multi_output_model, custom_add_model, build_lstm_gru_dummy_model, \
    transposed_conv_model, depthwise_transposed_conv_model, linear_split_into_matmul_add, _convert_to_onnx


def _compare_encodings(dst, src):
    return (dst.min == src.min and
            dst.max == src.max and
            dst.delta == src.delta and
            dst.offset == src.offset)


class DummyModel(SingleResidual):
    """
    Model
    """
    def __init__(self):
        super().__init__()
        # change padding size to 0, onnxruntime only support input size is the factor of output size for pooling
        self.conv4 = torch.nn.Conv2d(32, 8, kernel_size=2, stride=2, padding=0, bias=True)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        del self.bn1
        del self.bn2

    def forward(self, inputs):
        x = self.conv1(inputs)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Save the output of MaxPool as residual.
        residual = x

        x = self.conv2(x)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        # x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        residual = self.conv4(residual)
        residual = self.ada(residual)
        x += residual
        x = self.relu3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

@contextlib.contextmanager
def set_encoding_version(version):
    old_version = quantsim.encoding_version
    quantsim.encoding_version = version

    yield

    quantsim.encoding_version = old_version

class TestQuantSim:
    """Tests for QuantizationSimModel"""
    def test_insert_quantize_op_nodes(self):
        """ Test to insert qc quantize op to the graph"""
        model = build_dummy_model()
        dummy_input = make_dummy_input(model)
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, dummy_input, path=tempdir)
            assert len(sim.model.nodes()) == 14

            node_ls = [node.op_type for node in sim.model.nodes()]
            assert node_ls == ['Conv', 'Relu', 'MaxPool', 'Flatten', 'Gemm'] + ['QcQuantizeOp'] * 9

            # Check if qc quantize op node is correctly connect to the corresponding onnx node
            assert sim.model.find_node_by_name('QcQuantizeOp_input', [], sim.model.graph()).output[0] == \
                   sim.model.find_node_by_name('conv', [], sim.model.graph()).input[0]
            # Check if op_mode is set correctly for each qc quantize op node
            qc_quantize_op_dict = sim.get_qc_quantize_op()
            for name in sim.param_names:
                assert qc_quantize_op_dict[name].op_mode == OpMode.oneShotQuantizeDequantize
            for name in sim.activation_names:
                assert qc_quantize_op_dict[name].op_mode == OpMode.updateStats

    def test_create_quantsim_dynamic_batch_size(self):
        """ Test to insert qc quantize op to the graph"""
        model = BNAfterConv()
        inputs = torch.randn((2, 10, 24, 24))
        with tempfile.TemporaryDirectory() as tempdir:
            torch.onnx.export(model, inputs, os.path.join(tempdir, 'dummy_model.onnx'),
                              training=torch.onnx.TrainingMode.PRESERVE,
                              opset_version=12,
                              input_names=['input'], output_names=['output'],
                              dynamic_axes={
                                  'input': {0: 'batch_size'},
                                  'output': {0: 'batch_size'},
                              })
            onnx_model = load_model(os.path.join(tempdir, 'dummy_model.onnx'))
            dummy_input = make_dummy_input(onnx_model)
            sim = QuantizationSimModel(onnx_model, dummy_input, path=tempdir)
            sim.session.run(None, dummy_input)

    def test_compute_encodings(self):
        """Test to perform compute encodings"""
        model = build_dummy_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)

            for quantizer in sim.qc_quantize_op_dict:
                sim.qc_quantize_op_dict[quantizer].enabled = True

            for name, qc_op in sim.get_qc_quantize_op().items():
                assert qc_op.quant_info.tensorQuantizerRef[0].isEncodingValid is False

            def callback(session, args):
                in_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
                session.run(None, in_tensor)

            sim.compute_encodings(callback, None)

            for name, qc_op in sim.get_qc_quantize_op().items():
                assert qc_op.encodings[0].bw == 8

            for name, qc_op in sim.get_qc_quantize_op().items():
                assert qc_op.quant_info.tensorQuantizerRef[0].isEncodingValid is True
                assert qc_op.op_mode == OpMode.quantizeDequantize

    def test_export_model_with_quant_args(self):
        """Test to export encodings and model"""
        model = build_dummy_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, default_activation_bw=16, default_param_bw=16,
                                       quant_scheme=QuantScheme.post_training_tf, path=tempdir)

            for quantizer in sim.qc_quantize_op_dict:
                sim.qc_quantize_op_dict[quantizer].enabled = True

            def dummy_callback(session, args):
                pass

            sim.compute_encodings(dummy_callback, None)
            sim.export(tempdir, 'quant_sim_model_with_quant_args')
            with open(os.path.join(tempdir, 'quant_sim_model_with_quant_args.encodings')) as json_file:
                encoding_data = json.load(json_file)

            assert "quantizer_args" in encoding_data
            quantizer_args = encoding_data["quantizer_args"]
            assert quantizer_args["activation_bitwidth"] == 16
            assert quantizer_args["param_bitwidth"] == 16
            assert not quantizer_args["per_channel_quantization"]
            assert quantizer_args["quant_scheme"] == QuantScheme.post_training_tf.name
            assert quantizer_args["dtype"] == "int"
            assert "is_symmetric" in quantizer_args

    def test_export_model(self):
        """Test to export encodings and model"""
        model = build_dummy_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)

            for quantizer in sim.qc_quantize_op_dict:
                sim.qc_quantize_op_dict[quantizer].enabled = True

            def dummy_callback(session, args):
                pass

            sim.compute_encodings(dummy_callback, None)
            sim.export(tempdir, 'quant_sim_model')

            with open(os.path.join(tempdir, 'quant_sim_model.encodings'), 'rb') as json_file:
                encoding_data = json.load(json_file)
            activation_keys = list(encoding_data["activation_encodings"].keys())
            assert activation_keys == ['3', '4', '5', 'input', 'output']
            for act in activation_keys:
                act_encodings_keys = list(encoding_data["activation_encodings"][act][0].keys())
                assert act_encodings_keys == ['bitwidth', 'dtype', 'is_symmetric', 'max', 'min', 'offset', 'scale']

            param_keys = list(encoding_data['param_encodings'].keys())
            assert param_keys == ['conv_b', 'conv_w', 'fc_b', 'fc_w']
            for param in param_keys:
                param_encodings_keys = list(encoding_data["param_encodings"][param][0].keys())
                assert param_encodings_keys == ['bitwidth', 'dtype', 'is_symmetric', 'max', 'min', 'offset', 'scale']

    def test_export_model_1_0_0(self):
        """Test to export encodings and model in 1.0.0 format"""
        model = build_dummy_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir, config_file=get_path_for_per_channel_config())

            def dummy_callback(session, _):
                session.run(None, make_dummy_input(model))

            sim.compute_encodings(dummy_callback, None)
            with set_encoding_version("1.0.0"):
                sim.export(tempdir, 'quant_sim_model')

            with open(os.path.join(tempdir, 'quant_sim_model.encodings'), 'rb') as json_file:
                encoding_data = json.load(json_file)

            assert encoding_data["version"] == "1.0.0"
            assert isinstance(encoding_data["activation_encodings"], list)
            assert isinstance(encoding_data["param_encodings"], list)

            activation_keys = {enc["name"] for enc in encoding_data["activation_encodings"]}
            param_keys = {enc["name"] for enc in encoding_data["param_encodings"]}
            assert activation_keys == {'4', '5', 'input', 'output'}
            assert param_keys == {'conv_w', 'fc_w'}

            for enc in itertools.chain(encoding_data["param_encodings"], encoding_data["activation_encodings"]):
                assert isinstance(enc, dict)
                assert enc.keys() == {"name", "enc_type", "dtype", "bw", "is_sym", "scale", "offset"}
                assert isinstance(enc["scale"], list)
                assert enc["dtype"] == "INT"
                # Gemm layers do not use per-channel in the default_per_channel_config
                if enc["name"] == "conv_w":
                    assert enc["enc_type"] == EncodingType.PER_CHANNEL.name
                else:
                    assert enc["enc_type"] == EncodingType.PER_TENSOR.name

    def test_lstm_gru(self):
        """Test for LSTM and GRU dummy model"""
        model = build_lstm_gru_dummy_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)

            for quantizer in sim.qc_quantize_op_dict:
                sim.qc_quantize_op_dict[quantizer].enabled = True

            def callback(session, args):
                in_tensor = {'input': np.random.rand(1, 8, 64).astype(np.float32)}
                session.run(None, in_tensor)

            sim.compute_encodings(callback, None)

            for name, qc_op in sim.get_qc_quantize_op().items():
                assert qc_op.encodings[0].bw == 8

            for name, qc_op in sim.get_qc_quantize_op().items():
                assert qc_op.quant_info.tensorQuantizerRef[0].isEncodingValid is True
                assert qc_op.op_mode == OpMode.quantizeDequantize

            sim.export(tempdir, 'quant_sim_model')

            with open(os.path.join(tempdir, 'quant_sim_model.encodings'), 'rb') as json_file:
                encoding_data = json.load(json_file)
            activation_keys = list(encoding_data["activation_encodings"].keys())
            assert activation_keys == ['2', 'input', 'output']
            for act in activation_keys:
                act_encodings_keys = list(encoding_data["activation_encodings"][act][0].keys())
                assert act_encodings_keys == ['bitwidth', 'dtype', 'is_symmetric', 'max', 'min', 'offset', 'scale']

            param_keys = list(encoding_data['param_encodings'].keys())
            assert param_keys == ['gru_r_w', 'gru_w', 'lstm_r_w', 'lstm_w']
            for param in param_keys:
                param_encodings_keys = list(encoding_data["param_encodings"][param][0].keys())
                assert param_encodings_keys == ['bitwidth', 'dtype', 'is_symmetric', 'max', 'min', 'offset', 'scale']

    def test_single_residual(self):
        model = single_residual_model().model
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, use_cuda=False, path=tempdir)
            for quantizer in sim.qc_quantize_op_dict:
                sim.qc_quantize_op_dict[quantizer].enabled = True

            def dummy_callback(session, args):
                pass

            sim.compute_encodings(dummy_callback, None)
            sim.export(tempdir, 'quant_sim_model')

            with open(os.path.join(tempdir, 'quant_sim_model.encodings'), 'rb') as json_file:
                encoding_data = json.load(json_file)
            activation_keys = list(encoding_data["activation_encodings"].keys())

            for act in activation_keys:
                act_encodings_keys = list(encoding_data["activation_encodings"][act][0].keys())
                assert act_encodings_keys == ['bitwidth', 'dtype', 'is_symmetric', 'max', 'min', 'offset', 'scale']

            param_keys = list(encoding_data['param_encodings'].keys())
            for param in param_keys:
                param_encodings_keys = list(encoding_data["param_encodings"][param][0].keys())
                assert param_encodings_keys == ['bitwidth', 'dtype', 'is_symmetric', 'max', 'min', 'offset', 'scale']

            assert len(activation_keys + param_keys) == len(sim.qc_quantize_op_dict.keys())

    @pytest.mark.cuda
    def test_compare_encodings_cpu_gpu(self):
        """Test to compare encodings with PT"""
        def onnx_callback(session, inputs):
            in_tensor = {'input': inputs}
            session.run(None, in_tensor)
        np.random.seed(0)
        torch.manual_seed(0)

        inputs = np.random.rand(128, 3, 32, 32).astype(np.float32)
        model = DummyModel()
        model.eval()

        with tempfile.TemporaryDirectory() as tempdir:
            torch.onnx.export(model, torch.as_tensor(inputs), os.path.join(tempdir, 'dummy_model.onnx'),
                              training=torch.onnx.TrainingMode.PRESERVE,
                              input_names=['input'], output_names=['output'])

            onnx_model_cpu = load_model(os.path.join(tempdir, 'dummy_model.onnx'))
            onnx_model_gpu = load_model(os.path.join(tempdir, 'dummy_model.onnx'))

            onnx_sim_cpu = QuantizationSimModel(onnx_model_cpu, use_cuda=False,
                                                quant_scheme=QuantScheme.post_training_tf_enhanced, path=tempdir)
            onnx_sim_gpu = QuantizationSimModel(onnx_model_gpu, use_cuda=True,
                                                quant_scheme=QuantScheme.post_training_tf_enhanced, path=tempdir)

            for node in onnx_sim_gpu.model.graph().node:
                if node.op_type == "QcQuantizeOp":
                    if 'CUDAExecutionProvider' in ort.get_available_providers():
                        assert node.domain == "aimet.customop.cuda"
            for node in onnx_sim_cpu.model.graph().node:
                if node.op_type == "QcQuantizeOp":
                    assert node.domain == "aimet.customop.cpu"

            onnx_sim_cpu.compute_encodings(onnx_callback, inputs)
            onnx_sim_gpu.compute_encodings(onnx_callback, inputs)
            out_cpu = onnx_sim_cpu.session.run(None, {'input': inputs})[0]
            out_gpu = onnx_sim_gpu.session.run(None, {'input': inputs})[0]
            onnx_sim_cpu.export(tempdir, 'onnx_sim_cpu')
            onnx_sim_gpu.export(tempdir, 'onnx_sim_gpu')

            assert(np.max(np.abs(out_cpu - out_gpu)) < 0.05)
            print(np.max(np.abs(out_cpu - out_gpu)))

            with open(os.path.join(tempdir, 'onnx_sim_cpu.encodings')) as f:
                cpu_encodings = json.load(f)
            with open(os.path.join(tempdir, 'onnx_sim_gpu.encodings')) as f:
                gpu_encodings = json.load(f)

            for name in list(cpu_encodings['activation_encodings'].keys()):
                assert (np.max(np.abs(cpu_encodings['activation_encodings'][name][0]['max'] -
                                      gpu_encodings['activation_encodings'][name][0]['max'])) < 0.05)
                assert (np.max(np.abs(cpu_encodings['activation_encodings'][name][0]['min'] -
                                      gpu_encodings['activation_encodings'][name][0]['min'])) < 0.05)
                assert (np.max(np.abs(cpu_encodings['activation_encodings'][name][0]['scale'] -
                                      gpu_encodings['activation_encodings'][name][0]['scale'])) < 0.05)
                assert cpu_encodings['activation_encodings'][name][0]['offset'] == \
                       gpu_encodings['activation_encodings'][name][0]['offset']

            for name in list(cpu_encodings['param_encodings'].keys()):
                assert (np.max(np.abs(cpu_encodings['param_encodings'][name][0]['max'] -
                                      gpu_encodings['param_encodings'][name][0]['max'])) < 0.05)
                assert (np.max(np.abs(cpu_encodings['param_encodings'][name][0]['min'] -
                                      gpu_encodings['param_encodings'][name][0]['min'])) < 0.05)
                assert (np.max(np.abs(cpu_encodings['param_encodings'][name][0]['scale'] -
                                      gpu_encodings['param_encodings'][name][0]['scale'])) < 0.05)
                assert cpu_encodings['param_encodings'][name][0]['offset'] == \
                       gpu_encodings['param_encodings'][name][0]['offset']

    @pytest.mark.cuda
    def test_compare_encodings_cpu_gpu_fp16(self):
        """Test to compare encodings with PT"""
        np.random.seed(0)
        torch.manual_seed(0)

        inputs = np.random.rand(128, 3, 32, 32).astype(np.float32)
        model = DummyModel()
        model.eval()
        with tempfile.TemporaryDirectory() as tempdir:
            torch.onnx.export(model, torch.as_tensor(inputs), os.path.join(tempdir, 'dummy_model.onnx'),
                              training=torch.onnx.TrainingMode.PRESERVE,
                              input_names=['input'], output_names=['output'])

            onnx_model_cpu = load_model(os.path.join(tempdir, 'dummy_model.onnx'))
            onnx_model_gpu = load_model(os.path.join(tempdir, 'dummy_model.onnx'))

            onnx_sim_cpu = QuantizationSimModel(onnx_model_cpu, use_cuda=False,
                                                quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                default_data_type=QuantizationDataType.float, default_param_bw=16,
                                                default_activation_bw=16, path=tempdir)
            onnx_sim_gpu = QuantizationSimModel(onnx_model_gpu, use_cuda=True,
                                                quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                default_data_type=QuantizationDataType.float, default_param_bw=16,
                                                default_activation_bw=16, path=tempdir)

            for node in onnx_sim_gpu.model.graph().node:
                if node.op_type == "QcQuantizeOp":
                    if 'CUDAExecutionProvider' in ort.get_available_providers():
                        assert node.domain == "aimet.customop.cuda"
            for node in onnx_sim_cpu.model.graph().node:
                if node.op_type == "QcQuantizeOp":
                    assert node.domain == "aimet.customop.cpu"

            out_cpu = onnx_sim_cpu.session.run(None, {'input': inputs})[0]
            out_gpu = onnx_sim_gpu.session.run(None, {'input': inputs})[0]

            assert (np.max(np.abs(out_cpu - out_gpu)) < 0.05)

    def test_per_channel_quantization(self):
        model = single_residual_model().model
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, use_cuda=False, config_file=get_path_for_per_channel_config(),
                                       path=tempdir)
            def dummy_callback(session, args):
                in_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
                session.run(None, in_tensor)
            sim.qc_quantize_op_dict['fc.weight'].enable_per_channel_quantization()
            sim.compute_encodings(dummy_callback, None)

            sim.export(tempdir, 'encodings')
            with open(os.path.join(tempdir, 'encodings.encodings')) as json_file:
                encoding_data = json.load(json_file)

            for param_name in sim.param_names:
                qc_op = sim.qc_quantize_op_dict[param_name]
                if qc_op.quant_info.usePerChannelMode and qc_op.enabled:
                    num_channels = qc_op.tensor_quantizer_params.tensor_shape[qc_op.tensor_quantizer_params.channel_axis]
                    assert num_channels == len(qc_op.encodings)
                    assert num_channels == len(encoding_data['param_encodings'][param_name])
                    for encoding in qc_op.encodings:
                        assert encoding.bw == 8
                        assert encoding.min != encoding.max

    @pytest.mark.parametrize("model_factory", (transposed_conv_model, depthwise_transposed_conv_model))
    def test_per_channel_quant_conv_transpose(self, model_factory):
        model = model_factory()
        conv_transpose_weight_names = []
        for node in model.graph().node:
            if node.op_type == "ConvTranspose":
                conv_transpose_weight_names.append(node.input[1])

        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, use_cuda=False, config_file=get_path_for_per_channel_config(),
                                       path=tempdir)

            def dummy_callback(session, args):
                in_tensor = {'input': np.random.rand(10, 10, 4, 4).astype(np.float32)}
                session.run(None, in_tensor)

            sim.compute_encodings(dummy_callback, None)

            for param_name in sim.param_names:
                if param_name in conv_transpose_weight_names:
                    for weight in sim.model.graph().initializer:
                        if weight.name == param_name:
                            break
                    else:
                        raise RuntimeError(f"Param {param_name} not found in model")
                    qc_op = sim.qc_quantize_op_dict[param_name]
                    assert qc_op.quant_info.usePerChannelMode
                    assert qc_op.quant_info.enabled
                    assert qc_op.quant_info.channelAxis == 1
                    assert len(qc_op.encodings) == weight.dims[1]

    def test_load_encodings_ptq(self):
        model = single_residual_model().model
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)

            def callback(session, args):
                in_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
                session.run(None, in_tensor)

            dummy_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}

            sim.compute_encodings(callback, None)
            sim.export(tempdir, 'onnx_sim')

            out2 = sim.session.run(None, dummy_tensor)

            del sim

            sim = QuantizationSimModel(model, path=tempdir)
            load_encodings_to_sim(sim, os.path.join(tempdir, 'onnx_sim.encodings'))
            out3 = sim.session.run(None, dummy_tensor)

            assert np.allclose(out2, out3)

    def test_load_encodings_pcq(self):
        model = single_residual_model().model
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, config_file=get_path_for_per_channel_config(), path=tempdir)

            def callback(session, args):
                in_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
                session.run(None, in_tensor)

            dummy_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}

            sim.compute_encodings(callback, None)
            sim.export(tempdir, 'onnx_sim')

            out2 = sim.session.run(None, dummy_tensor)

            del sim

            sim = QuantizationSimModel(model, config_file=get_path_for_per_channel_config(), path=tempdir)
            load_encodings_to_sim(sim, os.path.join(tempdir, 'onnx_sim.encodings'))
            out3 = sim.session.run(None, dummy_tensor)
            assert np.allclose(out2, out3)

    def test_load_encodings_assertion(self):
        model = single_residual_model().model
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, config_file=get_path_for_per_channel_config(), path=tempdir)
            def callback(session, args):
                in_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
                session.run(None, in_tensor)

            sim.compute_encodings(callback, None)
            sim.export(tempdir, 'onnx_sim')
            model = multi_output_model().model
            sim = QuantizationSimModel(model, path=tempdir)
            with pytest.raises(AssertionError):
                load_encodings_to_sim(sim, os.path.join(tempdir, 'onnx_sim.encodings'), strict=False)

    @pytest.mark.parametrize('strict', [False, True])
    def test_load_encodings_strict_and_non_strict(self, strict):
        model = single_residual_model().model

        # Update weights for testing is_unsigned_symmetric override later
        weight_initializers = [i.name for i in model.graph.initializer if len(i.dims) > 1]
        weight_initializer_3 = [i for i in model.graph.initializer if i.name == weight_initializers[3]][0]
        weight_initializer_3_data = onnx.numpy_helper.to_array(weight_initializer_3)
        weight_initializer_3.raw_data = np.asarray(np.abs(weight_initializer_3_data), dtype=np.float32).tobytes()

        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, config_file=get_path_for_per_channel_config(), path=tempdir)

            conv_ops = [node for node in sim.model.model.graph.node if node.op_type == 'Conv']
            relu_ops = [node for node in sim.model.model.graph.node if node.op_type == 'Relu']
            avgpool_ops = [node for node in sim.model.model.graph.node if node.op_type == 'AveragePool']

            act_1 = conv_ops[0].output[0]
            act_2 = relu_ops[0].output[0]
            act_3 = avgpool_ops[0].output[0]
            act_4 = conv_ops[2].output[0]
            sim.get_qc_quantize_op()[act_1].enabled = True
            sim.get_qc_quantize_op()[act_2].enabled = False
            sim.get_qc_quantize_op()[act_3].data_type = QuantizationDataType.float
            sim.get_qc_quantize_op()[weight_initializers[0]].bitwidth = 16
            sim.get_qc_quantize_op()[act_4].bitwidth = 4
            sim.get_qc_quantize_op()[weight_initializers[1]].use_symmetric_encodings = False
            sim.get_qc_quantize_op()[weight_initializers[2]].use_strict_symmetric = True
            sim.get_qc_quantize_op()[weight_initializers[3]].use_unsigned_symmetric = True

            def callback(session, args):
                in_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
                session.run(None, in_tensor)

            dummy_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}

            sim.compute_encodings(callback, None)
            sim.export(tempdir, 'onnx_sim')
            out2 = sim.session.run(None, dummy_tensor)
            del sim

            sim = QuantizationSimModel(model, config_file=get_path_for_per_channel_config(), path=tempdir)
            if strict:
                with pytest.raises(AssertionError):
                    load_encodings_to_sim(sim, os.path.join(tempdir, 'onnx_sim.encodings'), strict=strict)
            else:
                mismatched_encodings = load_encodings_to_sim(sim, os.path.join(tempdir, 'onnx_sim.encodings'), strict=strict)
                out3 = sim.session.run(None, dummy_tensor)
                sim.export(tempdir, 'loaded_onnx_sim')

                assert sim.get_qc_quantize_op()[act_1].enabled
                assert not sim.get_qc_quantize_op()[act_2].enabled
                assert sim.get_qc_quantize_op()[act_3].data_type == QuantizationDataType.float
                assert sim.get_qc_quantize_op()[weight_initializers[0]].bitwidth == 16
                assert sim.get_qc_quantize_op()[act_4].bitwidth == 4
                assert not sim.get_qc_quantize_op()[weight_initializers[1]].use_symmetric_encodings
                assert sim.get_qc_quantize_op()[weight_initializers[2]].use_strict_symmetric
                assert sim.get_qc_quantize_op()[weight_initializers[3]].use_unsigned_symmetric
                assert len(mismatched_encodings) == 8
                assert np.allclose(out2, out3)

    def test_model_with_constants(self):
        model = multi_input_with_constant_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)
            assert sim.qc_quantize_op_dict['/add0/Constant_output_0'].enabled == True
            assert sim.qc_quantize_op_dict['/add2/Constant_output_0'].enabled == True


    def test_multiple_output_quantsim(self):
        model = multi_output_model()
        sample_input = np.random.rand(128, 3, 32, 32).astype(np.float32)
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model=model,
                                       quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       default_activation_bw=8,
                                       default_param_bw=8,
                                       path=tempdir)
            sim.session.run(None, {'input': sample_input})

    @pytest.mark.skip(reason="test requires exact version of torch that the code has built against.")
    def test_model_with_custom_ops(self):
        custom_ops_path = os.path.dirname(libquant_info.__file__)
        custom_ops_path = os.path.join(custom_ops_path, "customops")
        onnx_library = os.path.join(custom_ops_path, "libonnx_custom_add.so")

        def dummy_callback(session, args):
            calib_data = {'input': np.random.rand(1, 3, 64, 64).astype(np.float32)}
            _ = session.run(None, calib_data)

        model = custom_add_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model=model,
                                      quant_scheme=QuantScheme.post_training_tf_enhanced,
                                      default_activation_bw=8,
                                      default_param_bw=8,
                                      user_onnx_libs=[onnx_library],
                                      path=tempdir)
            sim.save_model_graph("./quantized_custom_model")
            sim.compute_encodings(dummy_callback, None)
            sim.export(tempdir, 'custom_op_model')


    @pytest.mark.parametrize("model", [models_for_tests.weight_matmul_model(10, 20),
                                       models_for_tests.weight_gemm_model(10, 20, False),
                                       models_for_tests.weight_gemm_model(10, 20, True)])
    def test_matmul_quantization_axis(self, model):
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                },
                "strict_symmetric": "False",
                "per_channel_quantization": "True"
            },
            "params": {
                "weight": {
                    "is_quantized": "True"
                }
            },
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        output_features = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
        dummy_input = make_dummy_input(model)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model=model,
                                       config_file=config_file,
                                       path=temp_dir)

            sim.compute_encodings(lambda session, _: session.run(None, dummy_input), None)
            assert len(sim.qc_quantize_op_dict["weight"].encodings) == output_features

    def test_linear_split_into_matmul_add(self):
        model = linear_split_into_matmul_add()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, default_activation_bw=16, path=tempdir)

            def callback(session, dummy_input):
                session.run(None, dummy_input)

            dummy_tensor = {'input': np.random.rand(1, 2, 4).astype(np.float32)}
            sim.compute_encodings(callback, dummy_tensor)
            sim.export(tempdir, 'linear_matmul_add_pattern')
            with open(os.path.join(tempdir, 'linear_matmul_add_pattern.encodings')) as json_file:
                encoding_data = json.load(json_file)
                # Ensure that the encodings for the second input of Add op (bias) isn't present in JSON file.
                assert len(encoding_data['activation_encodings']) == 3
                assert len(encoding_data['param_encodings']) == 1

    @pytest.mark.skip("OOM issues from high CPU memory usage, optimize quantsim memory usage before enabling")
    def test_large_model(self):
        """
        When: Model is > 2GB
        Then: 1) We can still run the model
              2) We can still export the model
              3) Exported model contains all weights
        """
        # First create a model with is >= 2GB
        # Model size: (2 ** 5 layers) * (2 ** 15 * 2 ** 15 weights/layer) * (4 bytes/weight) = 2 ** 31 bytes
        num_layers = 2 ** 5
        weight_shape = [2 ** 12, 2 ** 12]
        weights = []
        layers = []
        for idx in range(num_layers):
            layers.append(
                onnx.helper.make_node("MatMul", inputs=[f"act{idx}", f"weight_{idx}"], outputs=[f"act{idx+1}_relu"], name=f"matmul_{idx}")
            )
            layers.append(
                onnx.helper.make_node("Relu", inputs=[f"act{idx+1}_relu"], outputs=[f"act{idx + 1}"],
                                      name=f"relu_{idx}")
            )
            data = np.empty(weight_shape, dtype=np.float32)
            data[0][0] = idx # Prevents simplifier from combining weights
            weights.append(
                onnx.numpy_helper.from_array(data, name=f"weight_{idx}")
            )

        input_tensor = onnx.helper.make_tensor_value_info("act0", onnx.TensorProto.FLOAT, [1, weight_shape[0]])
        output_tensor = onnx.helper.make_tensor_value_info(f"act{num_layers}", onnx.TensorProto.FLOAT, [1, weight_shape[1]])
        graph = onnx.helper.make_graph(layers, "large_graph", initializer=weights, inputs=[input_tensor], outputs=[output_tensor])
        model = onnx.helper.make_model(graph)

        assert model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)
            sim.export(tempdir, "large_model")
            loaded_model = onnx.load(os.path.join(tempdir, "large_model.onnx"))
            # Check that all weights are contained in the loaded model
            assert len(loaded_model.graph.initializer) == len(model.graph.initializer)
            assert loaded_model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF
            assert sim.model.model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF

        # Check that the model data is unchanged
        for idx in range(num_layers):
            assert onnx.numpy_helper.to_array(sim.model.graph().initializer[idx])[0][0] == idx

    def test_op_params_to_ignore(self):
        model = models_for_tests.resize_op_model()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)
            # params of specific ops shouldn't be quantized (here resize op param is testified)
            assert not sim.qc_quantize_op_dict.get('const_scale', None)

    def test_groupnorm_exception_rule(self):
        model = models_for_tests.model_with_exceptional_ops()
        quantsim_config = {
            "defaults":
                {
                    "hw_version": "V73",
                    "ops":
                        {
                            "is_output_quantized": "True"
                        },
                    "params":
                        {
                            "is_quantized": "True",
                            "is_symmetric": "True"
                        },
                    "per_channel_quantization": "True",
                    "strict_symmetric": "False",
                    "unsigned_symmetric": "False"
                },
            "params": {
                "bias":
                    {
                        "is_quantized": "False"
                    }
            },
            "op_type": {
                "GroupNormalization":
                    {
                        "per_channel_quantization": "False",
                        "params": {
                            "bias":
                                {
                                    "is_quantized": "True"
                                }
                        }
                    },
            },
            "supergroups": [],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {
                "is_output_quantized": "True"
            }
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, default_param_bw=8, default_activation_bw=16,
                                       path=tempdir, config_file=os.path.join(tempdir, 'quantsim_config.json'))

            def callback(session, dummy_input):
                session.run(None, dummy_input)

            dummy_input = {'model_input': np.random.rand(1, 12, 8, 8).astype(np.float32)}
            sim.compute_encodings(callback, dummy_input)
            sim.export(tempdir, 'conv_matmul_groupnorm_model')

            with open(os.path.join(tempdir, 'conv_matmul_groupnorm_model.encodings')) as json_file:
                encoding_data = json.load(json_file)
                groupnorm_weight_enc = encoding_data['param_encodings']['groupnorm_0.scale'][0]
                groupnorm_bias_enc = encoding_data['param_encodings']['groupnorm_0.bias'][0]

                # groupnorm param-encodings should follow output-activation-encoding config
                assert groupnorm_weight_enc['bitwidth'] == 16
                assert groupnorm_weight_enc['is_symmetric'] == 'False'

                assert groupnorm_bias_enc['bitwidth'] == 16
                assert groupnorm_bias_enc['is_symmetric'] == 'False'

    def test_matmul_v73_lower_exception_rule(self):
        model = models_for_tests.model_with_exceptional_ops()
        quantsim_config = {
            "defaults":
                {
                    "hw_version": "V66",
                    "ops":
                        {
                            "is_output_quantized": "True"
                        },
                    "params":
                        {
                            "is_quantized": "True",
                            "is_symmetric": "False"
                        },
                    "per_channel_quantization": "True",
                    "strict_symmetric": "False",
                    "unsigned_symmetric": "False"
                },
            "params": {
                "bias":
                    {
                        "is_quantized": "False"
                    }
            },
            "op_type": {},
            "supergroups": [],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {
                "is_output_quantized": "True"
            }
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, default_param_bw=16, default_activation_bw=8,
                                       path=tempdir, config_file=os.path.join(tempdir, 'quantsim_config.json'))

            def callback(session, dummy_input):
                session.run(None, dummy_input)

            dummy_tensor = {'model_input': np.random.rand(1, 12, 8, 8).astype(np.float32)}
            sim.compute_encodings(callback, dummy_tensor)
            sim.export(tempdir, 'conv_matmul_groupnorm_model')

            with open(os.path.join(tempdir, 'conv_matmul_groupnorm_model.encodings')) as json_file:
                encoding_data = json.load(json_file)
                matmul_second_input = encoding_data['param_encodings']['matmul_0.weight'][0]

                # matmul's second input encoding should be of 8 bitwidth and symmetric
                assert matmul_second_input['bitwidth'] == 8
                assert matmul_second_input['is_symmetric'] == 'True'

    def test_matmul_v73_higher_exception_rule(self):
        model = models_for_tests.model_with_exceptional_ops()
        quantsim_config = {
            "defaults":
                {
                    "hw_version": "V73",
                    "ops":
                        {
                            "is_output_quantized": "True"
                        },
                    "params":
                        {
                            "is_quantized": "True",
                            "is_symmetric": "False"
                        },
                    "per_channel_quantization": "True",
                    "strict_symmetric": "False",
                    "unsigned_symmetric": "False"
                },
            "params": {
                "bias":
                    {
                        "is_quantized": "False"
                    }
            },
            "op_type": {},
            "supergroups": [],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {
                "is_output_quantized": "True"
            }
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, default_param_bw=16, default_activation_bw=8,
                                       path=tempdir, config_file=os.path.join(tempdir, 'quantsim_config.json'))

            def callback(session, dummy_input):
                session.run(None, dummy_input)

            dummy_tensor = {'model_input': np.random.rand(1, 12, 8, 8).astype(np.float32)}
            sim.compute_encodings(callback, dummy_tensor)
            sim.export(tempdir, 'conv_matmul_groupnorm_model')

            with open(os.path.join(tempdir, 'conv_matmul_groupnorm_model.encodings')) as json_file:
                encoding_data = json.load(json_file)
                matmul_first_input = encoding_data['activation_encodings']['/conv_0/output_0'][0]

                # if matmul's second input is 16bw then first input should also be 16bw
                assert matmul_first_input['bitwidth'] == 16

    def test_matmul_v73_exception_rule_matmul_branch(self, tmpdir):
        model = models_for_tests.add_matmul_model()
        quantsim_config = {
            "defaults":
                {
                    "hw_version": "V73",
                    "ops":
                        {
                            "is_output_quantized": "True"
                        },
                    "params":
                        {
                            "is_quantized": "True",
                            "is_symmetric": "False"
                        },
                    "per_channel_quantization": "True",
                    "strict_symmetric": "False",
                    "unsigned_symmetric": "False"
                },
            "params": {},
            "op_type": {
                "Gather":
                    {
                        "is_output_quantized": "False"
                    }
            },
            "supergroups": [],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }

        with open(os.path.join(tmpdir, 'quantsim_config.json'), 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(model, default_param_bw=16, default_activation_bw=16,
                                   path=tmpdir, config_file=os.path.join(tmpdir, 'quantsim_config.json'))

        def callback(session, dummy_input):
            session.run(None, dummy_input)

        dummy_tensor = {'input': np.random.rand(3, 3).astype(np.float32),
                        'input_2': np.random.rand(3, 3).astype(np.float32)}
        sim.compute_encodings(callback, dummy_tensor)

        quantizer_1 = sim.qc_quantize_op_dict.get("added_output")
        assert quantizer_1.bitwidth == 16
        assert quantizer_1.use_symmetric_encodings
        assert len(quantizer_1.encodings) == 1

    @pytest.mark.parametrize("model", (models_for_tests.pointwise_conv1d((1, 64, 32)),
                                       models_for_tests.conv_model((64, 64, 3, 3), (1, 64, 32, 32), (1, 64, 32, 32), transpose=False),
                                       models_for_tests.pointwise_conv3d((1, 64, 32, 32, 4))))
    def test_blockwise_quantization_conv(self, model):
        block_size = 16
        sim = QuantizationSimModel(model)
        set_blockwise_quantization_for_weights(sim, "Conv", 4, True, block_size=block_size, strict=True)
        dummy_input = make_dummy_input(model)

        sim.compute_encodings(lambda session, _: session.run(None, dummy_input), None)

        weight_quantizer = sim.get_qc_quantize_op()["weight"]
        assert weight_quantizer.quant_info.blockSize == block_size
        assert weight_quantizer.quant_info.usePerChannelMode
        assert weight_quantizer.quant_info.blockAxis == 1
        assert len(weight_quantizer.encodings) == 64 * 64 / block_size

    @pytest.mark.parametrize("model", (models_for_tests.pointwise_convtranspose1d((1, 64, 32)),
                                       models_for_tests.conv_model((64, 64, 3, 3), (1, 64, 32, 32), (1, 64, 32, 32), transpose=True),
                                       models_for_tests.pointwise_convtranspose3d((1, 64, 32, 32, 4))))
    def test_blockwise_quantization_convtranspose(self, model):
        block_size = 16
        sim = QuantizationSimModel(model)
        set_blockwise_quantization_for_weights(sim, "ConvTranspose", 4, True, block_size=block_size, strict=True)
        dummy_input = make_dummy_input(model)

        sim.compute_encodings(lambda session, _: session.run(None, dummy_input), None)

        weight_quantizer = sim.get_qc_quantize_op()["weight"]
        assert weight_quantizer.quant_info.blockSize == block_size
        assert weight_quantizer.quant_info.usePerChannelMode
        assert weight_quantizer.quant_info.blockAxis == 0
        assert len(weight_quantizer.encodings) == 64 * 64 / block_size

    @pytest.mark.parametrize("model", (models_for_tests.weight_gemm_model(in_features=16, out_features=32, transposed_weight=False),
                                       models_for_tests.weight_gemm_model(in_features=16, out_features=32, transposed_weight=True),
                                       models_for_tests.weight_matmul_model(in_features=16, out_features=32)))
    def test_blockwise_quantization_matmul(self, model):
        block_size = 4
        input_features = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
        output_features = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
        transposed_weight = model.graph.initializer[0].dims[0] == output_features
        sim = QuantizationSimModel(model)
        set_blockwise_quantization_for_weights(sim, ("MatMul", "Gemm"), 4, True, block_size=block_size, strict=True)
        dummy_input = make_dummy_input(model)

        sim.compute_encodings(lambda session, _: session.run(None, dummy_input), None)

        weight_quantizer = sim.get_qc_quantize_op()["weight"]
        assert len(weight_quantizer.encodings) == output_features * input_features / block_size
        assert weight_quantizer.quant_info.usePerChannelMode
        assert weight_quantizer.quant_info.channelAxis == (0 if transposed_weight else 1)
        assert weight_quantizer.quant_info.blockAxis == (1 if transposed_weight else 0)
        assert weight_quantizer.quant_info.blockSize == block_size
        sim.session.run(None, dummy_input)

    def test_blockwise_quantization_with_dynamic_matmul(self):
        block_size = 2
        model = models_for_tests.dynamic_matmul_model(batch_size=1)
        sim = QuantizationSimModel(model)
        set_blockwise_quantization_for_weights(sim, ("MatMul", "Gemm"), 4, True, block_size=block_size)

        assert sim.qc_quantize_op_dict["linear.weight"].quant_info.blockSize == 2

        for name, quantizer in sim.qc_quantize_op_dict.items():
            if name != "linear.weight":
                # Blockwise quantization should only be enabled for the linear layer
                assert quantizer.quant_info.blockSize == 0

    def test_blockwise_quantization_nonstrict(self):
        model = models_for_tests.weight_matmul_model(in_features=16, out_features=32)
        sim = QuantizationSimModel(model)
        with pytest.raises(ValueError):
            set_blockwise_quantization_for_weights(sim, ("MatMul", "Gemm"), 4, True, block_size=7, strict=True)

        set_blockwise_quantization_for_weights(sim, ("MatMul", "Gemm"), 4, True, block_size=7, strict=False)

        weight_quantizer = sim.get_qc_quantize_op()["weight"]
        assert weight_quantizer.quant_info.blockSize == 0
        sim.session.run(None, make_dummy_input(model))


    @pytest.mark.parametrize("model, block_size", ((models_for_tests.single_residual_model(), 4),
                                                   (test_models.linear_layer_model(), 64)))
    def test_blockwise_quantization(self, model, block_size, tmpdir):
        dummy_input = make_dummy_input(model.model)
        bq_layers = ("MatMul", "Conv", "Gemm")
        bq_weights = set()

        for node in model.graph().node:
            if node.op_type in bq_layers:
                bq_weights.add(node.input[1])

        # Input shape is not compatible with block size
        bq_weights.remove(model.graph().node[0].input[1])

        sim = QuantizationSimModel(model, dummy_input)
        set_blockwise_quantization_for_weights(sim, ("MatMul", "Conv", "Gemm"), 8, True, block_size, strict=False)
        sim.compute_encodings(lambda session, _: session.run(None, dummy_input), None)

        for name, quantizer in sim.qc_quantize_op_dict.items():
            if not quantizer.enabled:
                continue
            if name in bq_weights:
                assert quantizer.quant_info.usePerChannelMode
                assert quantizer.quant_info.blockSize == block_size
                assert len(quantizer.encodings) > 1
            else:
                assert quantizer.quant_info.blockSize == 0
                assert len(quantizer.encodings) == 1

        sim.export(tmpdir, "tmp_model")
        with open(os.path.join(tmpdir, "tmp_model.encodings")) as f:
            encodings = json.load(f)

        for key, enc in encodings["param_encodings"].items():
            if key not in bq_weights:
                assert len(enc) == 1
                continue
            for param in sim.model.graph().initializer:
                if param.name == key and key in bq_weights:
                    assert len(enc) == param.dims[0] * param.dims[1] / block_size

        for key, enc in encodings["activation_encodings"].items():
            assert len(enc) == 1

    def test_model_with_initializers_as_activations(self):
        model = models_for_tests.model_with_initializers_as_activations()
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, path=tempdir)

            def callback(session, dummy_input):
                session.run(None, dummy_input)

            dummy_tensor = {'model_input': np.random.rand(1, 3, 8, 8).astype(np.float32)}
            sim.compute_encodings(callback, dummy_tensor)
            sim.export(tempdir, 'model_with_initializers_as_activations')

            with open(os.path.join(tempdir, 'model_with_initializers_as_activations.encodings')) as json_file:
                encoding_data = json.load(json_file)

            assert all(x in [i.name for i in model.graph.initializer] for x in ['add_input2', 'mul_input2'])
            assert encoding_data['activation_encodings']['add_input2']
            assert encoding_data['activation_encodings']['mul_input2']

    def test_gather_exception_rule_for_float_data(self):
        model = models_for_tests.gather_op_model()
        quantsim_config = {
            "defaults":
                {
                    "hw_version": "V73",
                    "ops":
                        {
                            "is_output_quantized": "True"
                        },
                    "params":
                        {
                            "is_quantized": "True",
                            "is_symmetric": "True"
                        },
                    "per_channel_quantization": "False",
                    "strict_symmetric": "False",
                    "unsigned_symmetric": "False"
                },
            "params": {},
            "op_type": {
                "Gather":
                    {
                        "is_output_quantized": "False"
                    }
            },
            "supergroups": [],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, default_param_bw=8, default_activation_bw=16,
                                       path=tempdir, config_file=os.path.join(tempdir, 'quantsim_config.json'))

            def callback(session, dummy_input):
                session.run(None, dummy_input)

            dummy_input = {'model_input': np.asarray([[0, 1, 2, 3]], dtype=np.int64)}
            sim.compute_encodings(callback, dummy_input)
            sim.export(tempdir, 'gather_model')

            with open(os.path.join(tempdir, 'gather_model.encodings')) as json_file:
                encoding_data = json.load(json_file)
                gather_weight_enc = encoding_data['activation_encodings']['gather_weight'][0]

                # gather param-encodings should follow output-activation-encoding config
                assert gather_weight_enc['bitwidth'] == 16
                assert gather_weight_enc['is_symmetric'] == 'False'

    def test_gather_with_int_data(self):
        model = models_for_tests.gather_op_with_int_data_model()
        quantsim_config = {
            "defaults":
                {
                    "hw_version": "V73",
                    "ops":
                        {
                            "is_output_quantized": "True"
                        },
                    "params":
                        {
                            "is_quantized": "True",
                            "is_symmetric": "True"
                        },
                    "per_channel_quantization": "False",
                    "strict_symmetric": "False",
                    "unsigned_symmetric": "False"
                },
            "params": {},
            "op_type": {
                "Gather":
                    {
                        "is_output_quantized": "False"
                    }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            dummy_input = {'model_input': np.asarray([[0, 1, 2, 3]], dtype=np.int64)}

            sim = QuantizationSimModel(model, dummy_input, default_param_bw=8, default_activation_bw=16,
                                       path=tempdir, config_file=os.path.join(tempdir, 'quantsim_config.json'))

            def callback(session, dummy_input):
                session.run(None, dummy_input)

            sim.compute_encodings(callback, dummy_input)
            sim.export(tempdir, 'gather_model')

            with open(os.path.join(tempdir, 'gather_model.encodings')) as json_file:
                encoding_data = json.load(json_file)
                assert 'gather_weight' not in encoding_data['activation_encodings'].keys()

    @pytest.mark.parametrize("model, block_size", ((models_for_tests.single_residual_model(), 4),
                                                   (test_models.linear_layer_model(), 64)),)
    def test_low_power_blockwise_quantization(self, model, block_size, tmpdir):
        dummy_input = make_dummy_input(model.model)
        bq_layers = ("MatMul", "Conv", "Gemm")
        bq_weights = set()
        bitwidth = 4
        decompressed_bw = 8

        for node in model.graph().node:
            if node.op_type in bq_layers:
                bq_weights.add(node.input[1])

        # Input shape is not compatible with block size
        bq_weights.remove(model.graph().node[0].input[1])

        sim = QuantizationSimModel(model, dummy_input, default_param_bw=16, default_activation_bw=16)
        set_grouped_blockwise_quantization_for_weights(sim,
                                                       ("MatMul", "Conv", "Gemm"),
                                                       bitwidth,
                                                       decompressed_bw,
                                                       block_size,
                                                       strict=False)

        sim.compute_encodings(lambda session, _: session.run(None, dummy_input), None)
        for name, quantizer in sim.qc_quantize_op_dict.items():
            if not quantizer.enabled:
                continue
            if name in bq_weights:
                assert isinstance(quantizer, GroupedBlockQuantizeDequantize)
                assert quantizer.quant_info.usePerChannelMode
                assert quantizer.quant_info.blockSize == block_size
                assert len(quantizer.encodings) > 1
            else:
                assert quantizer.quant_info.blockSize == 0
                assert len(quantizer.encodings) == 1

        with set_encoding_version("1.0.0"):
            sim.export(tmpdir, "tmp_model")

        with open(os.path.join(tmpdir, "tmp_model.encodings")) as f:
            encodings = json.load(f)

        for enc in encodings["param_encodings"]:
            if enc["name"] not in bq_weights:
                assert enc["enc_type"] == EncodingType.PER_TENSOR.name
            else:
                assert enc["enc_type"] == EncodingType.LPBQ.name
                assert enc["compressed_bw"] == bitwidth
                assert enc["bw"] == decompressed_bw

    def test_lpbq_strict(self):
        model = models_for_tests.weight_matmul_model(in_features=16, out_features=32)
        sim = QuantizationSimModel(model, default_activation_bw=16, default_param_bw=16, default_data_type=QuantizationDataType.float)
        quantizers = set(sim.qc_quantize_op_dict.values())

        with pytest.raises(ValueError):
            set_grouped_blockwise_quantization_for_weights(sim, ("MatMul", "Gemm"), 4, 8, block_size=7, strict=True)
        """
        When: Call block size is incompatible with weight shape
        Then: Original quantizer/quant_info should be unchanged from the call
        """
        set_grouped_blockwise_quantization_for_weights(sim, ("MatMul", "Gemm"), 4, 8, block_size=7, strict=False)
        assert quantizers == set(sim.qc_quantize_op_dict.values())

        for quantizer in sim.qc_quantize_op_dict.values():
            assert quantizer.bitwidth == 16
            assert not quantizer.quant_info.usePerChannelMode
            assert quantizer.quant_info.blockSize == 0
            assert not quantizer.quant_info.isIntDataType

    def test_encoding_constraints(self, tmp_path):
        quantsim_config = {
            "defaults":
                {
                    "ops":
                        {
                            "is_output_quantized": "True"
                        },
                    "params":
                        {
                            "is_quantized": "True",
                            "is_symmetric": "True"
                        },
                    "per_channel_quantization": "False",
                    "strict_symmetric": "False",
                    "unsigned_symmetric": "False"
                },
            "params": {},
            "op_type": {
                "Softmax":
                    {
                        "encoding_constraints":
                            {
                                "min": 0.0,
                                "max": 1.0
                            }
                    },
                "Sigmoid":
                    {
                        "encoding_constraints":
                            {
                                "min": 0.0,
                                "max": 2.0
                            }
                    },
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        config_name = os.path.join(tmp_path, 'quantsim_config.json')
        with open(config_name, 'w') as f:
            json.dump(quantsim_config, f)
        model = models_for_tests.softmax_model()
        sim = QuantizationSimModel(model, config_file=config_name)
        sim.compute_encodings(lambda sess, _: sess.run(None, make_dummy_input(model)), None)
        assert sim.qc_quantize_op_dict['model_output'].encodings[0].max == 2.0
        assert sim.qc_quantize_op_dict['model_output'].encodings[0].min == 0.0
        assert sim.qc_quantize_op_dict['softmax.output'].encodings[0].max == 1.0
        assert sim.qc_quantize_op_dict['softmax.output'].encodings[0].min == 0.0
        assert sim.qc_quantize_op_dict['matmul.output'].encodings[0].max not in (1.0, 2.0)
        assert sim.qc_quantize_op_dict['matmul.output'].encodings[0].min != 0.0


class TestEncodingPropagation:

    def test_output(self):
        """
        Given: model as below

                   +-> q_in1 -> conv1 -> relu1 ---> q_out1 -------v
          [input] -+                                           concat -> q_out3 -> [output]
                   +-> q_in2 -> conv2 -> relu2 ---> q_out2 -------^
        """
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3,3,3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(3,3,3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                x1 = x2 = x
                x1 = self.conv1(x1)
                x1 = self.relu1(x1)
                x2 = self.conv2(x2)
                x2 = self.relu2(x2)
                return torch.cat([x1, x2])
        """
       When: _apply_constraints(True)

       Then: q_out1 and q_out2 are replaced with q_out3 as below

                  +-> q_in1 -> conv1 -> relu1 -> **q_out3** -----v
         [input] -+                                           concat -> q_out3- > [output]
                  +-> q_in2 -> conv2 -> relu2 -> **q_out3** -----^
        """
        pt_model = Model().eval()
        x = torch.randn(1, 3, 24, 24)
        model = _convert_to_onnx(pt_model, x)
        dummy_input = make_dummy_input(model.model)
        with _apply_constraints(True):
            sim = QuantizationSimModel(model, dummy_input)

            sim.compute_encodings(lambda session, _: session.run(None, dummy_input), None)
            assert _compare_encodings(sim.qc_quantize_op_dict['/relu1/Relu_output_0'].encodings[0],
                                      sim.qc_quantize_op_dict['output'].encodings[0])
            assert _compare_encodings(sim.qc_quantize_op_dict['/relu2/Relu_output_0'].encodings[0],
                                      sim.qc_quantize_op_dict['output'].encodings[0])

    def test_math_invariant(self):
        """
        Given: model as below

                   +-> q_in1 -> conv1 ---> relu1 -> q_out1 ------v
          [input] -+                                          concat -> q_out2 -> [output]
                   +-> q_in2 -> reshape -> permute --------------^
        """
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)
                self.relu1 = torch.nn.ReLU()

            def forward(self, x):
                x1 = x2 = x
                x1 = self.conv1(x1)
                x1 = self.relu1(x1)
                x2 = torch.reshape(x2, (-1, 24, 24, 3))
                x2 = torch.permute(x2, (0, 3, 1, 2))
                return torch.cat([x1, x2])
        """
        When: _apply_constraints(True)

        Then: q_out1 and q_in2 are replaced with q_out3 as below

                   +-> q_in1 -> conv1 ---> relu1 -----> **q_out2**- --------v
          [input] -+                                                     concat -> q_out2 -> [output] 
                   +-> **q_out2** -> reshape -> transpose -> permute -------^
        """
        pt_model = Model().eval()
        dummy_input = torch.randn(1, 3, 24, 24)
        model = _convert_to_onnx(pt_model, dummy_input)
        dummy_input = make_dummy_input(model.model)
        with _apply_constraints(True):
            sim = QuantizationSimModel(model, dummy_input)
            sim.compute_encodings(lambda session, _: session.run(None, dummy_input), None)

            assert _compare_encodings(sim.qc_quantize_op_dict['/relu1/Relu_output_0'].encodings[0],
                                      sim.qc_quantize_op_dict['output'].encodings[0])
            assert _compare_encodings(sim.qc_quantize_op_dict['input'].encodings[0],
                                      sim.qc_quantize_op_dict['output'].encodings[0])

    def test_concat_tree(self):
        """
        Given: model as below

                    +-> q_in1a -> conv1a -> q_out1a -> concat1 -> q_out1c -> reshape --+
                    +-> q_in1b -> conv1b -> q_out1b ------^                            v
          [input] --+                                                               concat3 -> q_out3 -> [output]
                    +-> q_in2a -> conv2a -> q_out2a -> concat2 -> q_out2c -------------^
                    +-> q_in2b -> conv2b -> q_out2b ------^
        """
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1a = torch.nn.Conv2d(3,3,3)
                self.conv1b = torch.nn.Conv2d(3,3,3)
                self.conv2a = torch.nn.Conv2d(3,3,3)
                self.conv2b = torch.nn.Conv2d(3,3,3)

            def forward(self, x):
                x1a = x1b = x2a = x2b = x
                x1a = self.conv1a(x1a)
                x1b = self.conv1b(x1b)
                x1 = torch.cat([x1a, x1b])
                x1 = torch.reshape(x1, (-1, 22, 22, 3))
                x1 = torch.permute(x1, (0, 3, 1, 2))
                x2a = self.conv2a(x2a)
                x2b = self.conv2b(x2b)
                x2 = torch.cat([x2a, x2b])
                return torch.cat([x1, x2])

        pt_model = Model().eval()
        dummy_input = torch.randn(1, 3, 24, 24)
        model = _convert_to_onnx(pt_model, dummy_input)
        dummy_input = make_dummy_input(model.model)
        """
        When: _apply_constraints(True)

        Then: All q_out{*} are replaced with q_out3 as below

                    +-> q_in1a -> conv1a -> *q_out3* -> concat1 -> *q_out3* -> reshape --+
                    +-> q_in1b -> conv1b -> *q_out3* ------^                             v
          [input] --+                                                                 concat3 -> q_out3 -> [output]
                    +-> q_in2a -> conv2a -> *q_out3* -> concat2 -> *q_out3* -------------^
                    +-> q_in2b -> conv2b -> *q_out3* ------^
        """
        with _apply_constraints(True):
            sim = QuantizationSimModel(model, dummy_input)
            sim.compute_encodings(lambda session, _: session.run(None, dummy_input), None)

            for cg_op in sim.connected_graph.ordered_ops:
                if cg_op.type in ['Conv', 'Concat']:
                    _, out_qtzr, __ = sim.get_op_quantizers(cg_op)
                    assert _compare_encodings(out_qtzr[0].encodings[0], sim.qc_quantize_op_dict['output'].encodings[0])

    @pytest.mark.parametrize('op_type_under_test', [torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.Upsample])
    def test_output_parametrized(self, op_type_under_test):
        """
        Given: model as below
           [input] -+-> q_in1 -> conv1 -> q_out1 -> op_type_under_test -> q_out2 -> [output]
        """
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3,3,3)
                self.op_type_under_test = op_type_under_test(3)
            def forward(self, x):
                x1 = self.conv1(x)
                return self.op_type_under_test(x1)
        """
       When: _apply_constraints(True)

       Then: q_out1 will be replaced with q_out2 as below
       
             [input] -+-> q_in1 -> conv1 -> *q_out2* -> op_type_under_test -> q_out2 -> [output]
        
        """
        pt_model = Model().eval()
        x = torch.randn(1, 3, 24, 24)
        model = _convert_to_onnx(pt_model, x)
        # simplifier required to transform torch.nn.Upsample into a single onnx Resize op
        model.model, _ = simplify(model.model)
        dummy_input = make_dummy_input(model.model)
        with _apply_constraints(True):
            sim = QuantizationSimModel(model, dummy_input)
            sim.compute_encodings(lambda session, _: session.run(None, dummy_input), None)

            for cg_op in sim.connected_graph.ordered_ops:
                if cg_op.type in ['Conv']:
                    _, out_qtzr, __ = sim.get_op_quantizers(cg_op)
                    assert _compare_encodings(out_qtzr[0].encodings[0], sim.qc_quantize_op_dict['output'].encodings[0])

    def test_integer_concat(self):
        """
        When: Model contains unquantizable layers with op_type in quantsim.op_types_to_tie_qtzrs
        Then: Error should not be thrown during quantsim init
        """
        model = models_for_tests.integer_concat_model()
        with _apply_constraints(True):
            sim = QuantizationSimModel(model)

    def test_clamp_activation_encodings(self):
        model = models_for_tests.matmul_add_model()
        dummy_input = {'model_input': np.expand_dims(np.identity(8, np.float32), axis=(0, 1))}
        quantsim_config = {
            "defaults":
                {
                    "hw_version": "V73",
                    "ops":
                        {
                            "is_output_quantized": "True"
                        },
                    "params":
                        {
                            "is_quantized": "True",
                            "is_symmetric": "False"
                        },
                    "per_channel_quantization": "False",
                    "strict_symmetric": "False",
                    "unsigned_symmetric": "False"
                },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {
                "is_output_quantized": "True"
            }
        }

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, dummy_input, path=tempdir, config_file=os.path.join(tempdir, 'quantsim_config.json'))

            def callback(session, dummy_input):
                session.run(None, dummy_input)

            sim.compute_encodings(callback, dummy_input)
            clamp_activation_encodings(sim, 100.0)

            sim.export(tempdir, 'matmul_add_quantsim')

            with open(os.path.join(tempdir, 'matmul_add_quantsim.encodings')) as json_file:
                encodings = json.load(json_file)

            add_act_encoding = encodings['activation_encodings']['add_1.output'][0]
            matmul_act_encoding = encodings['activation_encodings']['matmul_2.output'][0]

            assert round(add_act_encoding['max']) == 100.0
            assert round(matmul_act_encoding['max']) == 100.0
