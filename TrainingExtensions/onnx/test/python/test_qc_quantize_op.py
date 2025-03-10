# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
from packaging import version
import tempfile
import math
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, OperatorSetIdProto, TensorProto
import os
import pytest
from aimet_common import libpymo
from aimet_common.defs import QuantScheme, MAP_QUANT_SCHEME_TO_PYMO, MAP_ROUND_MODE_TO_PYMO, QuantizationDataType, EncodingType
from aimet_onnx.qc_quantize_op import QcQuantizeOp, OpMode, TensorQuantizerParams, GroupedBlockQuantizeDequantize
from aimet_common import libquant_info
from aimet_common.quantsim import calculate_delta_offset
from aimet_onnx import lpbq_utils


shared_library = os.path.dirname(libquant_info.__file__)
shared_library = os.path.join(shared_library, "libaimet_onnxrt_ops.so")

available_providers = [
    provider
    for provider in ort.get_available_providers()
    if provider not in {"TvmExecutionProvider", "TensorrtExecutionProvider"}
]

if "CUDAExecutionProvider" in available_providers:
    op_domain = "aimet.customop.cuda"
else:
    op_domain = "aimet.customop.cpu"
op_name = "QcQuantizeOp"
per_channel_op_name = "QcQuantizeOp"

def create_tensor_quantizer(tensor_shape,
                            bitwidth=8,
                            ch_axis=None,
                            block_axis=None,
                            block_size=0,
                            quant_scheme=QuantScheme.post_training_tf):
    shape = [1 for _ in tensor_shape]
    if ch_axis is not None:
        shape[ch_axis] = tensor_shape[ch_axis]
    if block_axis is not None:
        shape[block_axis] = tensor_shape[block_axis] // block_size

    return libpymo.BlockTensorQuantizer(shape, bitwidth, MAP_QUANT_SCHEME_TO_PYMO[quant_scheme])


def create_quant_info(tensor_quantizer,
                      opMode,
                      useSymmetricEncoding=False,
                      enabled=True):
    quant_info = libquant_info.QcQuantizeInfo()
    quant_info.tensorQuantizerRef = tensor_quantizer
    quant_info.opMode = opMode
    quant_info.useSymmetricEncoding = useSymmetricEncoding
    quant_info.enabled = enabled
    quant_info.isIntDataType = True
    return quant_info


def create_model_from_node(quant_node, shape):
    input_info = helper.make_tensor_value_info(name=quant_node.input[0], elem_type=helper.TensorProto.FLOAT,
                                               shape=shape)

    output_info = helper.make_tensor_value_info(name=quant_node.output[0], elem_type=helper.TensorProto.FLOAT,
                                                shape=shape)
    onnx_graph = helper.make_graph([quant_node],
                                   'dummy_graph', [input_info], [output_info],
                                   [])

    model = helper.make_model(onnx_graph)
    return model


def create_encoding(enc_min, enc_max, bitwidth, symmetric):
    enc_min = enc_min if isinstance(enc_min, list) else [enc_min]
    enc_max = enc_max if isinstance(enc_max, list) else [enc_max]
    encodings = []

    for qmin, qmax in zip(enc_min, enc_max):
        delta, offset = calculate_delta_offset(qmin, qmax, bitwidth, symmetric, False)
        encoding = libpymo.TfEncoding()
        encoding.min = qmin
        encoding.max = qmax
        encoding.bw = bitwidth
        encoding.delta = delta
        encoding.offset = offset
        encodings.append(encoding)

    return encodings

def build_session(model, providers):
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(shared_library)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        path_or_bytes=model.SerializeToString(),
        sess_options=sess_options,
        providers=providers,
    )
    return session


def create_qc_quantize_model_session(quant_info, input_shape):
    quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                  domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
    model = create_model_from_node(quant_node, input_shape)
    return build_session(model, available_providers)


class TestQcQuantizeOp:

    def test_update_stats_with_pymo(self):

        input_arr = np.random.rand(1, 3, 4, 4).astype(np.float32)

        tensor_quantizer = create_tensor_quantizer([], 8, quant_scheme=QuantScheme.post_training_tf)
        quant_info = create_quant_info(tensor_quantizer, OpMode.updateStats,
                                       useSymmetricEncoding=False)
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        session.run(None, {'input': input_arr})
        encodings = tensor_quantizer.computeEncodings(quant_info.useSymmetricEncoding)[0]
        print("Encoding returned: min={}, max={}, offset={}. delta={}, bw={}"
              .format(encodings.min, encodings.max, encodings.offset, encodings.delta, encodings.bw))
        assert encodings is not None
        tensor_quantizer.setEncodings([encodings])
        assert quant_info.tensorQuantizerRef.isEncodingValid

    def test_quantize_dequantize_with_pymo(self):

        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                             quant_scheme=QuantScheme.post_training_tf,
                             rounding_mode='nearest',
                             op_mode=OpMode.oneShotQuantizeDequantize,
                             bitwidth=8,
                             use_symmetric_encodings=False,
                             )

        session.run(None, {'input': input_arr})
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 1
        encodings.min = -5.0
        encodings.delta = (1 + 5) / 255.
        encodings.offset = - 5.0 / encodings.delta

        qc_op.load_encodings([encodings])

        output = session.run(None, {'input': input_arr})[0]

        assert np.max(output) <= 1.1
        assert np.min(output) >= -5.1

    def test_quantize_dequantize_fp16(self):

        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        intermediate_output = input_arr.astype(np.float16)
        fp32_array = intermediate_output.astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                             quant_scheme=QuantScheme.post_training_tf,
                             rounding_mode='nearest',
                             op_mode=OpMode.oneShotQuantizeDequantize,
                             bitwidth=8,
                             use_symmetric_encodings=False,
                             )

        qc_op.op_mode = OpMode.quantizeDequantize
        output = session.run(None, {'input': input_arr})[0]

        assert np.allclose(output, fp32_array)

    def test_update_stats_quantize_dequantize(self):

        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        input_arr2 = np.random.randn(*input_arr.shape).astype(np.float32) * 10
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                             quant_scheme=QuantScheme.post_training_tf,
                             rounding_mode='nearest',
                             op_mode=OpMode.updateStats,
                             bitwidth=8,
                             use_symmetric_encodings=False,
                             )

        session.run(None, {'input': input_arr})[0]
        qc_op.compute_encodings()
        assert math.isclose(qc_op.encodings[0].max, 2.5, rel_tol=1e-2)
        assert math.isclose(qc_op.encodings[0].min, -7, rel_tol=1e-2)

        qc_op.op_mode = OpMode.quantizeDequantize
        output = session.run(None, {'input': input_arr2})[0]
        assert np.max(output) <= 2.6
        assert np.min(output) >= -7.1
        assert not np.allclose(output, input_arr2)

    def test_compare_one_shot_with_pymo(self):

        input_arr = np.random.randn(2, 3, 5, 1).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                             quant_scheme=QuantScheme.post_training_tf,
                             rounding_mode='nearest',
                             op_mode=OpMode.oneShotQuantizeDequantize,
                             bitwidth=8,
                             use_symmetric_encodings=False,
                             )

        quantizer = create_tensor_quantizer([], 8, quant_scheme=QuantScheme.post_training_tf)
        out_tensor = np.zeros(input_arr.shape).astype(np.float32)
        # Perform one-shot quant-dequant in python
        quantizer.updateStats(input_arr)
        enc = quantizer.computeEncodings(False)[0]
        out_tensor = (np.round(np.clip(input_arr / enc.delta - enc.offset, 0, 255)) + enc.offset) * enc.delta

        output = session.run(None, {'input': input_arr})[0]
        assert quant_info.encoding[0].max == enc.max
        assert quant_info.encoding[0].min == enc.min
        assert np.allclose(output, out_tensor)

    def test_one_shot_quantize_dequantize_asymmetric_cpu(self):
        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)

        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                             quant_scheme=QuantScheme.post_training_tf,
                             rounding_mode='nearest',
                             op_mode=OpMode.oneShotQuantizeDequantize,
                             bitwidth=8,
                             use_symmetric_encodings=False,
                             )

        output_oneshot = session.run(None, {'input': input_arr})[0]

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 2.5
        encodings.min = -7
        encodings.offset = -188
        qc_op.load_encodings([encodings])

        output_qdq = session.run(None, {'input': input_arr})


        assert np.allclose(output_oneshot, output_qdq)

    def test_one_shot_quantize_dequantize_symmetric_signed_cpu(self):
        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                             quant_scheme=QuantScheme.post_training_tf,
                             rounding_mode='nearest',
                             op_mode=OpMode.oneShotQuantizeDequantize,
                             bitwidth=8,
                             use_symmetric_encodings=True,
                             )
        output_oneshot = session.run(None, {'input': input_arr})

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 7
        encodings.min = -7
        encodings.offset = -128
        qc_op.load_encodings([encodings])

        output_qdq = session.run(None, {'input': input_arr})

        assert np.allclose(output_oneshot, output_qdq)

    def test_one_shot_quantize_dequantize_symmetric_unsigned_cpu(self):
        input_arr = np.asarray([[[[0, 1.2, 1.5, 4.0, 4.9, 5.3]]]]).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                             quant_scheme=QuantScheme.post_training_tf,
                             rounding_mode='nearest',
                             op_mode=OpMode.oneShotQuantizeDequantize,
                             bitwidth=8,
                             use_symmetric_encodings=True,
                             )

        qc_op.use_unsigned_symmetric = True

        output_oneshot = session.run(None, {'input': input_arr})

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.3
        encodings.min = 0.0
        encodings.offset = 0
        qc_op.load_encodings([encodings])

        output_qdq = session.run(None, {'input': input_arr})

        assert np.allclose(output_oneshot, output_qdq)

    @pytest.mark.cuda
    def test_one_shot_quantize_dequantize_cpu_vs_gpu(self):
        input_arr = np.asarray([[[[0, 1.2, 1.5, 4.0, 4.9, 5.3]]]]).astype(np.float32)
        quant_info_cpu = libquant_info.QcQuantizeInfo()
        quant_node_cpu = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                          domain="aimet.customop.cpu", quant_info=libpymo.PtrToInt64(quant_info_cpu))
        model_cpu = create_model_from_node(quant_node_cpu, input_arr.shape)
        session_cpu = build_session(model_cpu, available_providers)
        qc_op_cpu = QcQuantizeOp(quant_info=quant_info_cpu,
                                 quant_scheme=QuantScheme.post_training_tf,
                                 rounding_mode='nearest',
                                 op_mode=OpMode.oneShotQuantizeDequantize,
                                 bitwidth=8,
                                 use_symmetric_encodings=True)

        output_cpu = session_cpu.run(None, {'input': input_arr})

        quant_info_gpu = libquant_info.QcQuantizeInfo()
        quant_node_gpu = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                          domain="aimet.customop.cuda", quant_info=libpymo.PtrToInt64(quant_info_gpu))
        model_gpu = create_model_from_node(quant_node_gpu, input_arr.shape)
        session_gpu = build_session(model_gpu, available_providers)
        qc_op_gpu = QcQuantizeOp(quant_info=quant_info_gpu,
                                 quant_scheme=QuantScheme.post_training_tf,
                                 rounding_mode='nearest',
                                 op_mode=OpMode.oneShotQuantizeDequantize,
                                 bitwidth=8,
                                 use_symmetric_encodings=True)


        output_gpu = session_gpu.run(None, {'input': input_arr})


        assert np.alltrue(output_gpu[0] == output_cpu[0])

    def test_set_get_properties(self):
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        qc_op = QcQuantizeOp(quant_info=quant_info,
                             quant_scheme=QuantScheme.post_training_tf,
                             rounding_mode='nearest',
                             op_mode=OpMode.oneShotQuantizeDequantize,
                             bitwidth=8,
                             use_symmetric_encodings=True,
                             )
        qc_op.use_strict_symmetric = True
        assert quant_info.tensorQuantizerRef.getStrictSymmetric() == True

        qc_op.use_unsigned_symmetric = False
        assert quant_info.tensorQuantizerRef.getUnsignedSymmetric()== False

        qc_op.use_unsigned_symmetric = True
        assert quant_info.tensorQuantizerRef.getUnsignedSymmetric() == True

        qc_op.data_type = QuantizationDataType.float
        assert qc_op.data_type == QuantizationDataType.float
        assert qc_op.quant_info.isIntDataType == False

    @pytest.mark.parametrize("quant_axis", [0, 1])
    @pytest.mark.parametrize("use_symmetric,strict_symmetric,unsigned_symmetric", [(True, True, False), (True, False, True), (False, False, False)])
    def test_per_channel_one_shot_quantize_dequantize(self, use_symmetric, strict_symmetric, unsigned_symmetric, quant_axis):
        """
        Compares the output of per-channel quantization to the output of each channel passing through
        a per-tensor quantizer.
        """
        input_shape = (12, 6, 3, 3)
        input_arr = np.random.randn(*input_shape, ).astype(np.float32)
        expected_output_arr = []

        tensor_params = TensorQuantizerParams(input_shape, quant_axis, None)
        quant_info = libquant_info.QcQuantizeInfo()
        qc_op = QcQuantizeOp(quant_info=quant_info,
                             quant_scheme=QuantScheme.post_training_tf,
                             op_mode=OpMode.oneShotQuantizeDequantize,
                             bitwidth=8,
                             use_symmetric_encodings=use_symmetric,
                             tensor_quantizer_params=tensor_params
                             )

        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        per_tensor_model = create_model_from_node(quant_node, input_arr.take(indices=0, axis=quant_axis).shape)
        session = build_session(per_tensor_model, available_providers)
        # Run each channel through a per-tensor quantizer
        for idx in range(input_shape[quant_axis]):
            channel_input = input_arr.take(indices=idx, axis=quant_axis)
            output = session.run(None, {'input': channel_input})[0]
            expected_output_arr.append(np.expand_dims(output, quant_axis))
            quant_info.opMode = OpMode.oneShotQuantizeDequantize
        expected_output_arr = np.concatenate(expected_output_arr, axis=quant_axis)

        qc_op.enable_per_channel_quantization()
        per_channel_quant_node = helper.make_node(per_channel_op_name, inputs=['input'], outputs=['output'],
                                                  domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        per_channel_model = create_model_from_node(per_channel_quant_node, input_arr.shape)
        # Run the entire tensor through the per-channel quantizer
        session = build_session(per_channel_model, available_providers)
        output_per_channel = session.run(None, {'input': input_arr})[0]
        assert np.allclose(output_per_channel, expected_output_arr)

    def test_per_channel_quantize_dequantize(self):
        inp_array = np.array([[-7, -5, -3, 0, .1, 2.5],
                              [-7, -5, -3, 0, .1, 2.5],
                              [-7, -5, -3, 0, .1, 2.5],
                              [-7, -5, -3, 0, .1, 2.5]],
                             ).astype(np.float32)
        encodings = [libpymo.TfEncoding() for _ in range(4)]
        for index in range(3):
            encodings[index].bw = 8
            encodings[index].max = 3.81
            encodings[index].min = -3.84
            encodings[index].delta = 0.03
            encodings[index].offset = -128
        encodings[3].bw = 8
        encodings[3].max = 6.35
        encodings[3].min = -6.4
        encodings[3].delta = 0.05
        encodings[3].offset = -128
        tensor_quantizer = create_tensor_quantizer(inp_array.shape, encodings[0].bw, ch_axis=0, quant_scheme=QuantScheme.post_training_tf)
        tensor_quantizer.setEncodings(encodings)
        quant_info = create_quant_info(tensor_quantizer, OpMode.quantizeDequantize, useSymmetricEncoding=True)
        per_channel_quant_node = helper.make_node(per_channel_op_name, inputs=['input'], outputs=['output'],
                                                  domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))

        per_channel_model = create_model_from_node(per_channel_quant_node, inp_array.shape)
        per_channel_session = build_session(per_channel_model, available_providers)


        expected_out = np.array([[-3.84, -3.84, -3, 0, .089999996, 2.49],
                                     [-3.84, -3.84, -3, 0, .089999996, 2.49],
                                     [-3.84, -3.84, -3, 0, .089999996, 2.49],
                                     [-6.4, -5, -3, 0, .1, 2.5]],
                                    ).astype(np.float32)
        output = per_channel_session.run(None, {'input': inp_array})[0]
        assert np.allclose(output, expected_out)

    @pytest.mark.parametrize("input_arr", (np.asarray([0, -3.4028e38]).astype(np.float32),
                                           np.asarray([0, 3.4028e38]).astype(np.float32),
                                           np.asarray([0, -3.4028e38, 3.4028e38]).astype(np.float32)))
    @pytest.mark.parametrize("quant_scheme", (QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced))
    @pytest.mark.parametrize("symmetric", (True, False))
    def test_update_stats_extreme_values(self, quant_scheme, input_arr, symmetric):

        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                             quant_scheme=quant_scheme,
                             rounding_mode='nearest',
                             op_mode=OpMode.updateStats,
                             bitwidth=8,
                             use_symmetric_encodings=True,
                             )

        session.run(None, {'input': input_arr})
        qc_op.compute_encodings()

        assert qc_op.encodings[0].max >= 0
        assert qc_op.encodings[0].min <= 0
        assert qc_op.encodings[0].delta > 0



blockwise_qdq_test_1 = {
    "input_shape": (2, 3, 4),
    "block_axis": 0,
    "block_size": 1,
    "channel_axis": 1,
    "bitwidth": 8,
    "min": [0, 0, 0, -2, -2.5, 0],
    "max": [255. * 0.25, 255.0, 127.5, 508., 245. * 0.25, 2550.],
    "in_tensor": [
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
        0.126, 10.4, -12.3, 10000,
    ],
    "expected": [
        0.25, 10.5, 0, 63.75,       # scale = .25
        0., 10., 0., 255.,          # scale = 1
        0., 10.5, 0., 127.5,        # scale = 0.5
        0., 10., -2., 508.,         # scale = 2. offset=-1
        0.25, 10.5, -2.5, 61.25,    # scale = .25
        0., 10., 0, 2550.,          # scale = 10
    ],
}


blockwise_qdq_test_2 = {
    "input_shape": (4, 2, 2),
    "block_axis": 0,
    "block_size": 2,
    "channel_axis": 2,
    "bitwidth": 8,
    "min": [-64.0, -128.0, -256.0, -512.0],
    "max": [63.5, 127.0, 254.0, 508.0],
    "in_tensor": [
        -125.1, -125.1,    48.3, 48.3,
        68.3, 68.3,       -3.1, -3.1,

        -125.1, -125.1,    48.3, 48.3,
        68.3, 68.3,        -3.1, -3.1,
    ],
    "expected": [
        -64.0, -125.0,     48.5, 48.0,
        63.5, 68.0,       -3.0, -3.0,

        -126.0, -124.0,    48.0, 48.0,
        68.0, 68.0,        -4.0, -4.0
    ],
}

blockwise_qdq_test_3 = {
    "input_shape": (4, 4),
    "block_axis": 1,
    "block_size": 2,
    "channel_axis": 0,
    "bitwidth": 8,
    "min": [-1.28, -12.8, -128, -1280, 0, 0, 0, 0],
    "max": [1.27, 12.7, 127, 1270, 2.55, 25.5, 255, 2550],
    "in_tensor": [
        40.23, .0321, # Scale = 0.01
        -40.23, -.0321, # Scale = 0.1
        23.44, -2.3111, # scale = 1
        23.44, -2.3111, # scale = 10

        -1000.1, 334, # scale = 0.01
        23.1111, -23.1111, # scale = 0.1
        23.1111, -23.1111, # scale = 1
        -1, 100000, # scale = 10
    ],
    "expected": [
        1.27, .03, # Scale = 0.01
        -12.8, 0.0, # Scale = 0.1
        23, -2, # scale = 1
        20., 0, # scale = 10

        0, 2.55, # scale = 0.01
        23.1, 0., # scale = 0.1
        23, 0., # scale = 1
        0, 2550, # scale = 10
    ],
}

def isclose(x1, x2, atol=1e-4):
    return abs(x1 - x2) <= atol


class TestBlockwiseQuantizeOp:

    @pytest.mark.parametrize("test_set", (blockwise_qdq_test_1,
                                          blockwise_qdq_test_2,
                                          blockwise_qdq_test_3))
    def test_blockwise_quantize_dequantize(self, test_set):
        input_shape = test_set["input_shape"]
        block_axis = test_set["block_axis"]
        block_size = test_set["block_size"]
        channel_axis = test_set["channel_axis"]
        in_tensor = np.array(test_set["in_tensor"], dtype=np.float32).reshape(input_shape)
        expected_output = np.array(test_set["expected"], dtype=np.float32).reshape(input_shape)
        encoding_min = test_set["min"]
        encoding_max = test_set["max"]

        encodings = create_encoding(encoding_min, encoding_max, 8, False)

        tensor_quantizer = create_tensor_quantizer(input_shape, test_set["bitwidth"], channel_axis, block_axis, block_size, quant_scheme=QuantScheme.post_training_tf)
        tensor_quantizer.setEncodings(encodings)

        quant_info = create_quant_info(tensor_quantizer, OpMode.quantizeDequantize, useSymmetricEncoding=True)

        quant_info.blockAxis = block_axis
        quant_info.blockSize = block_size

        session = create_qc_quantize_model_session(quant_info, expected_output.shape)
        output = session.run(None, {"input": in_tensor})[0]

        assert np.allclose(output, expected_output)

    def test_blockwise_compute_encodings_symmetric(self):
        input_shape = (2, 6)
        block_axis = 1
        block_size = 3
        channel_axis = 0
        bitwidth = 8
        symmetric = True

        input_tensor = np.asarray([
            -5.4, 10, -2,
            3.5, 23.1, 2.,
            -10, -2, -1,
            -.1, 0.3, 0.1
        ]).astype(np.float32).reshape(input_shape)

        tensor_quantizer = create_tensor_quantizer(input_shape, bitwidth, channel_axis, block_axis, block_size, quant_scheme=QuantScheme.post_training_tf)
        quant_info = create_quant_info(tensor_quantizer, OpMode.updateStats, useSymmetricEncoding=symmetric)
        session = create_qc_quantize_model_session(quant_info, input_shape)

        # Run calibration
        output_tensor = session.run(None, {'input': input_tensor})[0]

        # Compute encodings
        encodings = tensor_quantizer.computeEncodings(symmetric)

        # Op should be passthrough in update_stats mode
        assert np.alltrue(input_tensor == output_tensor)

        # Computed encodings should be symmetric and correspond to the absolute min/max in the block
        expected_max = np.max(np.abs(input_tensor.reshape(4, 3)), axis=1)
        for idx, enc in enumerate(encodings):
            assert isclose(enc.max, expected_max[idx])
            assert isclose((enc.max + enc.min), -1 * enc.delta)
            assert enc.offset == -128
            assert isclose(enc.delta, enc.max / (2 ** (bitwidth - 1) - 1))

    def test_blockwise_compute_encodings_asymmetric(self):
        input_shape = (6, 2)
        block_axis = 0
        block_size = 2
        channel_axis = 1
        bitwidth = 8
        symmetric = False

        input_tensor = np.asarray([
            -5.4, 10, -2,
            3.5, 23.1, 2.,
            -10, -2, -1,
            -.1, 0.3, 0.1
        ]).astype(np.float32).reshape(input_shape)

        tensor_quantizer = create_tensor_quantizer(input_shape, bitwidth, channel_axis, block_axis, block_size, quant_scheme=QuantScheme.post_training_tf)
        quant_info = create_quant_info(tensor_quantizer, OpMode.updateStats, useSymmetricEncoding=symmetric)
        session = create_qc_quantize_model_session(quant_info, input_shape)

        # Run calibration
        output_tensor = session.run(None, {'input': input_tensor})[0]

        # Compute encodings
        encodings = tensor_quantizer.computeEncodings(symmetric)

        # Op should be passthrough in update_stats mode
        assert np.alltrue(input_tensor == output_tensor)

        # Computed encodings should be symmetric and correspond to the absolute min/max in the block
        expected_max = np.maximum(np.max(input_tensor.reshape(3, 2, 2), axis=1), 0).flatten()
        expected_min = np.minimum(np.min(input_tensor.reshape(3, 2, 2), axis=1), 0).flatten()
        for idx, enc in enumerate(encodings):
            assert isclose(enc.max, expected_max[idx], atol=enc.delta)
            assert isclose(enc.min, expected_min[idx], atol=enc.delta)
            assert isclose(enc.delta, (enc.max - enc.min) / (2 ** bitwidth - 1))
            assert isclose(enc.offset, enc.min / enc.delta)

    def test_blockwise_one_shot_compute_encodings(self):
        input_shape = (2, 6)
        block_axis = 1
        block_size = 3
        channel_axis = 0
        bitwidth = 8
        symmetric = True

        input_tensor = np.asarray([
            -5.4, 10, -2,
            3.5, 23.1, 2.,
            -10, -2, -1,
            -.1, 0.3, 0.1
        ]).astype(np.float32).reshape(input_shape)

        tensor_quantizer = create_tensor_quantizer(input_shape, bitwidth, channel_axis, block_axis, block_size, quant_scheme=QuantScheme.post_training_tf)
        quant_info = create_quant_info(tensor_quantizer, OpMode.oneShotQuantizeDequantize, useSymmetricEncoding=symmetric)
        session = create_qc_quantize_model_session(quant_info, input_shape)

        # Run calibration
        output_tensor = session.run(None, {'input': input_tensor})[0]

        # Computed encodings should be symmetric and correspond to the absolute min/max in the block
        expected_max = np.max(np.abs(input_tensor.reshape(4, 3)), axis=1)
        cpp_encodings = quant_info.encoding
        for idx, enc in enumerate(cpp_encodings):
            assert isclose(enc.max, expected_max[idx])
            assert isclose((enc.max + enc.min), -1 * enc.delta)
            assert enc.offset == -128
            assert isclose(enc.delta, enc.max / (2 ** (bitwidth - 1) - 1))

        # Compute the expected output given the computed encodings
        delta = np.array([enc.delta for enc in cpp_encodings]).astype(np.float32).reshape(-1, 1)
        offset = np.array([enc.offset for enc in cpp_encodings]).astype(np.float32).reshape(-1, 1)
        expected_out = (np.clip(np.round(input_tensor.reshape(4, 3) / delta - offset), 0, 2 ** bitwidth - 1) + offset) * delta

        # Op should produce the quantDequant output
        assert np.allclose(output_tensor, expected_out.reshape(output_tensor.shape))

    @pytest.mark.parametrize("symmetric, bitwidth, delta, offset", [(True, 8, 0.1, -128),
                                                                    (False, 16, 0.0125, -1000)])
    def test_export_per_tensor_int_encodings(self, symmetric, bitwidth, delta, offset):
        quant_info = libquant_info.QcQuantizeInfo()
        qc_quantize_op = QcQuantizeOp(quant_info, use_symmetric_encodings=symmetric, op_mode=OpMode.quantizeDequantize)
        assert qc_quantize_op.export_encodings() is None
        encoding = libpymo.TfEncoding()
        encoding.min = delta * offset
        encoding.max = delta * (offset + 2 ** bitwidth - 1)
        encoding.bw = bitwidth
        encoding.offset = offset
        encoding.delta = delta
        qc_quantize_op.update_quantizer_and_load_encodings([encoding], symmetric, False, False, QuantizationDataType.int)
        exported_encodings = qc_quantize_op.export_encodings("0.6.1")
        assert len(exported_encodings) == 1
        assert exported_encodings[0]["scale"] == delta
        assert exported_encodings[0]["offset"] == offset
        assert exported_encodings[0]["bitwidth"] == bitwidth
        assert exported_encodings[0]["dtype"] == "int"
        assert exported_encodings[0]["is_symmetric"] == str(symmetric)

        exported_encodings = qc_quantize_op.export_encodings("1.0.0")
        assert isinstance(exported_encodings, dict)
        assert exported_encodings.keys() == {"enc_type", "dtype", "bw", "is_sym", "scale", "offset"}
        assert exported_encodings["dtype"] == "INT"
        assert exported_encodings["enc_type"] == EncodingType.PER_TENSOR.name
        assert exported_encodings["bw"] == bitwidth
        assert exported_encodings["is_sym"] == symmetric
        assert isinstance(exported_encodings["scale"], list)
        assert isinstance(exported_encodings["offset"], list)
        assert len(exported_encodings["scale"]) == 1
        assert len(exported_encodings["offset"]) == 1
        assert exported_encodings["scale"][0] == delta
        assert exported_encodings["offset"][0] == offset

    @pytest.mark.parametrize("symmetric, bitwidth, delta, offset", [(True, 8, 0.1, -128),])
    def test_export_per_channel_int_encodings(self, symmetric, bitwidth, delta, offset):
        channel_axis = 0
        block_axis = 1
        tensor_shape = [5, 8]
        params = TensorQuantizerParams(tensor_shape, channel_axis, block_axis)

        quant_info = libquant_info.QcQuantizeInfo()
        qc_quantize_op = QcQuantizeOp(quant_info, use_symmetric_encodings=symmetric, op_mode=OpMode.quantizeDequantize,
                                      tensor_quantizer_params=params)
        qc_quantize_op.enable_per_channel_quantization()
        assert qc_quantize_op.export_encodings() is None
        encodings = [libpymo.TfEncoding() for _ in range(tensor_shape[channel_axis])]
        for encoding in encodings:
            encoding.min = delta * offset
            encoding.max = delta * (offset + 2 ** bitwidth - 1)
            encoding.bw = bitwidth
            encoding.offset = offset
            encoding.delta = delta
        qc_quantize_op.load_encodings(encodings)
        exported_encodings = qc_quantize_op.export_encodings("0.6.1")
        assert len(exported_encodings) == tensor_shape[channel_axis]

        exported_encodings = qc_quantize_op.export_encodings("1.0.0")
        assert exported_encodings.keys() == {"enc_type", "dtype", "bw", "is_sym", "scale", "offset"}
        assert exported_encodings["enc_type"] == EncodingType.PER_CHANNEL.name
        assert len(exported_encodings["scale"]) == tensor_shape[channel_axis]
        assert len(exported_encodings["offset"]) == tensor_shape[channel_axis]

        block_size = 4
        qc_quantize_op._enable_blockwise_quantization(block_size)
        encodings = [libpymo.TfEncoding() for _ in range(tensor_shape[channel_axis] * 2)]
        qc_quantize_op.load_encodings(encodings)
        exported_encodings = qc_quantize_op.export_encodings("1.0.0")
        assert exported_encodings.keys() == {"enc_type", "dtype", "bw", "is_sym", "scale", "offset", "block_size"}
        assert exported_encodings["enc_type"] == EncodingType.PER_BLOCK.name
        assert len(exported_encodings["scale"]) == tensor_shape[channel_axis] * 2
        assert exported_encodings["block_size"] == block_size

    def test_export_float_encodings(self):
        quant_info = libquant_info.QcQuantizeInfo()
        qc_quantize_op = QcQuantizeOp(quant_info, bitwidth=16, op_mode=OpMode.quantizeDequantize,
                                      tensor_quantizer_params=TensorQuantizerParams([2, 2], 0, 1))
        qc_quantize_op.enable_per_channel_quantization()
        qc_quantize_op.data_type = QuantizationDataType.float
        encodings = qc_quantize_op.export_encodings("0.6.1")
        assert len(encodings) == 1
        assert encodings[0]["dtype"] == "float"
        assert encodings[0]["bitwidth"] == 16

        exported_encodings = qc_quantize_op.export_encodings("1.0.0")
        assert exported_encodings.keys() == {"enc_type", "dtype", "bw"}
        assert exported_encodings["dtype"] == "FLOAT"
        assert exported_encodings["bw"] == 16
        assert exported_encodings["enc_type"] == EncodingType.PER_TENSOR.name

    def test_load_float_encodings(self):
        quant_info = libquant_info.QcQuantizeInfo()
        qc_quantize_op = QcQuantizeOp(quant_info, bitwidth=16, op_mode=OpMode.quantizeDequantize)
        qc_quantize_op.data_type = QuantizationDataType.float
        with pytest.raises(RuntimeError):
            qc_quantize_op.load_encodings([libpymo.TfEncoding()])

class TestLPBQOp:

    def test_lpbq_quantize_op(self):
        input_shape = (2, 9)
        scale = np.asarray([
            [1.6, 1.1222, .00001],
            [16, 2.56, 4.9],
        ], np.float32)
        offset = np.ones_like(scale) * -8
        expected_lpbq_scale = np.asarray([
            [1.6, 1.1, .1],
            [16, 3, 5]
        ], np.float32)
        expected_per_channel_scale = np.asarray([1.6 / 2 ** 4, 16 / 2 ** 4])
        bitwidth = 4
        decompressed_bw = 8
        quant_info = libquant_info.QcQuantizeInfo()
        tensor_quantizer_params = TensorQuantizerParams(input_shape, channel_axis=0, block_axis=1)
        lpbq_op = GroupedBlockQuantizeDequantize(quant_info,
                                                 bitwidth,
                                                 decompressed_bw,
                                                 block_size=3,
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 op_mode=OpMode.quantizeDequantize,
                                                 tensor_quantizer_params=tensor_quantizer_params)

        encodings = lpbq_utils.scale_offset_arrays_to_encodings(scale, offset, bitwidth)
        """
        When: Load blockwise encodings to an LPBQ quantizer
        Then: Quantizer should apply LPBQ to encodings during load_encodings
        """
        lpbq_op.load_encodings(encodings)
        lpbq_encodings = lpbq_op.get_encodings()
        lpbq_scale, lpbq_offset = lpbq_utils.encodings_to_scale_offset_arrays(lpbq_encodings, (2, 3))
        assert np.allclose(lpbq_scale, expected_lpbq_scale)
        assert np.allclose(lpbq_offset, offset)
        """
        Run LPBQ Quantizer in QDQ mode
        """
        session = create_qc_quantize_model_session(quant_info, input_shape)
        input_tensor = np.random.randn(*input_shape).astype(np.float32)
        output_tensor = session.run(None, {'input': input_tensor})[0]
        """
        Compute the expected LPBQ Output
        """
        input_tensor_bcast, scale_bcast = input_tensor.reshape((2, 3, 3)), expected_lpbq_scale.reshape((2, 3, 1))
        expected_output = (np.round(np.clip(input_tensor_bcast / scale_bcast, -8, 7)) * scale_bcast).reshape(
            input_shape)
        """
        Check that output matches expectation
        """
        assert np.allclose(expected_output, output_tensor)
        """
        Verify 1.0.0 export logic
        """
        exported_encodings = lpbq_op.export_encodings("1.0.0")
        expected_int_scale = [
            16, 11, 1,
            16, 3, 5
        ]
        assert exported_encodings.keys() == {"enc_type", "dtype", "bw", "is_sym", "scale", "offset", "block_size",
                                             "compressed_bw", "per_block_int_scale"}

        assert all(offset == -128 for offset in exported_encodings["offset"])
        assert exported_encodings["per_block_int_scale"] == expected_int_scale
        assert exported_encodings["compressed_bw"] == 4
        assert exported_encodings["bw"] == 8
        assert exported_encodings["enc_type"] == EncodingType.LPBQ.name
        assert np.allclose(np.asarray(exported_encodings["scale"]), np.asarray(expected_per_channel_scale))
        assert exported_encodings["offset"] == [-128, -128]

        with pytest.raises(NotImplementedError):
            lpbq_op.export_encodings("0.6.1")

    def test_compute_lpbq_encodings(self):
        input_shape = (4, 2)
        bitwidth = 4
        decompressed_bw = 8
        block_size = 2
        quant_info = libquant_info.QcQuantizeInfo()
        tensor_quantizer_params = TensorQuantizerParams(input_shape, channel_axis=1, block_axis=0)
        lpbq_op = GroupedBlockQuantizeDequantize(quant_info,
                                                 bitwidth,
                                                 decompressed_bw,
                                                 block_size=block_size,
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 op_mode=OpMode.updateStats,
                                                 tensor_quantizer_params=tensor_quantizer_params)

        # Note: computed delta = abs_max / num_positive_steps = abs_max / 7
        input_tensor = np.asarray([
            [7. * 32, -7 * 1.6],
            [-.35, 7.343],
            [7. * 13.334, 7 * -1.1112],
            [22.1, .11233]
        ], np.float32)
        expected_scale = np.asarray([
            [32., 1.6],
            [14, 1.1]
        ], np.float32)
        session = create_qc_quantize_model_session(quant_info, input_shape)
        session.run(None, {"input": input_tensor})
        lpbq_op.compute_encodings()

        encodings = lpbq_op.get_encodings()
        scale, _ = lpbq_utils.encodings_to_scale_offset_arrays(encodings, expected_scale.shape)
        assert np.allclose(scale, expected_scale)

    def test_grouped_block_qdq_perchannel_mode(self):
        input_shape = (4, 2)
        bitwidth = 4
        decompressed_bw = 8
        block_size = 0
        quant_info = libquant_info.QcQuantizeInfo()
        tensor_quantizer_params = TensorQuantizerParams(input_shape, channel_axis=1, block_axis=0)
        lpbq_op = GroupedBlockQuantizeDequantize(quant_info,
                                                 bitwidth,
                                                 decompressed_bw,
                                                 block_size=block_size,
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 op_mode=OpMode.updateStats,
                                                 tensor_quantizer_params=tensor_quantizer_params)

        # Note: computed delta = abs_max / num_positive_steps = abs_max / 7
        input_tensor = np.asarray([
            [7. * 32, -7 * 1.6],
            [-.35, 7.343],
            [7. * 13.334, 7 * -1.1112],
            [22.1, .11233]
        ], np.float32)
        expected_scale = np.asarray([
            [32., 1.6],
        ], np.float32)
        session = create_qc_quantize_model_session(quant_info, input_shape)
        session.run(None, {"input": input_tensor})
        lpbq_op.compute_encodings()

        encodings = lpbq_op.export_encodings("1.0.0")
        assert np.allclose(np.asarray(encodings["scale"]).astype("float32"), expected_scale)


def _onnx_QuantizeDequantizeLinear(input_shape, y_scale, y_zero_point, axis, block_size, output_dtype):
    op = OperatorSetIdProto()
    op.version = 21

    assert output_dtype in ("int8", "int16", "uint8", "uint16")

    x_int_dtype = TensorProto.INT16 if output_dtype == "int16" else \
                  TensorProto.INT8 if output_dtype == "int8" else \
                  TensorProto.INT4 if output_dtype == "int4" else \
                  TensorProto.UINT16 if output_dtype == "uint16" else \
                  TensorProto.UINT8 if output_dtype == "uint8" else \
                  TensorProto.UINT4 if output_dtype == "uint4" else \
                  None
    assert x_int_dtype is not None

    x = helper.make_tensor_value_info(name='x',
                                      elem_type=TensorProto.FLOAT,
                                      shape=input_shape)

    y_scale = numpy_helper.from_array(np.array(y_scale).astype('float32'), name='y_scale')
    if y_zero_point is not None:
        y_zero_point = numpy_helper.from_array(np.array(y_zero_point).astype(output_dtype), name='y_zero_point')

    y = helper.make_tensor_value_info(name='y',
                                      elem_type=TensorProto.FLOAT,
                                      shape=input_shape)

    quantize_node = helper.make_node('QuantizeLinear',
                                     inputs=['x', 'y_scale', 'y_zero_point'] if y_zero_point else ['x', 'y_scale'],
                                     outputs=['x_int'],
                                     axis=axis,
                                     block_size=block_size,
                                     output_dtype=x_int_dtype)

    dequantize_node = helper.make_node('DequantizeLinear',
                                       inputs=['x_int', 'y_scale', 'y_zero_point'] if y_zero_point else ['x_int', 'y_scale'],
                                       outputs=['y'],
                                       axis=axis,
                                       block_size=block_size)

    onnx_graph = helper.make_graph([quantize_node, dequantize_node],
                                   name='quantize_dequantize',
                                   inputs=[x],
                                   outputs=[y],
                                   initializer=[y_scale, y_zero_point] if y_zero_point is not None else [y_scale])

    model = helper.make_model(onnx_graph, opset_imports=[op])
    onnx.checker.check_model(model, True)

    return model


@pytest.mark.parametrize(
    # NOTE: In onnx, "axis" is overloaded with two meanings.
    #
    #         +- channel axis (if block size is None)
    # axis := |
    #         +- block axis (otherwise)
    "input_shape,    channel_axis, block_axis,  block_size", [
    ((10, 10, 1, 1), None,         None,        None), # per-tensor
    ((10, 10, 1, 1), 0,            None,        None), # per-channel with axis=0 (Convolution)
    ((10, 10, 1, 1), 1,            None,        None), # per-channel with axis=1 (Convolution)
    ((10, 10),       0,            None,        None), # per-channel with axis=0 (Linear/Gemm)
    ((10, 10),       1,            None,        None), # per-channel with axis=1 (Linear/Gemm)
    ((10, 10, 1, 1), 0,            1,           5),    # per-block with block_axis=1 (Convolution)
    ((10, 10, 1, 1), 1,            0,           5),    # per-block with block_axis=0 (Convolution)
    ((10, 10),       0,            1,           5),    # per-block with block_axis=1 (Linear/Gemm)
    ((10, 10),       1,            0,           5),    # per-block with block_axis=0 (Linear/Gemm)
])
@pytest.mark.parametrize(
    "bitwidth, symmetric", [
    (4,        True),
    (4,        False),
    (8,        True),
    (8,        True),
    (8,        False),
    (16,       True),
    (16,       False),
    (32,       True),
    # NOTE: Skipping since simulating int32 with non-zero offset is numerically very unstable
    # (32,       False),
])
def test_affine_encoding_schema_2_0_0(input_shape, channel_axis, block_axis, block_size,
                                      bitwidth, symmetric):
    """
    Given: QcQuantizeOp
    """
    input = np.random.randn(*input_shape).astype(np.float32)
    quant_params = TensorQuantizerParams(input_shape, channel_axis, block_axis)

    quant_info = libquant_info.QcQuantizeInfo()
    quant_info.isIntDataType = True
    if channel_axis is not None:
        quant_info.channelAxis = channel_axis
    if block_axis is not None:
        quant_info.blockAxis = block_axis

    quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                  domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
    model = create_model_from_node(quant_node, input.shape)
    session = build_session(model, available_providers)
    qtzr = QcQuantizeOp(quant_info=quant_info,
                        quant_scheme=QuantScheme.post_training_tf,
                        rounding_mode='nearest',
                        op_mode=OpMode.oneShotQuantizeDequantize,
                        bitwidth=bitwidth,
                        use_symmetric_encodings=symmetric,
                        tensor_quantizer_params=quant_params)

    if block_axis is not None:
        qtzr._enable_blockwise_quantization(block_size)
    elif channel_axis is not None:
        qtzr.enable_per_channel_quantization()

    _, = session.run(None, {'input': input})
    qtzr.compute_encodings()

    """
    When: Export encoding in 2.0.0.beta schema
    """
    encoding = qtzr.export_encodings("2.0.0.beta")

    """
    Then: Exported qnn encoding should contain:
            * "y_scale"
            * "y_zero_point"
            * "axis"
            * "block_size"
            * "output_dtype"

          all of which are defined as onnx::QuantizeLinear
    """
    y_scale = np.array(encoding["y_scale"])

    if block_axis is not None:
        assert y_scale.shape[channel_axis] == input_shape[channel_axis]
        assert y_scale.shape[block_axis] == input_shape[block_axis] // block_size
        assert all(dim == 1 for axis, dim in enumerate(y_scale.shape) if axis not in (channel_axis, block_axis))
    elif channel_axis is not None:
        assert y_scale.shape == (input_shape[channel_axis],)
    else:
        assert y_scale.shape == ()

    if symmetric:
        assert encoding["y_zero_point"] is None
    else:
        assert np.array(encoding["y_zero_point"]).shape == y_scale.shape

    assert encoding["axis"] == (channel_axis if block_axis is None else block_axis)
    assert encoding["block_size"] == block_size
    assert encoding["output_dtype"] == (f"int{bitwidth}" if symmetric else f"uint{bitwidth}")

    """
    Then: The output of onnx::QuantizeLinear followed by DequantizeLinear with the exported qnn encoding
          should be all-close to AIMET qdq output with off-by-one tolerance threshold
    """
    if bitwidth not in (8, 16):
        pytest.skip(reason="onnx::QuantizeLinear only supports these data types")

    if block_axis is not None and version.parse(ort.__version__) < version.parse("1.20.0"):
        pytest.skip(reason="Remaining tests require onnxruntime>=1.20 for blockwise QuantizeLinear")

    onnx_QuantizeLinear = _onnx_QuantizeDequantizeLinear(input_shape=input.shape,
                                                         y_scale=encoding["y_scale"],
                                                         y_zero_point=encoding["y_zero_point"],
                                                         axis=encoding["axis"],
                                                         block_size=encoding["block_size"],
                                                         output_dtype=encoding["output_dtype"])
    with tempfile.TemporaryDirectory() as tmp_dir:
        full_path = os.path.join(tmp_dir, "model.onnx")

        with open(full_path, "wb") as f:
            f.write(onnx_QuantizeLinear.SerializeToString())

        sess = ort.InferenceSession(full_path, providers=['CPUExecutionProvider'])
        ort_out, = sess.run(None, {'x': input})

    aimet_out = session.run(None, {'input': input})
    atol = y_scale # Allow off-by-one error
    if block_axis is not None:
        atol = atol.max(axis=block_axis, keepdims=True)
    elif channel_axis is not None:
        atol = atol.reshape(*(1 if axis != channel_axis else -1 for axis in range(input.ndim)))
    assert np.allclose(ort_out, aimet_out, atol=atol)


def _onnx_LPBQ(input_shape, per_block_int_scale, per_channel_float_scale,
               y_zero_point, axis, block_size, output_dtype):
    op = OperatorSetIdProto()
    op.version = 21

    assert y_zero_point is None

    x_int_dtype = TensorProto.INT16 if output_dtype == "int16" else \
                  TensorProto.INT8 if output_dtype == "int8" else \
                  TensorProto.INT4 if output_dtype == "int4" else \
                  TensorProto.UINT16 if output_dtype == "uint16" else \
                  TensorProto.UINT8 if output_dtype == "uint8" else \
                  TensorProto.UINT4 if output_dtype == "uint4" else \
                  None
    assert x_int_dtype is not None

    x = helper.make_tensor_value_info(name='x',
                                      elem_type=TensorProto.FLOAT,
                                      shape=input_shape)

    per_block_int_scale = numpy_helper.from_array(np.array(per_block_int_scale).astype('float32'),
                                                  name='per_block_int_scale')
    per_channel_float_scale = numpy_helper.from_array(np.array(per_channel_float_scale).astype('float32'),
                                                      name='per_channel_float_scale')

    y = helper.make_tensor_value_info(name='y',
                                      elem_type=TensorProto.FLOAT,
                                      shape=input_shape)

    mul_node = helper.make_node('Mul',
                                inputs=['per_block_int_scale', 'per_channel_float_scale'],
                                outputs=['y_scale'])

    quantize_node = helper.make_node('QuantizeLinear',
                                     inputs=['x', 'y_scale'],
                                     outputs=['x_int'],
                                     axis=axis,
                                     block_size=block_size,
                                     output_dtype=x_int_dtype)

    dequantize_node = helper.make_node('DequantizeLinear',
                                       inputs=['x_int', 'y_scale'],
                                       outputs=['y'],
                                       axis=axis,
                                       block_size=block_size)

    onnx_graph = helper.make_graph([mul_node, quantize_node, dequantize_node],
                                   name='lpbq',
                                   inputs=[x],
                                   outputs=[y],
                                   initializer=[per_block_int_scale, per_channel_float_scale])

    model = helper.make_model(onnx_graph, opset_imports=[op])
    onnx.checker.check_model(model, True)

    return model


@pytest.mark.parametrize(
    "input_shape,    block_axis,  block_size", [
    ((10, 50, 1, 1), 1,           5),    # per-block with block_axis=1 (Convolution)
    ((50, 10, 1, 1), 0,           5),    # per-block with block_axis=0 (Convolution)
    ((10, 50),       1,           5),    # per-block with block_axis=1 (Linear/Gemm)
    ((50, 10),       0,           5),    # per-block with block_axis=0 (Linear/Gemm)
])
@pytest.mark.parametrize(
    "compressed_bw, decompressed_bw", [
    (4,             8),
    (8,             16),
])
def test_lpbq_encoding_schema_2_0_0(input_shape, block_axis, block_size, compressed_bw, decompressed_bw):
    """
    Given: QcQuantizeOp
    """
    input = np.random.randn(*input_shape).astype(np.float32)
    channel_axis = 0 if block_axis == 1 else 1
    quant_params = TensorQuantizerParams(input_shape, channel_axis, block_axis)

    quant_info = libquant_info.QcQuantizeInfo()
    quant_info.isIntDataType = True
    quant_info.channelAxis = channel_axis
    quant_info.blockAxis = block_axis

    quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                  domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
    model = create_model_from_node(quant_node, input.shape)
    session = build_session(model, available_providers)
    qtzr = GroupedBlockQuantizeDequantize(quant_info,
                                          compressed_bw,
                                          decompressed_bw,
                                          block_size=block_size,
                                          quant_scheme=QuantScheme.post_training_tf,
                                          op_mode=OpMode.oneShotQuantizeDequantize,
                                          tensor_quantizer_params=quant_params)

    _, = session.run(None, {'input': input})
    qtzr.compute_encodings()

    """
    When: Export encoding in 2.0.0.beta schema
    """
    encoding = qtzr.export_encodings("2.0.0.beta")

    """
    Then: Exported qnn encoding should contain:
            * "per_block_int_scale"
            * "per_channel_float_scale"
            * "y_zero_point"
            * "axis"
            * "block_size"
            * "output_dtype"

          all of which are defined as onnx::QuantizeLinear except
          per_block_int_scale * per_channel_float_scale == y_scale
    """

    per_block_int_scale = np.array(encoding["per_block_int_scale"])
    per_channel_float_scale = np.array(encoding["per_channel_float_scale"])

    assert per_block_int_scale.ndim == per_channel_float_scale.ndim == input.ndim
    assert per_block_int_scale.shape[channel_axis] == input.shape[channel_axis]
    assert per_block_int_scale.shape[block_axis] == input.shape[block_axis] // block_size
    assert all(dim == 1 for axis, dim in enumerate(per_block_int_scale.shape)
               if axis not in (channel_axis, block_axis))
    assert per_channel_float_scale.shape[channel_axis] == input.shape[channel_axis]
    assert all(dim == 1 for axis, dim in enumerate(per_channel_float_scale.shape) if axis != channel_axis)

    assert encoding["y_zero_point"] is None
    assert encoding["axis"] == block_axis
    assert encoding["block_size"] == block_size
    assert encoding["output_dtype"] == f"int{compressed_bw}"


    """
    Then: The output of onnx::QuantizeLinear followed by DequantizeLinear with the exported qnn encoding
          should be all-close to AIMET qdq output with off-by-one tolerance threshold
    """
    if version.parse(ort.__version__) < version.parse("1.20.0"):
        pytest.skip(reason="Remaining tests require onnxruntime>=1.20 for blockwise QuantizeLinear")

    onnx_LPBQ = _onnx_LPBQ(input_shape=input.shape,
                           per_block_int_scale=encoding["per_block_int_scale"],
                           per_channel_float_scale=encoding["per_channel_float_scale"],
                           y_zero_point=encoding["y_zero_point"],
                           axis=encoding["axis"],
                           block_size=encoding["block_size"],
                           output_dtype=encoding["output_dtype"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        full_path = os.path.join(tmp_dir, "model.onnx")

        with open(full_path, "wb") as f:
            f.write(onnx_LPBQ.SerializeToString())

        sess = ort.InferenceSession(full_path, providers=['CPUExecutionProvider'])
        ort_out, = sess.run(None, {'x': input})

    aimet_out = session.run(None, {'input': input})
    y_scale = per_block_int_scale * per_channel_float_scale
    atol = y_scale.max(axis=block_axis, keepdims=True) # Allow off-by-one error
    assert np.allclose(ort_out, aimet_out, atol=atol)
