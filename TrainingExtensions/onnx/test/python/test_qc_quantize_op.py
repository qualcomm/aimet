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
import numpy as np
import onnx
import math
import onnxruntime as ort
from onnx import helper
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

def create_quant_info(encoding,
                      tensor_quantizer,
                      opMode,
                      useSymmetricEncoding=False,
                      enabled=True,
                      bitwidth=8):
    quant_info = libquant_info.QcQuantizeInfo()
    encoding.bw = bitwidth
    quant_info.encoding = [encoding]
    quant_info.opMode = opMode
    quant_info.useSymmetricEncoding = useSymmetricEncoding
    quant_info.enabled = enabled
    quant_info.tensorQuantizerRef = [libpymo.PtrToInt64(tensor_quantizer)]
    quant_info.isIntDataType = True
    quant_info.usePerChannelMode = False
    return quant_info

def create_per_channel_quant_info(encoding,
                      tensor_quantizer,
                      opMode,
                      useSymmetricEncoding=False,
                      enabled=True,
                      ch_idx=0,
                      bitwidth=8):
    quant_info = libquant_info.QcQuantizeInfo()
    for enc in encoding:
        enc.bw = bitwidth
    quant_info.encoding = encoding
    quant_info.opMode = opMode
    quant_info.useSymmetricEncoding = useSymmetricEncoding
    quant_info.enabled = enabled
    quant_info.tensorQuantizerRef = [libpymo.PtrToInt64(item) for item in tensor_quantizer]
    quant_info.channelAxis = ch_idx
    quant_info.isIntDataType = True
    quant_info.usePerChannelMode = True
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

        tensor_quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                   MAP_ROUND_MODE_TO_PYMO['stochastic'])
        cpp_encodings = libpymo.TfEncoding()
        quant_info = create_quant_info(cpp_encodings, tensor_quantizer, OpMode.updateStats,
                                       useSymmetricEncoding=False)
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        session.run(None, {'input': input_arr})
        encodings = tensor_quantizer.computeEncoding(cpp_encodings.bw,
                                                     quant_info.useSymmetricEncoding)
        print("Encoding returned: min={}, max={}, offset={}. delta={}, bw={}"
              .format(encodings.min, encodings.max, encodings.offset, encodings.delta, encodings.bw))
        assert encodings is not None
        assert quant_info.tensorQuantizerRef[0].isEncodingValid is True

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

        quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                            MAP_ROUND_MODE_TO_PYMO['nearest'])
        out_tensor = np.zeros(input_arr.shape).astype(np.float32)
        # Perform one-shot quant-dequant in python
        quantizer.updateStats(input_arr, False)
        enc = quantizer.computeEncoding(8, False)
        quantizer.quantizeDequantize(input_arr.copy(), out_tensor, enc.min,
                                     enc.max, 8, False)

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
        assert quant_info.tensorQuantizerRef[0].getStrictSymmetric() == True

        qc_op.use_unsigned_symmetric = False
        assert quant_info.tensorQuantizerRef[0].getUnsignedSymmetric()== False

        qc_op.use_unsigned_symmetric = True
        assert quant_info.tensorQuantizerRef[0].getUnsignedSymmetric() == True

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
        tensor_quantizers = []
        encodings = []
        for idx in range (input_shape[quant_axis]):
            tensor_quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                       MAP_ROUND_MODE_TO_PYMO['nearest'])
            tensor_quantizer.setStrictSymmetric(strict_symmetric)
            tensor_quantizer.setUnsignedSymmetric(unsigned_symmetric)
            tensor_quantizers.append(tensor_quantizer)
            encodings.append(libpymo.TfEncoding())

        quant_info = create_per_channel_quant_info(encodings, tensor_quantizers, OpMode.oneShotQuantizeDequantize,
                                                   useSymmetricEncoding=use_symmetric, ch_idx=quant_axis)

        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        quant_info.usePerChannelMode = False
        per_tensor_model = create_model_from_node(quant_node, input_arr.take(indices=0, axis=quant_axis).shape)
        session = build_session(per_tensor_model, available_providers)
        # Run each channel through a per-tensor quantizer
        for idx in range(input_shape[quant_axis]):
            channel_input = input_arr.take(indices=idx, axis=quant_axis)
            output = session.run(None, {'input': channel_input})[0]
            expected_output_arr.append(np.expand_dims(output, quant_axis))
            quant_info.opMode = OpMode.oneShotQuantizeDequantize
        expected_output_arr = np.concatenate(expected_output_arr, axis=quant_axis)

        quant_info.usePerChannelMode = True
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
        tensor_quantizers = []
        encodings = [libpymo.TfEncoding() for _ in range(4)]
        for index in range(3):
            tensor_quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                       MAP_ROUND_MODE_TO_PYMO['nearest'])
            tensor_quantizer.isEncodingValid = True
            tensor_quantizers.append(tensor_quantizer)
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
        quant_info = create_per_channel_quant_info(encodings, tensor_quantizers, OpMode.quantizeDequantize,
                                                   useSymmetricEncoding=True, ch_idx=0)
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

        tensor_quantizers = [libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                     MAP_ROUND_MODE_TO_PYMO['nearest']) for _ in range(len(encoding_max))]

        for t in tensor_quantizers:
            t.isEncodingValid = True

        quant_info = create_per_channel_quant_info(encodings, tensor_quantizers, OpMode.quantizeDequantize,
                                                   useSymmetricEncoding=True, ch_idx=channel_axis)

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

        # Set up the quantizer op
        cpp_encodings = [libpymo.TfEncoding() for _ in range(4)]
        tensor_quantizer = [libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                   MAP_ROUND_MODE_TO_PYMO['nearest']) for _ in range(4)]
        quant_info = create_per_channel_quant_info(cpp_encodings, tensor_quantizer, OpMode.updateStats,
                                                   useSymmetricEncoding=symmetric, ch_idx=channel_axis)
        quant_info.blockSize = block_size
        quant_info.blockAxis = block_axis
        session = create_qc_quantize_model_session(quant_info, input_shape)

        # Run calibration
        output_tensor = session.run(None, {'input': input_tensor})[0]

        # Compute encodings
        encodings = [quantizer.computeEncoding(bitwidth, symmetric) for quantizer in tensor_quantizer]

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

        # Set up the quantizer op
        cpp_encodings = [libpymo.TfEncoding() for _ in range(6)]
        tensor_quantizer = [libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                    MAP_ROUND_MODE_TO_PYMO['nearest']) for _ in range(6)]
        quant_info = create_per_channel_quant_info(cpp_encodings, tensor_quantizer, OpMode.updateStats,
                                                   useSymmetricEncoding=symmetric, ch_idx=channel_axis)
        quant_info.blockSize = block_size
        quant_info.blockAxis = block_axis
        session = create_qc_quantize_model_session(quant_info, input_shape)

        # Run calibration
        output_tensor = session.run(None, {'input': input_tensor})[0]

        # Compute encodings
        encodings = [quantizer.computeEncoding(bitwidth, symmetric) for quantizer in tensor_quantizer]

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

        # Set up the quantizer op
        cpp_encodings = [libpymo.TfEncoding() for _ in range(4)]
        tensor_quantizer = [libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                    MAP_ROUND_MODE_TO_PYMO['nearest']) for _ in range(4)]
        quant_info = create_per_channel_quant_info(cpp_encodings, tensor_quantizer, OpMode.oneShotQuantizeDequantize,
                                                   useSymmetricEncoding=symmetric, ch_idx=channel_axis)
        quant_info.blockSize = block_size
        quant_info.blockAxis = block_axis
        session = create_qc_quantize_model_session(quant_info, input_shape)

        # Run calibration
        output_tensor = session.run(None, {'input': input_tensor})[0]

        # Computed encodings should be symmetric and correspond to the absolute min/max in the block
        expected_max = np.max(np.abs(input_tensor.reshape(4, 3)), axis=1)
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
        quant_info.usePerChannelMode = False
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
        quant_info.usePerChannelMode = False
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
        qc_quantize_op = QcQuantizeOp(quant_info, bitwidth=16, op_mode=OpMode.quantizeDequantize)
        qc_quantize_op.data_type = QuantizationDataType.float
        encodings = qc_quantize_op.export_encodings("0.6.1")
        assert len(encodings) == 1
        assert encodings[0]["dtype"] == "float"
        assert encodings[0]["bitwidth"] == 16

        exported_encodings = qc_quantize_op.export_encodings("1.0.0")
        assert exported_encodings.keys() == {"enc_type", "dtype", "bw"}
        assert exported_encodings["dtype"] == "FLOAT"
        assert exported_encodings["bw"] == 16

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
