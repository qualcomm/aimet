# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import sys
from packaging import version
import tempfile
import math
import ml_dtypes
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, OperatorSetIdProto, TensorProto
import os
import platform
import pytest
from aimet_onnx.common import libpymo
from aimet_onnx.common.defs import (
    QuantScheme,
    MAP_QUANT_SCHEME_TO_PYMO,
    QuantizationDataType,
    EncodingType,
)
from aimet_onnx.qc_quantize_op import (
    QcQuantizeOp,
    OpMode,
    TensorQuantizerParams,
    GroupedBlockQuantizeDequantize,
    LPBQScaleQuantizer,
)
from aimet_onnx.common import libquant_info
from aimet_onnx.common.quantsim import calculate_delta_offset, _get_minimum_scale
from aimet_onnx import qtype
from aimet_onnx.utils import numpy_from_TfEncoding, numpy_to_TfEncoding
from aimet_onnx._encoding import AffineEncoding
from aimet_onnx.defs import QSpec, LPBQ, Blockwise, PerChannel, PerTensor
import aimet_onnx


FLOAT32_MIN = np.finfo(np.float32).min
FLOAT32_MAX = np.finfo(np.float32).max

_DEFAULT_IR_VERSION = 10


def _get_shared_library_name():
    if sys.platform == "win32":
        return "libaimet_onnxrt_ops.dll"
    elif sys.platform == "darwin":
        return "libaimet_onnxrt_ops.dylib"
    else:
        return "libaimet_onnxrt_ops.so"


shared_library = os.path.join(
    os.path.dirname(libquant_info.__file__),
    _get_shared_library_name(),
)

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


def create_tensor_quantizer(
    tensor_shape,
    bitwidth=8,
    ch_axis=None,
    block_axis=None,
    block_size=0,
    quant_scheme=QuantScheme.post_training_tf,
):
    shape = [1 for _ in tensor_shape]
    if ch_axis is not None:
        shape[ch_axis] = tensor_shape[ch_axis]
    if block_axis is not None:
        shape[block_axis] = tensor_shape[block_axis] // block_size

    return libpymo.BlockTensorQuantizer(
        shape, bitwidth, MAP_QUANT_SCHEME_TO_PYMO[quant_scheme]
    )


def create_quant_info(
    tensor_quantizer, opMode, useSymmetricEncoding=False, enabled=True
):
    quant_info = libquant_info.QcQuantizeInfo()
    quant_info.tensorQuantizerRef = tensor_quantizer
    quant_info.opMode = opMode
    quant_info.useSymmetricEncoding = useSymmetricEncoding
    quant_info.enabled = enabled
    quant_info.isIntDataType = True
    return quant_info


def create_model_from_node(
    quant_node, shape, float_dtype: onnx.TensorProto.DataType = TensorProto.FLOAT
):
    input_info = helper.make_tensor_value_info(
        name=quant_node.input[0], elem_type=float_dtype, shape=shape
    )

    output_info = helper.make_tensor_value_info(
        name=quant_node.output[0], elem_type=float_dtype, shape=shape
    )
    onnx_graph = helper.make_graph(
        [quant_node], "dummy_graph", [input_info], [output_info], []
    )

    model = helper.make_model(
        onnx_graph,
        opset_imports=[helper.make_operatorsetid("", 20)],
        ir_version=_DEFAULT_IR_VERSION,
    )
    return model


# (numpy dtype, onnx TensorProto dtype) pairs covering both 16-bit floats.
HALF_FLOAT_DTYPES = [
    (np.float16, TensorProto.FLOAT16),
    (ml_dtypes.bfloat16, TensorProto.BFLOAT16),
]


def run_session(session, feed_dict):
    """
    Wrapper over session.run which supports bfloat16 through io_binding
    """
    if not any(arr.dtype == ml_dtypes.bfloat16 for arr in feed_dict.values()):
        return session.run(None, feed_dict)

    binding = session.io_binding()
    for name, arr in feed_dict.items():
        arr = np.ascontiguousarray(arr)  # ortvalue requires contiguous array
        binding.bind_ortvalue_input(
            name,
            ort.OrtValue.ortvalue_from_numpy_with_onnx_type(arr, TensorProto.BFLOAT16),
        )

    out_bufs = []
    for output in session.get_outputs():
        out_buf = np.empty(output.shape, dtype=np.dtype("bfloat16"))
        binding.bind_ortvalue_output(
            output.name,
            ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                out_buf, TensorProto.BFLOAT16
            ),
        )
        out_bufs.append(out_buf)

    session.run_with_iobinding(binding)
    return out_bufs


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


def create_qc_quantize_model_session(
    quant_info, input_shape, float_dtype: onnx.TensorProto.DataType = TensorProto.FLOAT
):
    quant_node = helper.make_node(
        op_name,
        inputs=["input"],
        outputs=["output"],
        domain=op_domain,
        quant_info=libpymo.PtrToInt64(quant_info),
    )
    model = create_model_from_node(quant_node, input_shape, float_dtype)
    return build_session(model, available_providers)


def create_qc_quantize_model_session_fp16(quant_info, input_shape):
    quant_node = helper.make_node(
        op_name,
        inputs=["input"],
        outputs=["output"],
        domain=op_domain,
        quant_info=libpymo.PtrToInt64(quant_info),
    )
    model = create_model_from_node_fp16(quant_node, input_shape)
    return build_session(model, available_providers)


class TestQcQuantizeOp:
    def test_update_stats_with_pymo(self):
        input_arr = np.random.rand(1, 3, 4, 4).astype(np.float32)

        tensor_quantizer = create_tensor_quantizer(
            [], 8, quant_scheme=QuantScheme.post_training_tf
        )
        quant_info = create_quant_info(
            tensor_quantizer, OpMode.updateStats, useSymmetricEncoding=False
        )
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        session.run(None, {"input": input_arr})
        encodings = tensor_quantizer.computeEncodings(quant_info.useSymmetricEncoding)[
            0
        ]
        print(
            "Encoding returned: min={}, max={}, offset={}. delta={}, bw={}".format(
                encodings.min,
                encodings.max,
                encodings.offset,
                encodings.delta,
                encodings.bw,
            )
        )
        assert encodings is not None
        tensor_quantizer.setEncodings([encodings])
        assert quant_info.tensorQuantizerRef.isEncodingValid

    def test_quantize_dequantize_with_pymo(self):
        input_arr = np.asarray([[[[-7, -5, -3, 0, 0.1, 2.5]]]]).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )

        session.run(None, {"input": input_arr})
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 1
        encodings.min = -5.0
        encodings.delta = (1 + 5) / 255.0
        encodings.offset = -5.0 / encodings.delta

        qc_op.load_encodings([encodings])

        output = session.run(None, {"input": input_arr})[0]

        assert np.max(output) <= 1.1
        assert np.min(output) >= -5.1

    def test_quantize_dequantize_fp16(self):
        input_arr = np.asarray([[[[-7, -5, -3, 0, 0.1, 2.5]]]]).astype(np.float32)
        intermediate_output = input_arr.astype(np.float16)
        fp32_array = intermediate_output.astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )
        qc_op.data_type = QuantizationDataType.float

        qc_op.op_mode = OpMode.quantizeDequantize
        output = session.run(None, {"input": input_arr})[0]

        assert np.allclose(output, fp32_array)

    def test_update_stats_quantize_dequantize(self):
        input_arr = np.asarray([[[[-7, -5, -3, 0, 0.1, 2.5]]]]).astype(np.float32)
        input_arr2 = np.random.randn(*input_arr.shape).astype(np.float32) * 10
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.updateStats,
            bitwidth=8,
            use_symmetric_encodings=False,
        )

        session.run(None, {"input": input_arr})[0]
        qc_op.compute_encodings()
        assert math.isclose(qc_op.get_encodings()[0].max, 2.5, rel_tol=1e-2)
        assert math.isclose(qc_op.get_encodings()[0].min, -7, rel_tol=1e-2)

        qc_op.op_mode = OpMode.quantizeDequantize
        output = session.run(None, {"input": input_arr2})[0]
        assert np.max(output) <= 2.6
        assert np.min(output) >= -7.1
        assert not np.allclose(output, input_arr2)

    @pytest.mark.parametrize(
        "bitwidth, symmetric, expected_min, expected_max",
        [
            (2, True, -10.5, 5.25),
            (2, False, -14.0, 7.0),
            (3, True, -14.0, 10.5),
            (3, False, -12.0, 9.0),
        ],
    )
    def test_update_stats_low_bw(self, bitwidth, symmetric, expected_min, expected_max):
        input_arr = np.asarray([[[[-10.5, 10.5]]]]).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.updateStats,
            bitwidth=bitwidth,
            use_symmetric_encodings=symmetric,
        )

        session.run(None, {"input": input_arr})[0]
        qc_op.compute_encodings()
        assert qc_op.get_encodings()[0].max == expected_max
        assert qc_op.get_encodings()[0].min == expected_min

    def test_compare_one_shot_with_pymo(self):
        input_arr = np.random.randn(2, 3, 5, 1).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )

        quantizer = create_tensor_quantizer(
            [], 8, quant_scheme=QuantScheme.post_training_tf
        )
        out_tensor = np.zeros(input_arr.shape).astype(np.float32)
        # Perform one-shot quant-dequant in python
        quantizer.updateStats(input_arr)
        enc = quantizer.computeEncodings(False)[0]
        out_tensor = (
            np.round(np.clip(input_arr / enc.delta - enc.offset, 0, 255)) + enc.offset
        ) * enc.delta

        output = session.run(None, {"input": input_arr})[0]
        assert quant_info.encoding[0].max == enc.max
        assert quant_info.encoding[0].min == enc.min
        assert np.allclose(output, out_tensor)

    def test_one_shot_quantize_dequantize_asymmetric_cpu(self):
        input_arr = np.asarray([[[[-7, -5, -3, 0, 0.1, 2.5]]]]).astype(np.float32)

        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )

        output_oneshot = session.run(None, {"input": input_arr})[0]

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 2.5
        encodings.min = -7
        encodings.offset = -188
        encodings.delta = (7 + 2.5) / 255
        qc_op.load_encodings([encodings])

        output_qdq = session.run(None, {"input": input_arr})

        assert np.allclose(output_oneshot, output_qdq)

    def test_one_shot_quantize_dequantize_symmetric_signed_cpu(self):
        input_arr = np.asarray([[[[-7, -5, -3, 0, 0.1, 2.5]]]]).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=True,
        )
        output_oneshot = session.run(None, {"input": input_arr})

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 7 * 127 / 128
        encodings.min = -7
        encodings.offset = -128
        encodings.delta = 7 / 128
        qc_op.load_encodings([encodings])

        output_qdq = session.run(None, {"input": input_arr})

        assert np.allclose(output_oneshot, output_qdq)

    def test_one_shot_quantize_dequantize_symmetric_unsigned_cpu(self):
        input_arr = np.asarray([[[[0, 1.2, 1.5, 4.0, 4.9, 5.3]]]]).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=True,
        )

        qc_op.use_unsigned_symmetric = True

        output_oneshot = session.run(None, {"input": input_arr})

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.3
        encodings.min = 0.0
        encodings.offset = 0
        encodings.delta = 5.3 / 255
        qc_op.load_encodings([encodings])

        output_qdq = session.run(None, {"input": input_arr})

        assert np.allclose(output_oneshot, output_qdq)

    @pytest.mark.cuda
    def test_one_shot_quantize_dequantize_cpu_vs_gpu(self):
        input_arr = np.asarray([[[[0, 1.2, 1.5, 4.0, 4.9, 5.3]]]]).astype(np.float32)
        quant_info_cpu = libquant_info.QcQuantizeInfo()
        quant_node_cpu = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain="aimet.customop.cpu",
            quant_info=libpymo.PtrToInt64(quant_info_cpu),
        )
        model_cpu = create_model_from_node(quant_node_cpu, input_arr.shape)
        session_cpu = build_session(model_cpu, available_providers)
        qc_op_cpu = QcQuantizeOp(
            quant_info=quant_info_cpu,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=True,
        )

        output_cpu = session_cpu.run(None, {"input": input_arr})

        quant_info_gpu = libquant_info.QcQuantizeInfo()
        quant_node_gpu = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain="aimet.customop.cuda",
            quant_info=libpymo.PtrToInt64(quant_info_gpu),
        )
        model_gpu = create_model_from_node(quant_node_gpu, input_arr.shape)
        session_gpu = build_session(model_gpu, available_providers)
        qc_op_gpu = QcQuantizeOp(
            quant_info=quant_info_gpu,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=True,
        )

        output_gpu = session_gpu.run(None, {"input": input_arr})

        assert np.all(output_gpu[0] == output_cpu[0])

    def test_set_get_properties(self):
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=True,
        )
        qc_op.use_strict_symmetric = True
        assert quant_info.tensorQuantizerRef.getStrictSymmetric() == True

        qc_op.use_unsigned_symmetric = False
        assert quant_info.tensorQuantizerRef.getUnsignedSymmetric() == False

        qc_op.use_unsigned_symmetric = True
        assert quant_info.tensorQuantizerRef.getUnsignedSymmetric() == True

        qc_op.data_type = QuantizationDataType.float
        assert qc_op.data_type == QuantizationDataType.float
        assert qc_op.quant_info.isIntDataType == False

    @pytest.mark.parametrize("quant_axis", [0, 1])
    @pytest.mark.parametrize(
        "use_symmetric,strict_symmetric,unsigned_symmetric",
        [(True, True, False), (True, False, True), (False, False, False)],
    )
    def test_per_channel_one_shot_quantize_dequantize(
        self, use_symmetric, strict_symmetric, unsigned_symmetric, quant_axis
    ):
        """
        Compares the output of per-channel quantization to the output of each channel passing through
        a per-tensor quantizer.
        """
        input_shape = (12, 6, 3, 3)
        input_arr = np.random.randn(
            *input_shape,
        ).astype(np.float32)
        expected_output_arr = []

        tensor_params = TensorQuantizerParams(input_shape, quant_axis, None)
        quant_info = libquant_info.QcQuantizeInfo()
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=use_symmetric,
            tensor_quantizer_params=tensor_params,
        )

        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        per_tensor_model = create_model_from_node(
            quant_node, input_arr.take(indices=0, axis=quant_axis).shape
        )
        session = build_session(per_tensor_model, available_providers)
        # Run each channel through a per-tensor quantizer
        for idx in range(input_shape[quant_axis]):
            channel_input = input_arr.take(indices=idx, axis=quant_axis)
            output = session.run(None, {"input": channel_input})[0]
            expected_output_arr.append(np.expand_dims(output, quant_axis))
            quant_info.opMode = OpMode.oneShotQuantizeDequantize
        expected_output_arr = np.concatenate(expected_output_arr, axis=quant_axis)

        qc_op.enable_per_channel_quantization()
        per_channel_quant_node = helper.make_node(
            per_channel_op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        per_channel_model = create_model_from_node(
            per_channel_quant_node, input_arr.shape
        )
        # Run the entire tensor through the per-channel quantizer
        session = build_session(per_channel_model, available_providers)
        output_per_channel = session.run(None, {"input": input_arr})[0]
        assert np.allclose(output_per_channel, expected_output_arr)

    def test_per_channel_quantize_dequantize(self):
        inp_array = np.array(
            [
                [-7, -5, -3, 0, 0.1, 2.5],
                [-7, -5, -3, 0, 0.1, 2.5],
                [-7, -5, -3, 0, 0.1, 2.5],
                [-7, -5, -3, 0, 0.1, 2.5],
            ],
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
        tensor_quantizer = create_tensor_quantizer(
            inp_array.shape,
            encodings[0].bw,
            ch_axis=0,
            quant_scheme=QuantScheme.post_training_tf,
        )
        tensor_quantizer.setEncodings(encodings)
        quant_info = create_quant_info(
            tensor_quantizer, OpMode.quantizeDequantize, useSymmetricEncoding=True
        )
        per_channel_quant_node = helper.make_node(
            per_channel_op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )

        per_channel_model = create_model_from_node(
            per_channel_quant_node, inp_array.shape
        )
        per_channel_session = build_session(per_channel_model, available_providers)

        expected_out = np.array(
            [
                [-3.84, -3.84, -3, 0, 0.089999996, 2.49],
                [-3.84, -3.84, -3, 0, 0.089999996, 2.49],
                [-3.84, -3.84, -3, 0, 0.089999996, 2.49],
                [-6.4, -5, -3, 0, 0.1, 2.5],
            ],
        ).astype(np.float32)
        output = per_channel_session.run(None, {"input": inp_array})[0]
        assert np.allclose(output, expected_out)

    def test_quantize_dequantize_large_tensor(self):
        """
        Verify correctness with a tensor large enough to activate threadpool parallelism
        (> 1024 elements) and with a shape not evenly divisible by 1024.
        Tests per-tensor, per-channel, and per-block QDQ through the ORT session and
        compares against the Python QDQ path.
        """
        input_shape = (7, 500)
        tensor_quantizer_params = TensorQuantizerParams(input_shape, 0, 1)
        calibration_tensor = np.random.randn(*input_shape).astype(np.float32)
        input_tensor = np.random.randn(*input_shape).astype(np.float32) * 10

        quant_info = libquant_info.QcQuantizeInfo()
        session = create_qc_quantize_model_session(quant_info, input_tensor.shape)

        quantizer = QcQuantizeOp(
            quant_info,
            bitwidth=8,
            op_mode=OpMode.updateStats,
            tensor_quantizer_params=tensor_quantizer_params,
        )

        # per-tensor
        quantizer.update_encoding_stats(calibration_tensor)
        quantizer.compute_encodings()
        quantizer.op_mode = OpMode.quantizeDequantize
        output = session.run(None, {"input": input_tensor})[0]
        qdq_output = quantizer.quantize_dequantize(input_tensor)
        assert np.array_equal(output, qdq_output)

        # per-channel
        quantizer.reset_encoding_stats()
        quantizer.enable_per_channel_quantization()
        quantizer.update_encoding_stats(calibration_tensor)
        quantizer.compute_encodings()
        quantizer.op_mode = OpMode.quantizeDequantize
        output = session.run(None, {"input": input_tensor})[0]
        qdq_output = quantizer.quantize_dequantize(input_tensor)
        assert np.array_equal(output, qdq_output)

        # per-block
        quantizer.reset_encoding_stats()
        quantizer._enable_blockwise_quantization(block_size=25)
        quantizer.update_encoding_stats(calibration_tensor)
        quantizer.compute_encodings()
        quantizer.op_mode = OpMode.quantizeDequantize
        output = session.run(None, {"input": input_tensor})[0]
        qdq_output = quantizer.quantize_dequantize(input_tensor)
        assert np.array_equal(output, qdq_output)

    @pytest.mark.parametrize(
        "input_arr",
        (
            np.asarray([0, FLOAT32_MIN]).astype(np.float32),
            np.asarray([0, FLOAT32_MAX]).astype(np.float32),
            np.asarray([0, FLOAT32_MIN, FLOAT32_MAX]).astype(np.float32),
        ),
    )
    @pytest.mark.parametrize(
        "quant_scheme",
        (QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced),
    )
    @pytest.mark.parametrize("symmetric", (True, False))
    @pytest.mark.parametrize("bitwidth", [2, 4, 8, 16])
    def test_update_stats_extreme_values(
        self, quant_scheme, input_arr, symmetric, bitwidth
    ):
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=quant_scheme,
            rounding_mode="nearest",
            op_mode=OpMode.updateStats,
            bitwidth=bitwidth,
            use_symmetric_encodings=symmetric,
        )

        session.run(None, {"input": input_arr})
        qc_op.compute_encodings()

        max = np.array(qc_op.get_encodings()[0].max, dtype=np.float32)
        min = np.array(qc_op.get_encodings()[0].min, dtype=np.float32)
        delta = np.array(qc_op.get_encodings()[0].delta, dtype=np.float32)
        offset = np.array(qc_op.get_encodings()[0].offset, dtype=np.float32)
        num_steps = np.array(2 ** qc_op.get_encodings()[0].bw - 1, dtype=np.float32)

        assert FLOAT32_MIN <= min <= 0
        assert FLOAT32_MIN <= delta * offset <= 0
        assert np.allclose(min, delta * offset)
        assert 0 <= max <= FLOAT32_MAX
        assert 0 <= delta * (offset + num_steps) <= FLOAT32_MAX
        assert np.allclose(max, delta * (offset + num_steps))

    def test_merge_constraints(self):
        """
        Given:
          - q1: Symmetric quantizer
          - q2: Quantizer with fixed range [x, y]
        When: _merge_constraints
        Then: Resulting quantizer should be a symmetric quantizer with range[-z, z]
              (z = max(abs(x), abs(y)))
        """
        q1 = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=16,
            use_symmetric_encodings=True,
        )
        q2 = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )
        q2._encoding_min_max_fixed_vals = (-2, 1)
        q1._merge_constraints(q2)

        assert q1.bitwidth == 8
        assert q1.use_symmetric_encodings
        assert q1._encoding_min_max_fixed_vals == (-2, 2)

        """
        Given: Quantizers with different granularity
        When: _merge_constraints
        Then: Throw runtime error
        """
        per_tensor_qtzr = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )
        per_channel_qtzr = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
            tensor_quantizer_params=TensorQuantizerParams(
                tensor_shape=(10, 10),
                channel_axis=0,
            ),
        )
        per_channel_qtzr.enable_per_channel_quantization(True)
        with pytest.raises(RuntimeError):
            per_tensor_qtzr._merge_constraints(per_channel_qtzr)

        blockwise_qtzr = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
            tensor_quantizer_params=TensorQuantizerParams(
                tensor_shape=(10, 10),
                channel_axis=0,
                block_axis=1,
            ),
        )
        blockwise_qtzr.enable_per_channel_quantization(True)
        blockwise_qtzr._enable_blockwise_quantization(block_size=5)
        with pytest.raises(RuntimeError):
            per_channel_qtzr._merge_constraints(blockwise_qtzr)

        """
        Given: Quantizers with different channel/block axis
        When: _merge_constraints
        Then: Throw runtime error
        """
        per_channel_qtzr_ = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
            tensor_quantizer_params=TensorQuantizerParams(
                tensor_shape=(10, 10),
                channel_axis=1,
            ),
        )
        per_channel_qtzr_.enable_per_channel_quantization(True)
        with pytest.raises(RuntimeError):
            per_channel_qtzr._merge_constraints(per_channel_qtzr_)

        blockwise_qtzr_ = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
            tensor_quantizer_params=TensorQuantizerParams(
                tensor_shape=(10, 10),
                channel_axis=1,
                block_axis=0,
            ),
        )
        blockwise_qtzr_.enable_per_channel_quantization(True)
        blockwise_qtzr_._enable_blockwise_quantization(block_size=5)
        with pytest.raises(RuntimeError):
            blockwise_qtzr_._merge_constraints(blockwise_qtzr)

        """
        Given: Quantizers with different block size
        When: _merge_constraints
        Then: Throw runtime error
        """
        blockwise_qtzr__ = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
            tensor_quantizer_params=TensorQuantizerParams(
                tensor_shape=(10, 10),
                channel_axis=1,
                block_axis=0,
            ),
        )
        blockwise_qtzr__.enable_per_channel_quantization(True)
        blockwise_qtzr__._enable_blockwise_quantization(block_size=2)
        with pytest.raises(RuntimeError):
            blockwise_qtzr__._merge_constraints(blockwise_qtzr_)

        """
        Given: Quantizers with different fixed output range
        When: _merge_constraints
        Then: Throw runtime error
        """
        q1 = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )
        q1._encoding_min_max_fixed_vals = (-1, 1)
        q2 = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )
        q2._encoding_min_max_fixed_vals = (-2, 1)
        with pytest.raises(RuntimeError):
            q1._merge_constraints(q2)

    @pytest.mark.parametrize("dtype", (np.float32, np.float16))
    @pytest.mark.parametrize("contiguous", (True, False))
    def test_quantize_dequantize(self, contiguous: bool, dtype: np.dtype):
        tensor_quantizer_params = TensorQuantizerParams((10, 15), 0, 1)
        calibration_tensor = np.random.randn(10, 15).astype(dtype)
        input_tensor = np.random.randn(*calibration_tensor.shape).astype(dtype) * 10
        if not contiguous:
            input_tensor = input_tensor.T.copy()
            input_tensor = input_tensor.T
            assert not input_tensor.flags["C_CONTIGUOUS"]

        quant_info = libquant_info.QcQuantizeInfo()
        session = create_qc_quantize_model_session(
            quant_info,
            input_tensor.shape,
            float_dtype=onnx.helper.np_dtype_to_tensor_dtype(input_tensor.dtype),
        )

        quantizer = QcQuantizeOp(
            quant_info,
            bitwidth=4,
            op_mode=OpMode.updateStats,
            tensor_quantizer_params=tensor_quantizer_params,
        )
        # per-tensor
        quantizer.update_encoding_stats(calibration_tensor)
        quantizer.compute_encodings()
        quantizer.op_mode = OpMode.quantizeDequantize
        output = session.run(None, {"input": input_tensor})[0]
        qdq_output = quantizer.quantize_dequantize(input_tensor)
        assert np.array_equal(output, qdq_output)
        assert output.dtype == input_tensor.dtype

        # per-channel
        quantizer.reset_encoding_stats()
        quantizer.enable_per_channel_quantization()
        quantizer.update_encoding_stats(input_tensor)
        quantizer.compute_encodings()
        quantizer.op_mode = OpMode.quantizeDequantize
        output = session.run(None, {"input": input_tensor})[0]
        qdq_output = quantizer.quantize_dequantize(input_tensor)
        assert np.array_equal(output, qdq_output)
        assert output.dtype == input_tensor.dtype

        # per-block
        quantizer.reset_encoding_stats()
        quantizer._enable_blockwise_quantization(block_size=3)
        quantizer.update_encoding_stats(input_tensor)
        quantizer.compute_encodings()
        quantizer.op_mode = OpMode.quantizeDequantize
        output = session.run(None, {"input": input_tensor})[0]
        qdq_output = quantizer.quantize_dequantize(input_tensor)
        assert np.array_equal(output, qdq_output)
        assert output.dtype == input_tensor.dtype

    def test_merge_constraints(self):
        """
        Given:
          - q1: Symmetric quantizer
          - q2: Quantizer with fixed range [x, y]
        When: _merge_constraints
        Then: Resulting quantizer should be a symmetric quantizer with range[-z, z]
              (z = max(abs(x), abs(y)))
        """
        q1 = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=16,
            use_symmetric_encodings=True,
        )
        q2 = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )
        q2._encoding_min_max_fixed_vals = (-2, 1)
        q1._merge_constraints(q2)

        assert q1.bitwidth == 8
        assert q1.use_symmetric_encodings
        assert q1._encoding_min_max_fixed_vals == (-2, 2)

        """
        Given: Quantizers with different granularity
        When: _merge_constraints
        Then: Throw runtime error
        """
        per_tensor_qtzr = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )
        per_channel_qtzr = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
            tensor_quantizer_params=TensorQuantizerParams(
                tensor_shape=(10, 10),
                channel_axis=0,
            ),
        )
        per_channel_qtzr.enable_per_channel_quantization(True)
        with pytest.raises(RuntimeError):
            per_tensor_qtzr._merge_constraints(per_channel_qtzr)

        blockwise_qtzr = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
            tensor_quantizer_params=TensorQuantizerParams(
                tensor_shape=(10, 10),
                channel_axis=0,
                block_axis=1,
            ),
        )
        blockwise_qtzr.enable_per_channel_quantization(True)
        blockwise_qtzr._enable_blockwise_quantization(block_size=5)
        with pytest.raises(RuntimeError):
            per_channel_qtzr._merge_constraints(blockwise_qtzr)

        """
        Given: Quantizers with different channel/block axis
        When: _merge_constraints
        Then: Throw runtime error
        """
        per_channel_qtzr_ = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
            tensor_quantizer_params=TensorQuantizerParams(
                tensor_shape=(10, 10),
                channel_axis=1,
            ),
        )
        per_channel_qtzr_.enable_per_channel_quantization(True)
        with pytest.raises(RuntimeError):
            per_channel_qtzr._merge_constraints(per_channel_qtzr_)

        blockwise_qtzr_ = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
            tensor_quantizer_params=TensorQuantizerParams(
                tensor_shape=(10, 10),
                channel_axis=1,
                block_axis=0,
            ),
        )
        blockwise_qtzr_.enable_per_channel_quantization(True)
        blockwise_qtzr_._enable_blockwise_quantization(block_size=5)
        with pytest.raises(RuntimeError):
            blockwise_qtzr_._merge_constraints(blockwise_qtzr)

        """
        Given: Quantizers with different block size
        When: _merge_constraints
        Then: Throw runtime error
        """
        blockwise_qtzr__ = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
            tensor_quantizer_params=TensorQuantizerParams(
                tensor_shape=(10, 10),
                channel_axis=1,
                block_axis=0,
            ),
        )
        blockwise_qtzr__.enable_per_channel_quantization(True)
        blockwise_qtzr__._enable_blockwise_quantization(block_size=2)
        with pytest.raises(RuntimeError):
            blockwise_qtzr__._merge_constraints(blockwise_qtzr_)

        """
        Given: Quantizers with different fixed output range
        When: _merge_constraints
        Then: Throw runtime error
        """
        q1 = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )
        q1._encoding_min_max_fixed_vals = (-1, 1)
        q2 = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )
        q2._encoding_min_max_fixed_vals = (-2, 1)
        with pytest.raises(RuntimeError):
            q1._merge_constraints(q2)

    @pytest.mark.parametrize("np_dtype, tp_dtype", HALF_FLOAT_DTYPES)
    def test_quantize_dequantize_with_pymo_half_float(self, np_dtype, tp_dtype):
        input_arr = np.asarray([[[[-7, -5, -3, 0, 0.1, 2.5]]]]).astype(np_dtype)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(
            quant_node, input_arr.shape, float_dtype=tp_dtype
        )
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=False,
        )

        run_session(session, {"input": input_arr})
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 1
        encodings.min = -5.0
        encodings.delta = (1 + 5) / 255.0
        encodings.offset = -5.0 / encodings.delta

        qc_op.load_encodings([encodings])

        (output,) = run_session(session, {"input": input_arr})

        assert np.max(output) <= 1.1
        assert np.min(output) >= -5.1

    @pytest.mark.parametrize("np_dtype, tp_dtype", HALF_FLOAT_DTYPES)
    def test_update_stats_quantize_dequantize_half_float(self, np_dtype, tp_dtype):
        input_arr = np.asarray([[[[-7, -5, -3, 0, 0.1, 2.5]]]]).astype(np_dtype)
        input_arr2 = (np.random.randn(*input_arr.shape) * 10).astype(np_dtype)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(
            quant_node, input_arr.shape, float_dtype=tp_dtype
        )
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            rounding_mode="nearest",
            op_mode=OpMode.updateStats,
            bitwidth=8,
            use_symmetric_encodings=False,
        )

        run_session(session, {"input": input_arr})
        qc_op.compute_encodings()
        assert math.isclose(qc_op.get_encodings()[0].max, 2.5, rel_tol=1e-2)
        assert math.isclose(qc_op.get_encodings()[0].min, -7, rel_tol=1e-2)

        qc_op.op_mode = OpMode.quantizeDequantize
        (output,) = run_session(session, {"input": input_arr2})
        assert np.max(output) <= 2.6
        assert np.min(output) >= -7.1
        assert not np.allclose(output.astype(np.float32), input_arr2.astype(np.float32))

    @pytest.mark.parametrize("contiguous", (True, False))
    @pytest.mark.parametrize("np_dtype, tp_dtype", HALF_FLOAT_DTYPES)
    def test_quantize_dequantize_half_float_model(self, contiguous, np_dtype, tp_dtype):
        np.random.seed(0)
        tensor_quantizer_params = TensorQuantizerParams((10, 15), 0, 1)
        calibration_tensor = np.random.randn(10, 15).astype(np_dtype)
        input_tensor = (np.random.randn(*calibration_tensor.shape) * 10).astype(
            np_dtype
        )
        if not contiguous:
            input_tensor = input_tensor.T.copy()
            input_tensor = input_tensor.T
            assert not input_tensor.flags["C_CONTIGUOUS"]

        quant_info = libquant_info.QcQuantizeInfo()
        session = create_qc_quantize_model_session(
            quant_info, input_tensor.shape, float_dtype=tp_dtype
        )

        quantizer = QcQuantizeOp(
            quant_info,
            bitwidth=4,
            op_mode=OpMode.updateStats,
            tensor_quantizer_params=tensor_quantizer_params,
        )
        # per-tensor
        quantizer.update_encoding_stats(calibration_tensor)
        quantizer.compute_encodings()
        quantizer.op_mode = OpMode.quantizeDequantize
        (output,) = run_session(session, {"input": input_tensor})
        qdq_output = quantizer.quantize_dequantize(input_tensor)
        assert np.array_equal(output, qdq_output.astype(np_dtype))

        # per-channel
        quantizer.reset_encoding_stats()
        quantizer.enable_per_channel_quantization()
        quantizer.update_encoding_stats(input_tensor)
        quantizer.compute_encodings()
        quantizer.op_mode = OpMode.quantizeDequantize
        (output,) = run_session(session, {"input": input_tensor})
        qdq_output = quantizer.quantize_dequantize(input_tensor)
        assert np.array_equal(output, qdq_output.astype(np_dtype))

        # per-block
        quantizer.reset_encoding_stats()
        quantizer._enable_blockwise_quantization(block_size=3)
        quantizer.update_encoding_stats(input_tensor)
        quantizer.compute_encodings()
        quantizer.op_mode = OpMode.quantizeDequantize
        (output,) = run_session(session, {"input": input_tensor})
        qdq_output = quantizer.quantize_dequantize(input_tensor)
        assert np.array_equal(output, qdq_output.astype(np_dtype))

    @pytest.mark.parametrize(
        "quant_scheme",
        [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced],
    )
    @pytest.mark.skip_on_windows_arm64(
        "#6364: divide-by-zero error on Windows ARM64 builds"
    )
    def test_compute_encodings_with_size_zero_tensor(self, quant_scheme):
        """
        When: quantizer is updated with a size zero tensor
        Then: compute_encoding does not throw error and encodings are computed based on previous stats
        """
        quantizer = create_tensor_quantizer((), quant_scheme=quant_scheme)
        quant_info = create_quant_info(quantizer, OpMode.updateStats)
        session = create_qc_quantize_model_session(quant_info, ("x", 100))
        session.run(None, {"input": np.random.randn(1, 100).astype(np.float32)})
        session.run(None, {"input": np.random.randn(0, 100).astype(np.float32)})
        (encoding,) = quantizer.computeEncodings(False)
        assert encoding.min < 0 and encoding.min > -100
        assert encoding.max > 0 and encoding.max < 100
        assert encoding.delta > 0 and encoding.delta < 1

    @pytest.mark.parametrize("quant_scheme", [QuantScheme.min_max])
    @pytest.mark.parametrize("symmetric", [False, True])
    @pytest.mark.parametrize("zero_point_shift", [0.0, 0.5])
    @pytest.mark.parametrize("bitwidth", [4, 8, 16, 32])
    def test_minimum_scale(
        self,
        bitwidth: int,
        symmetric: bool,
        zero_point_shift,
        quant_scheme: QuantScheme,
    ):
        if not symmetric and zero_point_shift:
            pytest.skip("zero_point_shift is not applicable for symmetric quantization")

        input = np.zeros((1, 10)).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.isIntDataType = True
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=op_domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=quant_scheme,
            op_mode=OpMode.updateStats,
            bitwidth=bitwidth,
            use_symmetric_encodings=symmetric,
        )
        qc_op.set_zero_point_shift(zero_point_shift)

        session.run(None, {"input": input})
        qc_op.compute_encodings()
        (enc,) = qc_op.get_encodings()

        assert enc.delta == _get_minimum_scale(2**bitwidth - 1)
        assert enc.offset % 1 == zero_point_shift
        _assert_encoding_coherence(enc)


def _assert_encoding_coherence(encoding: libpymo.TfEncoding):
    num_steps = 2**encoding.bw - 1
    assert np.isfinite(encoding.delta) and encoding.delta > 0
    assert np.isfinite(encoding.offset)
    assert np.isfinite(encoding.min) and encoding.min <= 0
    assert np.isfinite(encoding.max) and encoding.max >= 0
    assert np.allclose(encoding.max, encoding.delta * (encoding.offset + num_steps))
    assert np.allclose(encoding.min, encoding.delta * (encoding.offset))


blockwise_qdq_test_1 = {
    "input_shape": (2, 3, 4),
    "block_axis": 0,
    "block_size": 1,
    "channel_axis": 1,
    "bitwidth": 8,
    "min": [0, 0, 0, -2, -2.5, 0],
    "max": [255.0 * 0.25, 255.0, 127.5, 508.0, 245.0 * 0.25, 2550.0],
    "in_tensor": [
        0.126,
        10.4,
        -12.3,
        10000,
        0.126,
        10.4,
        -12.3,
        10000,
        0.126,
        10.4,
        -12.3,
        10000,
        0.126,
        10.4,
        -12.3,
        10000,
        0.126,
        10.4,
        -12.3,
        10000,
        0.126,
        10.4,
        -12.3,
        10000,
    ],
    "expected": [
        0.25,
        10.5,
        0,
        63.75,  # scale = .25
        0.0,
        10.0,
        0.0,
        255.0,  # scale = 1
        0.0,
        10.5,
        0.0,
        127.5,  # scale = 0.5
        0.0,
        10.0,
        -2.0,
        508.0,  # scale = 2. offset=-1
        0.25,
        10.5,
        -2.5,
        61.25,  # scale = .25
        0.0,
        10.0,
        0,
        2550.0,  # scale = 10
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
        -125.1,
        -125.1,
        48.3,
        48.3,
        68.3,
        68.3,
        -3.1,
        -3.1,
        -125.1,
        -125.1,
        48.3,
        48.3,
        68.3,
        68.3,
        -3.1,
        -3.1,
    ],
    "expected": [
        -64.0,
        -125.0,
        48.5,
        48.0,
        63.5,
        68.0,
        -3.0,
        -3.0,
        -126.0,
        -124.0,
        48.0,
        48.0,
        68.0,
        68.0,
        -4.0,
        -4.0,
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
        40.23,
        0.0321,  # Scale = 0.01
        -40.23,
        -0.0321,  # Scale = 0.1
        23.44,
        -2.3111,  # scale = 1
        23.44,
        -2.3111,  # scale = 10
        -1000.1,
        334,  # scale = 0.01
        23.1111,
        -23.1111,  # scale = 0.1
        23.1111,
        -23.1111,  # scale = 1
        -1,
        100000,  # scale = 10
    ],
    "expected": [
        1.27,
        0.03,  # Scale = 0.01
        -12.8,
        0.0,  # Scale = 0.1
        23,
        -2,  # scale = 1
        20.0,
        0,  # scale = 10
        0,
        2.55,  # scale = 0.01
        23.1,
        0.0,  # scale = 0.1
        23,
        0.0,  # scale = 1
        0,
        2550,  # scale = 10
    ],
}


def isclose(x1, x2, atol=1e-4):
    return abs(x1 - x2) <= atol


class TestBlockwiseQuantizeOp:
    @pytest.mark.parametrize(
        "test_set", (blockwise_qdq_test_1, blockwise_qdq_test_2, blockwise_qdq_test_3)
    )
    def test_blockwise_quantize_dequantize(self, test_set):
        input_shape = test_set["input_shape"]
        block_axis = test_set["block_axis"]
        block_size = test_set["block_size"]
        channel_axis = test_set["channel_axis"]
        in_tensor = np.array(test_set["in_tensor"], dtype=np.float32).reshape(
            input_shape
        )
        expected_output = np.array(test_set["expected"], dtype=np.float32).reshape(
            input_shape
        )
        encoding_min = test_set["min"]
        encoding_max = test_set["max"]

        encodings = create_encoding(encoding_min, encoding_max, 8, False)

        tensor_quantizer = create_tensor_quantizer(
            input_shape,
            test_set["bitwidth"],
            channel_axis,
            block_axis,
            block_size,
            quant_scheme=QuantScheme.post_training_tf,
        )
        tensor_quantizer.setEncodings(encodings)

        quant_info = create_quant_info(
            tensor_quantizer, OpMode.quantizeDequantize, useSymmetricEncoding=True
        )

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

        input_tensor = (
            np.asarray([-5.4, 10, -2, 3.5, 23.1, 2.0, -10, -2, -1, -0.1, 0.3, 0.1])
            .astype(np.float32)
            .reshape(input_shape)
        )

        tensor_quantizer = create_tensor_quantizer(
            input_shape,
            bitwidth,
            channel_axis,
            block_axis,
            block_size,
            quant_scheme=QuantScheme.post_training_tf,
        )
        quant_info = create_quant_info(
            tensor_quantizer, OpMode.updateStats, useSymmetricEncoding=symmetric
        )
        session = create_qc_quantize_model_session(quant_info, input_shape)

        # Run calibration
        output_tensor = session.run(None, {"input": input_tensor})[0]

        # Compute encodings
        encodings = tensor_quantizer.computeEncodings(symmetric)

        # Op should be passthrough in update_stats mode
        assert np.all(input_tensor == output_tensor)

        # Computed encodings should be symmetric and correspond to the absolute min/max in the block
        expected_max = np.max(np.abs(input_tensor.reshape(4, 3)), axis=1)
        for idx, enc in enumerate(encodings):
            assert isclose(enc.max, expected_max[idx]) or isclose(
                -enc.min, expected_max[idx]
            )
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

        input_tensor = (
            np.asarray([-5.4, 10, -2, 3.5, 23.1, 2.0, -10, -2, -1, -0.1, 0.3, 0.1])
            .astype(np.float32)
            .reshape(input_shape)
        )

        tensor_quantizer = create_tensor_quantizer(
            input_shape,
            bitwidth,
            channel_axis,
            block_axis,
            block_size,
            quant_scheme=QuantScheme.post_training_tf,
        )
        quant_info = create_quant_info(
            tensor_quantizer, OpMode.updateStats, useSymmetricEncoding=symmetric
        )
        session = create_qc_quantize_model_session(quant_info, input_shape)

        # Run calibration
        output_tensor = session.run(None, {"input": input_tensor})[0]

        # Compute encodings
        encodings = tensor_quantizer.computeEncodings(symmetric)

        # Op should be passthrough in update_stats mode
        assert np.all(input_tensor == output_tensor)

        # Computed encodings should be symmetric and correspond to the absolute min/max in the block
        expected_max = np.maximum(
            np.max(input_tensor.reshape(3, 2, 2), axis=1), 0
        ).flatten()
        expected_min = np.minimum(
            np.min(input_tensor.reshape(3, 2, 2), axis=1), 0
        ).flatten()
        for idx, enc in enumerate(encodings):
            assert isclose(enc.max, expected_max[idx], atol=enc.delta)
            assert isclose(enc.min, expected_min[idx], atol=enc.delta)
            assert isclose(enc.delta, (enc.max - enc.min) / (2**bitwidth - 1))
            assert isclose(enc.offset, enc.min / enc.delta)

    def test_blockwise_one_shot_compute_encodings(self):
        input_shape = (2, 6)
        block_axis = 1
        block_size = 3
        channel_axis = 0
        bitwidth = 8
        symmetric = True

        input_tensor = (
            np.asarray([-5.4, 10, -2, 3.5, 23.1, 2.0, -10, -2, -1, -0.1, 0.3, 0.1])
            .astype(np.float32)
            .reshape(input_shape)
        )

        tensor_quantizer = create_tensor_quantizer(
            input_shape,
            bitwidth,
            channel_axis,
            block_axis,
            block_size,
            quant_scheme=QuantScheme.post_training_tf,
        )
        quant_info = create_quant_info(
            tensor_quantizer,
            OpMode.oneShotQuantizeDequantize,
            useSymmetricEncoding=symmetric,
        )
        session = create_qc_quantize_model_session(quant_info, input_shape)

        # Run calibration
        output_tensor = session.run(None, {"input": input_tensor})[0]

        # Computed encodings should be symmetric and correspond to the absolute min/max in the block
        expected_max = np.max(np.abs(input_tensor.reshape(4, 3)), axis=1)
        cpp_encodings = quant_info.encoding
        for idx, enc in enumerate(cpp_encodings):
            assert isclose(enc.max, expected_max[idx]) or isclose(
                -enc.min, expected_max[idx]
            )
            assert isclose((enc.max + enc.min), -1 * enc.delta)
            assert enc.offset == -128
            assert isclose(enc.delta, enc.max / (2 ** (bitwidth - 1) - 1))

        # Compute the expected output given the computed encodings
        delta = (
            np.array([enc.delta for enc in cpp_encodings])
            .astype(np.float32)
            .reshape(-1, 1)
        )
        offset = (
            np.array([enc.offset for enc in cpp_encodings])
            .astype(np.float32)
            .reshape(-1, 1)
        )
        expected_out = (
            np.clip(
                np.round(input_tensor.reshape(4, 3) / delta - offset),
                0,
                2**bitwidth - 1,
            )
            + offset
        ) * delta

        # Op should produce the quantDequant output
        assert np.allclose(output_tensor, expected_out.reshape(output_tensor.shape))

    @pytest.mark.parametrize(
        "symmetric, bitwidth, delta, offset",
        [(True, 8, 0.1, -128), (False, 16, 0.0125, -1000)],
    )
    def test_export_per_tensor_int_encodings(self, symmetric, bitwidth, delta, offset):
        quant_info = libquant_info.QcQuantizeInfo()
        qc_quantize_op = QcQuantizeOp(
            quant_info,
            use_symmetric_encodings=symmetric,
            op_mode=OpMode.quantizeDequantize,
        )
        assert qc_quantize_op.export_encodings() is None
        encoding = libpymo.TfEncoding()
        encoding.min = delta * offset
        encoding.max = delta * (offset + 2**bitwidth - 1)
        encoding.bw = bitwidth
        encoding.offset = offset
        encoding.delta = delta
        qc_quantize_op.update_quantizer_and_load_encodings(
            [encoding], symmetric, False, False, QuantizationDataType.int
        )
        exported_encodings = qc_quantize_op.export_encodings("0.6.1")
        assert len(exported_encodings) == 1
        assert exported_encodings[0]["scale"] == delta
        assert exported_encodings[0]["offset"] == offset
        assert exported_encodings[0]["bitwidth"] == bitwidth
        assert exported_encodings[0]["dtype"] == "int"
        assert exported_encodings[0]["is_symmetric"] == str(symmetric)

        exported_encodings = qc_quantize_op.export_encodings("1.0.0")
        assert isinstance(exported_encodings, dict)
        assert exported_encodings.keys() == {
            "enc_type",
            "dtype",
            "bw",
            "is_sym",
            "scale",
            "offset",
        }
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

        qc_quantize_op.enabled = False
        assert qc_quantize_op.export_encodings("0.6.1") is None
        assert qc_quantize_op.export_encodings("1.0.0") is None

    @pytest.mark.parametrize(
        "symmetric, bitwidth, delta, offset",
        [
            (True, 8, 0.1, -128),
        ],
    )
    def test_export_per_channel_int_encodings(self, symmetric, bitwidth, delta, offset):
        channel_axis = 0
        block_axis = 1
        tensor_shape = [5, 8]
        params = TensorQuantizerParams(tensor_shape, channel_axis, block_axis)

        quant_info = libquant_info.QcQuantizeInfo()
        qc_quantize_op = QcQuantizeOp(
            quant_info,
            use_symmetric_encodings=symmetric,
            op_mode=OpMode.quantizeDequantize,
            tensor_quantizer_params=params,
        )
        qc_quantize_op.enable_per_channel_quantization()
        assert qc_quantize_op.export_encodings() is None
        encodings = [libpymo.TfEncoding() for _ in range(tensor_shape[channel_axis])]
        for encoding in encodings:
            encoding.min = delta * offset
            encoding.max = delta * (offset + 2**bitwidth - 1)
            encoding.bw = bitwidth
            encoding.offset = offset
            encoding.delta = delta
        qc_quantize_op.load_encodings(encodings)
        exported_encodings = qc_quantize_op.export_encodings("0.6.1")
        assert len(exported_encodings) == tensor_shape[channel_axis]

        exported_encodings = qc_quantize_op.export_encodings("1.0.0")
        assert exported_encodings.keys() == {
            "enc_type",
            "dtype",
            "bw",
            "is_sym",
            "scale",
            "offset",
        }
        assert exported_encodings["enc_type"] == EncodingType.PER_CHANNEL.name
        assert len(exported_encodings["scale"]) == tensor_shape[channel_axis]
        assert len(exported_encodings["offset"]) == tensor_shape[channel_axis]

        block_size = 4
        qc_quantize_op._enable_blockwise_quantization(block_size)
        encodings = [
            libpymo.TfEncoding() for _ in range(tensor_shape[channel_axis] * 2)
        ]
        for encoding in encodings:
            encoding.min = delta * offset
            encoding.max = delta * (offset + 2**bitwidth - 1)
            encoding.bw = bitwidth
            encoding.offset = offset
            encoding.delta = delta
        qc_quantize_op.load_encodings(encodings)
        exported_encodings = qc_quantize_op.export_encodings("1.0.0")
        assert exported_encodings.keys() == {
            "enc_type",
            "dtype",
            "bw",
            "is_sym",
            "scale",
            "offset",
            "block_size",
        }
        assert exported_encodings["enc_type"] == EncodingType.PER_BLOCK.name
        assert len(exported_encodings["scale"]) == tensor_shape[channel_axis] * 2
        assert exported_encodings["block_size"] == block_size

        qc_quantize_op.enabled = False
        assert qc_quantize_op.export_encodings("0.6.1") is None
        assert qc_quantize_op.export_encodings("1.0.0") is None

    def test_export_float_encodings(self):
        quant_info = libquant_info.QcQuantizeInfo()
        qc_quantize_op = QcQuantizeOp(
            quant_info,
            bitwidth=16,
            op_mode=OpMode.quantizeDequantize,
            tensor_quantizer_params=TensorQuantizerParams([2, 2], 0, 1),
        )
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
        qc_quantize_op = QcQuantizeOp(
            quant_info, bitwidth=16, op_mode=OpMode.quantizeDequantize
        )
        qc_quantize_op.data_type = QuantizationDataType.float
        with pytest.raises(RuntimeError):
            qc_quantize_op.load_encodings([libpymo.TfEncoding()])

    def test_load_encoding_granularity(self):
        tensor_quantizer_params = TensorQuantizerParams((10, 15), 0, 1)
        qc_quantize_op = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            bitwidth=8,
            op_mode=OpMode.updateStats,
            tensor_quantizer_params=tensor_quantizer_params,
        )
        qc_quantize_op.update_encoding_stats(np.random.randn(10, 15))
        qc_quantize_op.compute_encodings()
        assert qc_quantize_op._encoding_shape() == ()
        per_tensor_enc_dict = qc_quantize_op.export_encodings("1.0.0")

        # Enable per-channel quantization and compute encodings
        qc_quantize_op.enable_per_channel_quantization()
        qc_quantize_op.update_encoding_stats(np.random.randn(10, 15))
        qc_quantize_op.compute_encodings()
        enc_dict = qc_quantize_op.export_encodings("1.0.0")
        assert enc_dict

        # Create a new per-tensor quantizer
        qc_quantize_op = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            bitwidth=8,
            op_mode=OpMode.updateStats,
            tensor_quantizer_params=tensor_quantizer_params,
        )
        assert qc_quantize_op._encoding_shape() == ()

        # After loading encodings, should be in per-channel mode
        qc_quantize_op._load_encodings_dict(enc_dict)
        assert len(qc_quantize_op.get_encodings()) == 10
        assert qc_quantize_op._encoding_shape() == (10,)

        # Enable blockwise quantization and compute encodings
        qc_quantize_op._enable_blockwise_quantization(block_size=3)
        assert qc_quantize_op._encoding_shape() == (10, 5)
        qc_quantize_op.update_encoding_stats(np.random.randn(10, 15))
        qc_quantize_op.compute_encodings()

        block_enc_dict = qc_quantize_op.export_encodings("1.0.0")

        # Create new per-tensor qc_quantize_op
        qc_quantize_op = QcQuantizeOp(
            libquant_info.QcQuantizeInfo(),
            bitwidth=8,
            op_mode=OpMode.updateStats,
            tensor_quantizer_params=tensor_quantizer_params,
        )
        assert qc_quantize_op._encoding_shape() == ()

        # After loading encodings, should be blockwise
        qc_quantize_op._load_encodings_dict(block_enc_dict)
        assert len(qc_quantize_op.get_encodings()) == 50
        assert qc_quantize_op._encoding_shape() == (10, 5)

        # After loading per-tensor encodings, should be per-tensor quantizer
        qc_quantize_op._load_encodings_dict(per_tensor_enc_dict)
        assert qc_quantize_op._encoding_shape() == ()
        assert len(qc_quantize_op.get_encodings()) == 1


class TestLPBQOp:
    def test_lpbq_quantize_op(self):
        input_shape = (2, 9)
        scale = np.asarray(
            [
                [1.6, 1.1222, 0.00001],
                [16, 2.56, 4.9],
            ],
            np.float32,
        )
        offset = np.ones_like(scale) * -8
        expected_lpbq_scale = np.asarray([[1.6, 1.1, 0.1], [16, 3, 5]], np.float32)
        expected_per_channel_scale = np.asarray([1.6 / 2**4, 16 / 2**4], np.float32)
        bitwidth = 4
        decompressed_bw = 8
        quant_info = libquant_info.QcQuantizeInfo()
        tensor_quantizer_params = TensorQuantizerParams(
            input_shape, channel_axis=0, block_axis=1
        )
        lpbq_op = QcQuantizeOp(
            quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            tensor_quantizer_params=tensor_quantizer_params,
        )
        lpbq_op.set_qspec(
            QSpec.lpbq(
                qtype.int(bitwidth), block_size=3, scale_bits=decompressed_bw - bitwidth
            )
        )

        encodings = numpy_to_TfEncoding(scale, offset, qtype.int(bitwidth))
        """
        When: Load blockwise encodings to an LPBQ quantizer
        Then: Quantizer should apply LPBQ to encodings during load_encodings
        """
        lpbq_op.load_encodings(encodings)
        lpbq_encodings = lpbq_op.get_encodings()
        lpbq_scale, lpbq_offset = numpy_from_TfEncoding(lpbq_encodings, (2, 3))
        assert np.allclose(lpbq_scale, expected_lpbq_scale)
        assert np.allclose(lpbq_offset, offset)
        """
        Run LPBQ Quantizer in QDQ mode
        """
        session = create_qc_quantize_model_session(quant_info, input_shape)
        input_tensor = np.random.randn(*input_shape).astype(np.float32)
        output_tensor = session.run(None, {"input": input_tensor})[0]
        """
        Compute the expected LPBQ Output
        """
        input_tensor_bcast, scale_bcast = (
            input_tensor.reshape((2, 3, 3)),
            expected_lpbq_scale.reshape((2, 3, 1)),
        )
        expected_output = (
            np.round(np.clip(input_tensor_bcast / scale_bcast, -8, 7)) * scale_bcast
        ).reshape(input_shape)
        """
        Check that output matches expectation
        """
        assert np.allclose(expected_output, output_tensor)
        """
        Verify 1.0.0 export logic
        """
        exported_encodings = lpbq_op.export_encodings("1.0.0")
        expected_int_scale = [16, 11, 1, 16, 3, 5]
        assert exported_encodings.keys() == {
            "enc_type",
            "dtype",
            "bw",
            "is_sym",
            "scale",
            "offset",
            "block_size",
            "compressed_bw",
            "per_block_int_scale",
        }

        assert all(offset == -128 for offset in exported_encodings["offset"])
        assert exported_encodings["per_block_int_scale"] == expected_int_scale
        assert exported_encodings["compressed_bw"] == 4
        assert exported_encodings["bw"] == 8
        assert exported_encodings["enc_type"] == EncodingType.LPBQ.name
        assert np.allclose(
            np.asarray(exported_encodings["scale"]),
            np.asarray(expected_per_channel_scale),
        )
        assert exported_encodings["offset"] == [-128, -128]

        with pytest.raises(ValueError):
            lpbq_op.export_encodings("0.6.1")

        expected_per_channel_scale = expected_per_channel_scale.reshape(2, 1)
        expected_per_block_int_scale = (
            (expected_lpbq_scale / expected_per_channel_scale).round().astype(np.int32)
        )
        assert lpbq_op.export_encodings("2.0.0") == {
            "output_dtype": "int4",
            "per_channel_float_scale": expected_per_channel_scale.tolist(),
            "per_block_int_scale": expected_per_block_int_scale.tolist(),
            "axis": 1,
            "block_size": 3,
        }

    def test_decompressed_bw_property(self):
        input_shape = (2, 9)
        bitwidth = 4
        decompressed_bw = 8
        quant_info = libquant_info.QcQuantizeInfo()
        tensor_quantizer_params = TensorQuantizerParams(
            input_shape, channel_axis=0, block_axis=1
        )
        lpbq_op = GroupedBlockQuantizeDequantize(
            quant_info,
            bitwidth,
            decompressed_bw,
            block_size=3,
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.quantizeDequantize,
            tensor_quantizer_params=tensor_quantizer_params,
        )
        """
        When: Reading decompressed_bw
        Then: Should reflect the compressed bitwidth plus the scale quantizer's scale bits
        """
        assert lpbq_op.decompressed_bw == decompressed_bw
        assert lpbq_op._scale_quantizer.scale_bits == decompressed_bw - bitwidth

        """
        When: Writing decompressed_bw
        Then: Scale quantizer should be rebuilt with the new scale bitwidth, and reading
              decompressed_bw back should return the newly set value
        """
        new_decompressed_bw = 16
        lpbq_op.decompressed_bw = new_decompressed_bw
        assert lpbq_op._scale_quantizer.scale_bits == new_decompressed_bw - bitwidth
        assert lpbq_op.decompressed_bw == new_decompressed_bw

    def test_compute_lpbq_encodings(self):
        input_shape = (4, 2)
        bitwidth = 4
        decompressed_bw = 8
        block_size = 2
        quant_info = libquant_info.QcQuantizeInfo()
        tensor_quantizer_params = TensorQuantizerParams(
            input_shape, channel_axis=1, block_axis=0
        )
        lpbq_op = QcQuantizeOp(
            quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.updateStats,
            tensor_quantizer_params=tensor_quantizer_params,
        )
        lpbq_op.set_qspec(
            QSpec.lpbq(
                qtype.int(bitwidth),
                block_size=block_size,
                scale_bits=decompressed_bw - bitwidth,
            )
        )

        # Note: computed delta = abs_max / num_positive_steps = abs_max / 7
        input_tensor = np.asarray(
            [
                [7.0 * 32, -8 * 1.6],
                [-0.35, 7.343],
                [7.0 * 13.334, 8 * -1.1112],
                [22.1, 0.11233],
            ],
            np.float32,
        )
        expected_scale = np.asarray([[32.0, 1.6], [14, 1.1]], np.float32)
        session = create_qc_quantize_model_session(quant_info, input_shape)
        session.run(None, {"input": input_tensor})
        lpbq_op.compute_encodings()

        encodings = lpbq_op.get_encodings()
        scale, _ = numpy_from_TfEncoding(encodings, expected_scale.shape)
        assert np.allclose(scale, expected_scale)

    def test_grouped_block_qdq_perchannel_mode(self):
        input_shape = (4, 2)
        bitwidth = 4
        decompressed_bw = 8
        block_size = 0
        quant_info = libquant_info.QcQuantizeInfo()
        tensor_quantizer_params = TensorQuantizerParams(
            input_shape, channel_axis=1, block_axis=0
        )
        lpbq_op = QcQuantizeOp(
            quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.updateStats,
            tensor_quantizer_params=tensor_quantizer_params,
        )
        lpbq_op.set_qspec(
            QSpec.lpbq(
                qtype.int(bitwidth),
                block_size=block_size,
                scale_bits=decompressed_bw - bitwidth,
            )
        )
        lpbq_op.enable_per_channel_quantization()

        # Note: computed delta = abs_max / num_positive_steps = abs_max / 7
        input_tensor = np.asarray(
            [
                [7.0 * 32, -8 * 1.6],
                [-0.35, 7.343],
                [7.0 * 13.334, 8 * -1.1112],
                [22.1, 0.11233],
            ],
            np.float32,
        )
        expected_scale = np.asarray(
            [
                [32.0, 1.6],
            ],
            np.float32,
        )
        session = create_qc_quantize_model_session(quant_info, input_shape)
        session.run(None, {"input": input_tensor})
        lpbq_op.compute_encodings()

        encodings = lpbq_op.export_encodings("1.0.0")
        assert np.allclose(
            np.asarray(encodings["scale"]).astype("float32"), expected_scale
        )


def _onnx_QuantizeDequantizeLinear(
    input_shape, y_scale, y_zero_point, axis, block_size, output_dtype
):
    op = OperatorSetIdProto()
    op.version = 21

    assert output_dtype in ("int8", "int16", "uint8", "uint16")

    x_int_dtype = (
        TensorProto.INT16
        if output_dtype == "int16"
        else TensorProto.INT8
        if output_dtype == "int8"
        else TensorProto.INT4
        if output_dtype == "int4"
        else TensorProto.UINT16
        if output_dtype == "uint16"
        else TensorProto.UINT8
        if output_dtype == "uint8"
        else TensorProto.UINT4
        if output_dtype == "uint4"
        else None
    )
    assert x_int_dtype is not None

    x = helper.make_tensor_value_info(
        name="x", elem_type=TensorProto.FLOAT, shape=input_shape
    )

    y_scale = numpy_helper.from_array(
        np.array(y_scale).astype("float32"), name="y_scale"
    )
    if y_zero_point is not None:
        y_zero_point = numpy_helper.from_array(
            np.array(y_zero_point).astype(output_dtype), name="y_zero_point"
        )

    y = helper.make_tensor_value_info(
        name="y", elem_type=TensorProto.FLOAT, shape=input_shape
    )

    quantize_node = helper.make_node(
        "QuantizeLinear",
        inputs=["x", "y_scale", "y_zero_point"] if y_zero_point else ["x", "y_scale"],
        outputs=["x_int"],
        axis=axis,
        block_size=block_size,
        output_dtype=x_int_dtype,
    )

    dequantize_node = helper.make_node(
        "DequantizeLinear",
        inputs=["x_int", "y_scale", "y_zero_point"]
        if y_zero_point
        else ["x_int", "y_scale"],
        outputs=["y"],
        axis=axis,
        block_size=block_size,
    )

    onnx_graph = helper.make_graph(
        [quantize_node, dequantize_node],
        name="quantize_dequantize",
        inputs=[x],
        outputs=[y],
        initializer=[y_scale, y_zero_point] if y_zero_point is not None else [y_scale],
    )

    model = helper.make_model(
        onnx_graph, opset_imports=[op], ir_version=_DEFAULT_IR_VERSION
    )
    onnx.checker.check_model(model, True)

    return model


@pytest.mark.parametrize(
    # NOTE: In onnx, "axis" is overloaded with two meanings.
    #
    #         +- channel axis (if block size is None)
    # axis := |
    #         +- block axis (otherwise)
    "input_shape,    channel_axis, block_axis,  block_size",
    [
        ((10, 10, 1, 1), None, None, None),  # per-tensor
        ((10, 10, 1, 1), 0, None, None),  # per-channel with axis=0 (Convolution)
        ((10, 10, 1, 1), 1, None, None),  # per-channel with axis=1 (Convolution)
        ((10, 10), 0, None, None),  # per-channel with axis=0 (Linear/Gemm)
        ((10, 10), 1, None, None),  # per-channel with axis=1 (Linear/Gemm)
        ((10, 10, 1, 1), 0, 1, 5),  # per-block with block_axis=1 (Convolution)
        ((10, 10, 1, 1), 1, 0, 5),  # per-block with block_axis=0 (Convolution)
        ((10, 10), 0, 1, 5),  # per-block with block_axis=1 (Linear/Gemm)
        ((10, 10), 1, 0, 5),  # per-block with block_axis=0 (Linear/Gemm)
    ],
)
@pytest.mark.parametrize(
    "bitwidth, symmetric",
    [
        (4, True),
        (4, False),
        (8, True),
        (8, True),
        (8, False),
        (16, True),
        (16, False),
        (32, True),
        # NOTE: Skipping since simulating int32 with non-zero offset is numerically very unstable
        # (32,       False),
    ],
)
def test_affine_encoding_schema_2_0_0(
    input_shape, channel_axis, block_axis, block_size, bitwidth, symmetric
):
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

    quant_node = helper.make_node(
        op_name,
        inputs=["input"],
        outputs=["output"],
        domain=op_domain,
        quant_info=libpymo.PtrToInt64(quant_info),
    )
    model = create_model_from_node(quant_node, input.shape)
    session = build_session(model, available_providers)
    qtzr = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        rounding_mode="nearest",
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=bitwidth,
        use_symmetric_encodings=symmetric,
        tensor_quantizer_params=quant_params,
    )

    if block_axis is not None:
        qtzr._enable_blockwise_quantization(block_size)
    elif channel_axis is not None:
        qtzr.enable_per_channel_quantization()

    (_,) = session.run(None, {"input": input})
    qtzr.compute_encodings()

    """
    When: Export encoding in 2.0.0 schema
    """
    encoding = qtzr.export_encodings("2.0.0")

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
        assert all(
            dim == 1
            for axis, dim in enumerate(y_scale.shape)
            if axis not in (channel_axis, block_axis)
        )
    elif channel_axis is not None:
        assert y_scale.shape == (input_shape[channel_axis],)
    else:
        assert y_scale.shape == ()

    if symmetric:
        assert "y_zero_point" not in encoding
    else:
        assert np.array(encoding["y_zero_point"]).shape == y_scale.shape

    if block_axis is not None:
        assert encoding["axis"] == block_axis
    elif channel_axis is not None:
        assert encoding["axis"] == channel_axis
    else:
        assert "axis" not in encoding

    if block_size is None:
        assert "block_size" not in encoding
    else:
        assert encoding["block_size"] == block_size

    assert encoding["output_dtype"] == (
        f"int{bitwidth}" if symmetric else f"uint{bitwidth}"
    )

    """
    Then: The output of onnx::QuantizeLinear followed by DequantizeLinear with the exported qnn encoding
          should be all-close to AIMET qdq output with off-by-one tolerance threshold
    """
    if bitwidth not in (8, 16):
        pytest.skip(reason="onnx::QuantizeLinear only supports these data types")

    if block_axis is not None and version.parse(ort.__version__) < version.parse(
        "1.20.0"
    ):
        pytest.skip(
            reason="Remaining tests require onnxruntime>=1.20 for blockwise QuantizeLinear"
        )

    onnx_QuantizeLinear = _onnx_QuantizeDequantizeLinear(
        input_shape=input.shape,
        y_scale=encoding["y_scale"],
        y_zero_point=encoding.get("y_zero_point", None),
        axis=encoding.get("axis", None),
        block_size=encoding.get("block_size", None),
        output_dtype=encoding["output_dtype"],
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        full_path = os.path.join(tmp_dir, "model.onnx")

        with open(full_path, "wb") as f:
            f.write(onnx_QuantizeLinear.SerializeToString())

        sess = ort.InferenceSession(full_path, providers=["CPUExecutionProvider"])
        (ort_out,) = sess.run(None, {"x": input})

    aimet_out = session.run(None, {"input": input})
    atol = y_scale  # Allow off-by-one error
    if block_axis is not None:
        atol = atol.max(axis=block_axis, keepdims=True)
    elif channel_axis is not None:
        atol = atol.reshape(
            *(1 if axis != channel_axis else -1 for axis in range(input.ndim))
        )
    assert np.allclose(ort_out, aimet_out, atol=atol)


def _onnx_LPBQ(
    input_shape,
    per_block_int_scale,
    per_channel_float_scale,
    y_zero_point,
    axis,
    block_size,
    output_dtype,
):
    op = OperatorSetIdProto()
    op.version = 21

    assert y_zero_point is None

    x_int_dtype = (
        TensorProto.INT16
        if output_dtype == "int16"
        else TensorProto.INT8
        if output_dtype == "int8"
        else TensorProto.INT4
        if output_dtype == "int4"
        else TensorProto.UINT16
        if output_dtype == "uint16"
        else TensorProto.UINT8
        if output_dtype == "uint8"
        else TensorProto.UINT4
        if output_dtype == "uint4"
        else None
    )
    assert x_int_dtype is not None

    x = helper.make_tensor_value_info(
        name="x", elem_type=TensorProto.FLOAT, shape=input_shape
    )

    per_block_int_scale = numpy_helper.from_array(
        np.array(per_block_int_scale).astype("float32"), name="per_block_int_scale"
    )
    per_channel_float_scale = numpy_helper.from_array(
        np.array(per_channel_float_scale).astype("float32"),
        name="per_channel_float_scale",
    )

    y = helper.make_tensor_value_info(
        name="y", elem_type=TensorProto.FLOAT, shape=input_shape
    )

    mul_node = helper.make_node(
        "Mul",
        inputs=["per_block_int_scale", "per_channel_float_scale"],
        outputs=["y_scale"],
    )

    quantize_node = helper.make_node(
        "QuantizeLinear",
        inputs=["x", "y_scale"],
        outputs=["x_int"],
        axis=axis,
        block_size=block_size,
        output_dtype=x_int_dtype,
    )

    dequantize_node = helper.make_node(
        "DequantizeLinear",
        inputs=["x_int", "y_scale"],
        outputs=["y"],
        axis=axis,
        block_size=block_size,
    )

    onnx_graph = helper.make_graph(
        [mul_node, quantize_node, dequantize_node],
        name="lpbq",
        inputs=[x],
        outputs=[y],
        initializer=[per_block_int_scale, per_channel_float_scale],
    )

    model = helper.make_model(
        onnx_graph, opset_imports=[op], ir_version=_DEFAULT_IR_VERSION
    )
    onnx.checker.check_model(model, True)

    return model


@pytest.mark.parametrize(
    "input_shape,    block_axis,  block_size",
    [
        ((10, 50, 1, 1), 1, 5),  # per-block with block_axis=1 (Convolution)
        ((50, 10, 1, 1), 0, 5),  # per-block with block_axis=0 (Convolution)
        ((10, 50), 1, 5),  # per-block with block_axis=1 (Linear/Gemm)
        ((50, 10), 0, 5),  # per-block with block_axis=0 (Linear/Gemm)
    ],
)
@pytest.mark.parametrize(
    "compressed_bw, decompressed_bw",
    [
        (4, 8),
        (8, 16),
    ],
)
def test_lpbq_encoding_schema_2_0_0(
    input_shape, block_axis, block_size, compressed_bw, decompressed_bw
):
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

    quant_node = helper.make_node(
        op_name,
        inputs=["input"],
        outputs=["output"],
        domain=op_domain,
        quant_info=libpymo.PtrToInt64(quant_info),
    )
    model = create_model_from_node(quant_node, input.shape)
    session = build_session(model, available_providers)
    qtzr = QcQuantizeOp(
        quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        tensor_quantizer_params=quant_params,
    )
    qtzr.set_qspec(
        QSpec.lpbq(
            qtype.int(compressed_bw),
            block_size=block_size,
            scale_bits=decompressed_bw - compressed_bw,
        )
    )

    (_,) = session.run(None, {"input": input})
    qtzr.compute_encodings()

    """
    When: Export encoding in 2.0.0 schema
    """
    encoding = qtzr.export_encodings("2.0.0")

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
    assert (
        per_block_int_scale.shape[block_axis] == input.shape[block_axis] // block_size
    )
    assert all(
        dim == 1
        for axis, dim in enumerate(per_block_int_scale.shape)
        if axis not in (channel_axis, block_axis)
    )
    assert per_channel_float_scale.shape[channel_axis] == input.shape[channel_axis]
    assert all(
        dim == 1
        for axis, dim in enumerate(per_channel_float_scale.shape)
        if axis != channel_axis
    )

    assert "y_zero_point" not in encoding
    assert encoding["axis"] == block_axis
    assert encoding["block_size"] == block_size
    assert encoding["output_dtype"] == f"int{compressed_bw}"

    """
    Then: The output of onnx::QuantizeLinear followed by DequantizeLinear with the exported qnn encoding
          should be all-close to AIMET qdq output with off-by-one tolerance threshold
    """
    if version.parse(ort.__version__) < version.parse("1.20.0"):
        pytest.skip(
            reason="Remaining tests require onnxruntime>=1.20 for blockwise QuantizeLinear"
        )

    onnx_LPBQ = _onnx_LPBQ(
        input_shape=input.shape,
        per_block_int_scale=encoding["per_block_int_scale"],
        per_channel_float_scale=encoding["per_channel_float_scale"],
        y_zero_point=None,
        axis=encoding["axis"],
        block_size=encoding["block_size"],
        output_dtype=encoding["output_dtype"],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        full_path = os.path.join(tmp_dir, "model.onnx")

        with open(full_path, "wb") as f:
            f.write(onnx_LPBQ.SerializeToString())

        sess = ort.InferenceSession(full_path, providers=["CPUExecutionProvider"])
        (ort_out,) = sess.run(None, {"x": input})

    aimet_out = session.run(None, {"input": input})
    y_scale = per_block_int_scale * per_channel_float_scale
    atol = y_scale.max(axis=block_axis, keepdims=True)  # Allow off-by-one error
    assert np.allclose(ort_out, aimet_out, atol=atol)


def test_quantizer_with_zero_point_shift():
    input_shape = (3, 20)
    channel_axis = 0
    bitwidth = 2
    zero_point_shift = 0.5

    tensor_quantizer = create_tensor_quantizer(
        input_shape,
        bitwidth,
        channel_axis,
        quant_scheme=QuantScheme.post_training_tf,
    )
    tensor_quantizer.setZeroPointShift(zero_point_shift)
    assert tensor_quantizer.getZeroPointShift() == zero_point_shift

    exp_max = np.array([1.5, 1, 0.5]).astype(np.float32).reshape(3, 1)
    exp_delta = (2 * exp_max) / (2**bitwidth - 1)

    input_tensor = np.random.randn(*input_shape).astype(np.float32) * 5
    # Clip observed tensor to control the min/max range
    clipped_tensor = np.clip(input_tensor, -exp_max, exp_max)
    tensor_quantizer.updateStats(clipped_tensor)
    enc = tensor_quantizer.computeEncodings(True)
    tensor_quantizer.setEncodings(enc)

    for enc, expected_max in zip(enc, exp_max.flatten().tolist()):
        assert enc.max == expected_max
        assert enc.min == -expected_max
        assert enc.delta == (enc.max - enc.min) / (2**bitwidth - 1)
        assert enc.offset == -(2 ** (bitwidth - 1)) + zero_point_shift

    # Quantized un-clipped tensor
    qdq_tensor = tensor_quantizer.quantizeDequantize(input_tensor)
    assert not np.any(qdq_tensor == 0)

    exp_qdq_tensor = (
        np.round(clipped_tensor / exp_delta - zero_point_shift) + zero_point_shift
    ) * exp_delta
    assert np.allclose(qdq_tensor, exp_qdq_tensor)


def test_qc_quantize_op_maintains_zero_point_shift():
    input_shape = (3, 20)
    tensor_quantizer_params = TensorQuantizerParams(input_shape, 0, 1)
    qc_op = QcQuantizeOp(
        quant_info=libquant_info.QcQuantizeInfo(),
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=2,
        use_symmetric_encodings=True,
        tensor_quantizer_params=tensor_quantizer_params,
    )
    qc_op._tensor_quantizer.setZeroPointShift(0.5)
    assert qc_op._tensor_quantizer.getZeroPointShift() == 0.5
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    qc_op.update_encoding_stats(dummy_input)
    qc_op.compute_encodings()
    assert all(enc.offset == -1.5 for enc in qc_op.get_encodings())
    assert all(enc.min == -enc.max for enc in qc_op.get_encodings())
    assert all(enc.min == enc.offset * enc.delta for enc in qc_op.get_encodings())

    qc_op.enable_per_channel_quantization()
    assert qc_op._tensor_quantizer.getZeroPointShift() == 0.5
    qc_op.update_encoding_stats(dummy_input)
    qc_op.compute_encodings()
    assert all(enc.offset == -1.5 for enc in qc_op.get_encodings())
    assert all(enc.min == -enc.max for enc in qc_op.get_encodings())
    assert all(enc.min == enc.offset * enc.delta for enc in qc_op.get_encodings())

    qc_op._enable_blockwise_quantization(block_size=10)
    assert qc_op._tensor_quantizer.getZeroPointShift() == 0.5
    qc_op.update_encoding_stats(dummy_input)
    qc_op.compute_encodings()
    assert all(enc.offset == -1.5 for enc in qc_op.get_encodings())
    assert all(enc.min == -enc.max for enc in qc_op.get_encodings())
    assert all(enc.min == enc.offset * enc.delta for enc in qc_op.get_encodings())


def test_load_encodings_with_zero_point_shift():
    np.random.seed(0)
    input_shape = (3, 20)
    channel_axis = 0
    block_axis = 1
    bitwidth = 2
    zero_point_shift = 0.5
    min_val = -0.75
    max_val = 0.75

    quant_info = libquant_info.QcQuantizeInfo()
    tensor_quantizer_params = TensorQuantizerParams(
        input_shape, channel_axis, block_axis
    )
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=8,
        use_symmetric_encodings=True,
        tensor_quantizer_params=tensor_quantizer_params,
    )
    qc_op.enable_per_channel_quantization()

    """
    When: Loading encodings with non-zero zero_point_shift
    Then: 1) Stored encodings should reflect the zero_point_shift correctly
          2) Tensors should be quantized correctly using shifted offset
    """
    encodings = []
    for _ in range(input_shape[channel_axis]):
        encoding = libpymo.TfEncoding()
        encoding.min = min_val
        encoding.max = max_val
        encoding.bw = bitwidth
        encoding.delta = (encoding.max - encoding.min) / (2**bitwidth - 1)
        encoding.offset = -(2 ** (bitwidth - 1)) + zero_point_shift
        encodings.append(encoding)

    qc_op.load_encodings(encodings)
    loaded_encodings = qc_op.get_encodings()

    for enc in loaded_encodings:
        assert enc.offset == -(2 ** (bitwidth - 1)) + zero_point_shift
        assert enc.min == min_val
        assert enc.max == max_val
        assert enc.delta == (enc.max - enc.min) / (2**bitwidth - 1)
        assert enc.bw == bitwidth

    assert qc_op.bitwidth == bitwidth
    input_tensor = np.random.randn(*input_shape).astype(np.float32)
    qdq_tensor = qc_op.quantize_dequantize(input_tensor)

    delta = np.array([enc.delta for enc in loaded_encodings]).reshape(-1, 1)

    rounded = np.round(input_tensor / delta - zero_point_shift)
    clipped = np.clip(rounded, -(2 ** (bitwidth - 1)), 2 ** (bitwidth - 1) - 1)
    exp_qdq_tensor = (clipped + zero_point_shift) * delta

    assert np.all(qdq_tensor == exp_qdq_tensor)
    assert not np.any(qdq_tensor == 0)

    """
    When: Loading encodings with no zero_point_shift
    Then: 1) Stored encodings should have integer offsets
          2) Tensors should be quantized correctly using integer offset
    """
    for enc in encodings:
        enc.offset = -(2 ** (bitwidth - 1))
        enc.min -= enc.delta * zero_point_shift
        enc.max -= enc.delta * zero_point_shift

    qc_op.load_encodings(encodings)
    loaded_encodings = qc_op.get_encodings()
    for enc in loaded_encodings:
        assert enc.offset == -(2 ** (bitwidth - 1))

    qdq_tensor = qc_op.quantize_dequantize(input_tensor)

    rounded = np.round(input_tensor / delta)
    clipped = np.clip(rounded, -(2 ** (bitwidth - 1)), 2 ** (bitwidth - 1) - 1)
    exp_qdq_tensor = clipped * delta
    assert np.all(qdq_tensor == exp_qdq_tensor)


@pytest.mark.parametrize("freeze", [False, True])
def test_load_encodings_with_frozen_encodings(freeze: bool):
    input_shape = (3, 20)
    bitwidth = 8
    qc_op = _new_quantizer(
        input_shape, bitwidth=bitwidth, use_symmetric_encodings=False
    )
    _calibrate_and_qdq(qc_op, np.random.randn(*input_shape).astype(np.float32))
    if freeze:
        qc_op.freeze_encodings()

    """
    When: Loading encodings into a quantizer
    Then: Encodings should only be overwritten if they are not frozen
    """
    scale_before = qc_op._get_scale()
    qc_op.load_encodings(create_encoding(-1e4, 1e4, bitwidth, False))

    if freeze:
        assert np.array_equal(qc_op._get_scale(), scale_before)
    else:
        assert not np.array_equal(qc_op._get_scale(), scale_before)


def _quantizer_state(qc_op: QcQuantizeOp):
    """Returns the full observable configuration and encodings of a quantizer"""
    scale, offset = qc_op._get_scale(), qc_op._get_offset()
    return (
        qc_op.precision(),
        qc_op.bitwidth,
        qc_op.data_type,
        qc_op.use_symmetric_encodings,
        qc_op.use_strict_symmetric,
        qc_op.use_unsigned_symmetric,
        qc_op.get_zero_point_shift(),
        qc_op.is_initialized(),
        qc_op.quant_info.usePerChannelMode,
        qc_op.quant_info.blockSize,
        None if scale is None else scale.tolist(),
        None if offset is None else offset.tolist(),
    )


def _new_frozen_quantizer(input_shape=(3, 20)) -> QcQuantizeOp:
    qc_op = _new_quantizer(input_shape, bitwidth=8, use_symmetric_encodings=False)
    _calibrate_and_qdq(qc_op, np.random.randn(*input_shape).astype(np.float32))
    qc_op.freeze_encodings()
    return qc_op


_FROZEN_NO_OPS = {
    "data_type": lambda q: setattr(q, "data_type", QuantizationDataType.float),
    "use_symmetric_encodings": lambda q: setattr(q, "use_symmetric_encodings", True),
    "load_encodings": lambda q: q.load_encodings(create_encoding(-1e4, 1e4, 16, False)),
    "update_quantizer_and_load_encodings": lambda q: q.update_quantizer_and_load_encodings(
        create_encoding(-1e4, 1e4, 16, False),
        True,
        True,
        True,
        QuantizationDataType.int,
    ),
    "_load_encodings_dict": lambda q: q._load_encodings_dict(
        AffineEncoding(
            scale=np.array([1e4]), offset=np.array([-32768]), dtype="int16"
        ).to_qnn_encoding_dict("1.0.0")
    ),
    "reset_encoding_stats": lambda q: q.reset_encoding_stats(),
    "_reset_encodings": lambda q: q._reset_encodings(),
    "set_bitwidth": lambda q: q.set_bitwidth(16),
    "set_precision": lambda q: q.set_precision(aimet_onnx.float16),
    "set_qspec": lambda q: q.set_qspec(QSpec.per_channel("int16", symmetric=True)),
    "set_zero_point_shift": lambda q: q.set_zero_point_shift(0.5),
    # update_encoding_stats is not frozen-guarded, so new stats are observed but must
    # not make it into the frozen encodings
    "compute_encodings": lambda q: (
        q.update_encoding_stats(np.full((3, 20), 1e4, dtype=np.float32)),
        q.compute_encodings(),
    ),
    "clip_and_recompute_encodings": lambda q: q.clip_and_recompute_encodings(1e-3),
}


@pytest.mark.parametrize(
    "operation", _FROZEN_NO_OPS.values(), ids=_FROZEN_NO_OPS.keys()
)
def test_frozen_quantizer_ignores_reconfiguration(operation):
    """
    Given: A quantizer with frozen encodings
    When: Applying an operation which would reconfigure or recalibrate the quantizer
    Then: The quantizer is left untouched
    """
    qc_op = _new_frozen_quantizer()

    state_before = _quantizer_state(qc_op)
    operation(qc_op)
    assert _quantizer_state(qc_op) == state_before


def test_frozen_quantizer_rejects_bitwidth_assignment():
    """
    Given: A quantizer with frozen encodings
    When: Assigning to bitwidth directly
    Then: Raise, since the assignment cannot be honored without invalidating the encodings
    """
    qc_op = _new_frozen_quantizer()

    state_before = _quantizer_state(qc_op)
    with pytest.raises(RuntimeError):
        qc_op.bitwidth = 16
    assert _quantizer_state(qc_op) == state_before


@pytest.mark.parametrize("encoding_version", ["1.0.0", "2.0.0"])
def test_encoding_dict_with_zero_point_shift(encoding_version: str):
    encoding = AffineEncoding(
        scale=np.array([0.25, 0.5]),
        offset=np.array([0.5, 0.5]),
        dtype="int2",
        channel_axis=0,
    )
    encoding_dict = encoding.to_qnn_encoding_dict(encoding_version)

    input_shape = (2, 1)
    quant_info = libquant_info.QcQuantizeInfo()
    tensor_quantizer_params = TensorQuantizerParams(input_shape, 0)
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=8,
        use_symmetric_encodings=True,
        tensor_quantizer_params=tensor_quantizer_params,
    )
    assert qc_op._tensor_quantizer.getZeroPointShift() == 0.0

    """
    When: Loading encoding dict with zero point shift
    Then: 1) Stored encodings contain shifted offset
          2) quantizer.zeroPointShift is updated accordingly
    """
    qc_op._load_encodings_dict(encoding_dict)
    assert qc_op._tensor_quantizer.getZeroPointShift() == 0.5
    assert AffineEncoding.from_quantizer(qc_op) == encoding
    assert qc_op.export_encodings(encoding_version) == encoding_dict

    """
    When: Try to import encodings with inconsistent zero_point_shift
    Then: Raise error
    """
    encoding.offset[-1] = 0.75
    with pytest.raises(RuntimeError):
        qc_op._load_encodings_dict(encoding.to_qnn_encoding_dict(encoding_version))

    """
    When: Loading encoding dict with no zero point shift
    Then: 1) Stored encodings contain integer offset
          2) quantizer.zeroPointShift should be 0.0
    """
    encoding.offset[:] = 0.0
    encoding_dict = encoding.to_qnn_encoding_dict(encoding_version)
    qc_op._load_encodings_dict(encoding_dict)
    assert qc_op._tensor_quantizer.getZeroPointShift() == 0.0
    assert AffineEncoding.from_quantizer(qc_op) == encoding
    assert qc_op.export_encodings(encoding_version) == encoding_dict


@pytest.mark.skip(
    reason="Exporting/Loading LPBQ encodings with zero point shift is not supported yet"
)
def test_import_1_0_0_LPBQ_encodings_with_zero_point_shift():
    encoding_dict = {
        "block_size": 2,
        "bw": 8,
        "compressed_bw": 2,
        "dtype": "INT",
        "enc_type": "LPBQ",
        "is_sym": True,
        "offset": [-128, -128],
        "per_block_int_scale": [32, 64, 32, 64],
        "scale": [3.0, 3.0],
        "zero_point_shift": [0.5, 0.5],
    }
    input_shape = (2, 4)
    quant_info = libquant_info.QcQuantizeInfo()
    tensor_quantizer_params = TensorQuantizerParams(input_shape, 0, 1)
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        tensor_quantizer_params=tensor_quantizer_params,
    )
    qc_op.set_qspec(QSpec.lpbq(qtype.int(4), block_size=0, scale_bits=4))
    """
    When: Loading LPBQ encodings with zero_point_shift
    Then: 1) Stored encodings contain shifted offset
          2) quantizer.getZeroPointShift is updated appropriately
    """
    qc_op._load_encodings_dict(encoding_dict)
    assert qc_op._tensor_quantizer.getZeroPointShift() == 0.5
    for pb_scale, enc in zip(
        encoding_dict["per_block_int_scale"], qc_op.get_encodings()
    ):
        assert enc.delta == pb_scale * 3.0
        assert enc.offset == -1.5
        assert enc.min == enc.delta * -1.5
        assert enc.max == enc.delta * 1.5

    """
    When: Try to export encodings with zeroPointShift
    Then: Raise NotImplementedError
    """
    with pytest.raises(NotImplementedError):
        qc_op.export_encodings("1.0.0")

    """
    When: Try to import encodings with inconsistent zero_point_shift
    Then: Raise error
    """
    encoding_dict["zero_point_shift"][0] = 0.0
    with pytest.raises(RuntimeError):
        qc_op._load_encodings_dict(encoding_dict)

    """
    When: Loading encoding dict with no zero point shift
    Then: 1) Stored encodings contain integer offset
          2) quantizer.zeroPointShift should be 0.0
    """
    encoding_dict.pop("zero_point_shift")
    qc_op._load_encodings_dict(encoding_dict)
    assert qc_op._tensor_quantizer.getZeroPointShift() == 0.0

    for i, enc in enumerate(qc_op.get_encodings()):
        assert enc.delta == encoding_dict["per_block_int_scale"][i] * 3.0
        assert enc.min == enc.delta * -2
        assert enc.max == enc.delta * 1
        assert enc.offset == -2


@pytest.mark.parametrize(
    "qtype",
    [
        aimet_onnx.int2,
        aimet_onnx.int4,
        aimet_onnx.int8,
        aimet_onnx.int16,
        aimet_onnx.float16,
        "int2",
        "int4",
        "int8",
        "int16",
        "float16",
    ],
)
def test_set_precision(qtype: aimet_onnx.qtype | str):
    quant_info = libquant_info.QcQuantizeInfo()
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=8,
        use_symmetric_encodings=True,
    )

    assert qc_op.bitwidth == 8
    assert qc_op.data_type == QuantizationDataType.int

    qc_op.set_precision(qtype)

    if isinstance(qtype, str):
        qtype = aimet_onnx.qtype.from_string(qtype)

    dtype, bitwidth = qtype.to_legacy_repr()
    assert qc_op.data_type == dtype
    assert qc_op.bitwidth == bitwidth

    qc_op.update_encoding_stats(np.random.randn(10, 10).astype(np.float32))
    qc_op.compute_encodings()

    encoding_export = qc_op.export_encodings("2.0.0")

    if qtype in (aimet_onnx.float16,):
        assert not encoding_export
    else:
        assert "output_dtype" in encoding_export
        assert encoding_export["output_dtype"].removeprefix("u") == repr(qtype)

    qc_op.freeze_encodings()
    qc_op.set_precision(aimet_onnx.float16)

    assert qc_op.bitwidth == bitwidth
    assert qc_op.data_type == dtype


def test_set_fp8_e4m3fn_precision_computes_encodings():
    """
    set_precision("float8e4m3fn") must build an FP8-backed C++ quantizer which, unlike
    fp16, requires calibration before it can quantize-dequantize.
    """
    quant_info = libquant_info.QcQuantizeInfo()
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=8,
        use_symmetric_encodings=True,
    )

    qc_op.set_precision("float8e4m3fn")

    assert qc_op.precision() == aimet_onnx.float8e4m3fn
    assert qc_op.data_type == QuantizationDataType.float
    assert qc_op.bitwidth == 8
    assert not qc_op.quant_info.isIntDataType
    assert not qc_op.is_initialized()

    data = np.array([-7.25, -1.5, 0.0, 1.5, 7.25], dtype=np.float32)
    qc_op.update_encoding_stats(data)
    encodings = qc_op.compute_encodings()

    assert qc_op.is_initialized()
    assert qc_op.get_encodings()
    assert len(encodings) == 1
    assert encodings[0].bw == 8
    assert encodings[0].delta > 0.0
    assert encodings[0].offset == 0.0
    assert np.isclose(encodings[0].min, -encodings[0].delta * 448.0)
    assert np.isclose(encodings[0].max, encodings[0].delta * 448.0)

    # The FP8 grid must cover the observed amax. It may slightly exceed it, since the
    # scale is currently derived from the analyzer's range rather than raw amax stats.
    assert encodings[0].max >= 7.25
    assert np.isclose(encodings[0].max, 7.25, rtol=0.05)

    qdq_output = qc_op.quantize_dequantize(data)
    assert qdq_output.shape == data.shape
    assert qdq_output.dtype == data.dtype


def test_fp8_e4m3fn_precision_roundtrips_through_legacy_repr():
    """(float, 8) must resolve to float8e4m3fn so precision() needs no shadow state."""
    quant_info = libquant_info.QcQuantizeInfo()
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=8,
    )

    qc_op.set_precision(aimet_onnx.float8e4m3fn)
    assert qc_op.precision() == aimet_onnx.float8e4m3fn

    # Round tripping through the legacy (dtype, bitwidth) repr must be lossless
    dtype, bitwidth = aimet_onnx.float8e4m3fn.to_legacy_repr()
    assert aimet_onnx.qtype.from_legacy_repr(dtype, bitwidth) == aimet_onnx.float8e4m3fn

    # Switching away from and back to FP8 must land on an FP8-backed C++ quantizer
    qc_op.set_precision(aimet_onnx.int8)
    assert qc_op.precision() == aimet_onnx.int8
    qc_op.set_precision(aimet_onnx.float8e4m3fn)

    qc_op.update_encoding_stats(np.array([-1.0, 1.0], dtype=np.float32))
    (encoding,) = qc_op.compute_encodings()
    assert np.isclose(encoding.max / encoding.delta, 448.0)


def test_set_precision_does_not_rebuild_quantizer_when_unchanged():
    """Redundant set_precision calls must not discard calibration state."""
    quant_info = libquant_info.QcQuantizeInfo()
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=8,
    )
    qc_op.set_precision(aimet_onnx.int8)

    qc_op.update_encoding_stats(np.random.randn(64).astype(np.float32))
    qc_op.compute_encodings()
    assert qc_op.is_initialized()

    quantizer_before = qc_op._tensor_quantizer
    qc_op.set_precision(aimet_onnx.int8)

    assert qc_op._tensor_quantizer is quantizer_before
    assert qc_op.is_initialized()


@pytest.mark.parametrize(
    "precision, expected_max_representable",
    [("float8e4m3fn", 448.0), ("float8e5m2", 57344.0), ("int8", None)],
)
def test_encoding_constraints_are_honored_for_precision(
    precision, expected_max_representable
):
    """
    An encoding_constraints range must be reproduced exactly. For float precisions that
    means re-deriving the symmetric scale from the format's max representable value,
    rather than computing an affine integer delta (which the kernel reads as a scale).
    """
    quant_info = libquant_info.QcQuantizeInfo()
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.updateStats,
        bitwidth=8,
    )
    qc_op.set_precision(precision)
    qc_op._encoding_min_max_fixed_vals = (0.0, 1.0)

    qc_op.update_encoding_stats(np.random.rand(256).astype(np.float32))
    (encoding,) = qc_op.compute_encodings()

    if expected_max_representable is not None:
        # Grid must span exactly [-1, 1], i.e. scale == 1 / max_representable_value
        assert np.isclose(encoding.delta, 1.0 / expected_max_representable)
        assert np.isclose(encoding.max, 1.0)
    else:
        assert np.isclose(encoding.max, 1.0, atol=encoding.delta)


@pytest.mark.parametrize("precision", ["float8e4m3fn", "float8e5m2"])
def test_float_precision_is_read_back_from_quantizer(precision):
    """
    Float formats of equal width are indistinguishable via (data_type, bitwidth), so
    precision() must recover the exact format from the underlying C++ quantizer.
    """
    quant_info = libquant_info.QcQuantizeInfo()
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=8,
    )
    qc_op.set_precision(precision)

    assert qc_op.precision() == aimet_onnx.qtype.from_string(precision)
    assert str(qc_op.precision()) == precision
    assert qc_op.bitwidth == 8
    assert qc_op.data_type == QuantizationDataType.float

    qc_op.update_encoding_stats(np.array([-7.25, 7.25], dtype=np.float32))
    (encoding,) = qc_op.compute_encodings()
    assert encoding.offset == 0.0
    assert encoding.delta > 0.0


@pytest.mark.parametrize(
    "precision, reference_dtype, other_reference_dtype",
    [
        ("float8e4m3fn", ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2),
        ("float8e5m2", ml_dtypes.float8_e5m2, ml_dtypes.float8_e4m3fn),
    ],
)
def test_fp8_quantize_dequantize_lands_on_own_grid(
    precision, reference_dtype, other_reference_dtype
):
    """
    Quantize-dequantize must round onto the requested format's grid, bit for bit.

    The two FP8 formats differ in mantissa width (e4m3fn has 3 mantissa bits, e5m2 has
    2), so they round the same input differently. Encoding-level checks cannot catch a
    mix-up in the rounding itself, only in the max representable value, hence this
    comparison against ml_dtypes as an independent reference for each grid.
    """
    quant_info = libquant_info.QcQuantizeInfo()
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=8,
        use_symmetric_encodings=True,
    )
    qc_op.set_precision(precision)

    # Calibrate so that the scale is known, then quantize multiples of that scale. Values
    # are kept well inside the representable range so that saturation, which AIMET applies
    # but ml_dtypes does not, cannot influence the comparison.
    qc_op.update_encoding_stats(np.array([-8.0, 8.0], dtype=np.float32))
    (encoding,) = qc_op.compute_encodings()
    scale = np.float32(encoding.delta)

    grid_steps = np.array(
        [-3.7, -1.2, -1.1, -0.5, 0.0, 0.5, 1.1, 1.2, 3.7, 100.0], dtype=np.float32
    )
    data = grid_steps * scale
    qdq_output = qc_op.quantize_dequantize(data)

    expected = grid_steps.astype(reference_dtype).astype(np.float32) * scale
    assert np.array_equal(qdq_output, expected)

    # The other FP8 format rounds at least one of these values differently, so this
    # would fail if the requested format were ignored
    other = grid_steps.astype(other_reference_dtype).astype(np.float32) * scale
    assert not np.array_equal(qdq_output, other)


def test_one_shot_fp8_e4m3fn_quantize_dequantize_cpu():
    """
    End-to-end check that the ORT custom op routes FP8 through the C++ tensor quantizer:
    oneShotQuantizeDequantize must calibrate, advance the op mode, and alter the tensor.
    """
    input_arr = np.array([-7.25, -1.5, 0.0, 1.5, 7.25], dtype=np.float32)
    quant_info = libquant_info.QcQuantizeInfo()
    quant_node = helper.make_node(
        op_name,
        inputs=["input"],
        outputs=["output"],
        domain="aimet.customop.cpu",
        quant_info=libpymo.PtrToInt64(quant_info),
    )
    model = create_model_from_node(quant_node, input_arr.shape)
    session = build_session(model, ["CPUExecutionProvider"])
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=8,
        use_symmetric_encodings=True,
    )
    qc_op.set_precision("float8e4m3fn")

    output = session.run(None, {"input": input_arr})[0]

    assert output.shape == input_arr.shape
    assert output.dtype == input_arr.dtype
    assert qc_op.op_mode == OpMode.quantizeDequantize
    assert qc_op.is_initialized()
    assert qc_op.get_encodings()
    assert qc_op.get_encodings()[0].offset == 0.0
    assert qc_op.get_encodings()[0].bw == 8
    assert not np.array_equal(output, input_arr)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "precision, reference_dtype",
    [
        ("float8e4m3fn", ml_dtypes.float8_e4m3fn),
        ("float8e5m2", ml_dtypes.float8_e5m2),
    ],
)
def test_one_shot_fp8_quantize_dequantize_cpu_vs_gpu(precision, reference_dtype):
    """CPU and CUDA FP8 custom ops must agree across grid points and rounding ties."""
    positive_grid = (
        np.arange(1, 128, dtype=np.uint8).view(reference_dtype).astype(np.float32)
    )
    positive_grid = positive_grid[np.isfinite(positive_grid)]
    grid = np.concatenate(([0.0], positive_grid))
    midpoints = (grid[:-1] + grid[1:]) / 2
    input_arr = np.concatenate((grid, -grid, midpoints, -midpoints)).astype(np.float32)

    def quantize_dequantize(domain):
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(
            op_name,
            inputs=["input"],
            outputs=["output"],
            domain=domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=True,
        )
        qc_op.set_precision(precision)

        output = session.run(None, {"input": input_arr})[0]
        (encoding,) = qc_op.get_encodings()
        return output, encoding.delta

    cpu_output, cpu_scale = quantize_dequantize("aimet.customop.cpu")
    gpu_output, gpu_scale = quantize_dequantize("aimet.customop.cuda")

    # Including the format maximum in calibration pins the scale to one.
    assert cpu_scale == 1.0
    assert gpu_scale == cpu_scale
    assert np.array_equal(gpu_output, cpu_output)

    # Ensure the midpoint inputs exercised rounding rather than passing through.
    assert not np.array_equal(gpu_output, input_arr)


@pytest.mark.cuda
@pytest.mark.parametrize("precision", ["float8e4m3fn", "float8e5m2"])
def test_per_channel_fp8_quantize_dequantize_cpu_vs_gpu(precision):
    """CPU and CUDA per-channel FP8 custom ops must agree bit for bit."""
    input_shape = (12, 6, 3, 3)
    quant_axis = 0
    input_arr = np.random.RandomState(0).randn(*input_shape).astype(np.float32)

    def quantize_dequantize(domain):
        quant_info = libquant_info.QcQuantizeInfo()
        qc_op = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=QuantScheme.post_training_tf,
            op_mode=OpMode.oneShotQuantizeDequantize,
            bitwidth=8,
            use_symmetric_encodings=True,
            tensor_quantizer_params=TensorQuantizerParams(
                input_shape, quant_axis, None
            ),
        )
        qc_op.set_precision(precision)
        qc_op.enable_per_channel_quantization()

        quant_node = helper.make_node(
            per_channel_op_name,
            inputs=["input"],
            outputs=["output"],
            domain=domain,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        model = create_model_from_node(quant_node, input_shape)
        session = build_session(model, available_providers)

        output = session.run(None, {"input": input_arr})[0]
        return output, [encoding.delta for encoding in qc_op.get_encodings()]

    cpu_output, cpu_scales = quantize_dequantize("aimet.customop.cpu")
    gpu_output, gpu_scales = quantize_dequantize("aimet.customop.cuda")

    assert len(cpu_scales) == input_shape[quant_axis]
    assert gpu_scales == cpu_scales
    assert np.array_equal(gpu_output, cpu_output)
    assert not np.array_equal(gpu_output, input_arr)


@pytest.mark.parametrize("qtype", ["int0", "uint8", "float8e5m2fnuz", "float8"])
def test_set_invalid_precision_raises(qtype: str):
    quant_info = libquant_info.QcQuantizeInfo()
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=8,
        use_symmetric_encodings=True,
    )
    with pytest.raises(ValueError):
        qc_op.set_precision(qtype)


def _assert_is_qspec(quantizer: QcQuantizeOp, spec: QSpec):
    assert quantizer.precision() == spec.dtype
    assert quantizer.use_symmetric_encodings == spec.symmetric
    zp_shift = 0.5 if spec.shift_zero_point else 0.0
    assert quantizer.get_zero_point_shift() == zp_shift

    if isinstance(spec.granularity, aimet_onnx.defs.PerTensor):
        assert quantizer._encoding_type() == EncodingType.PER_TENSOR
    if isinstance(spec.granularity, aimet_onnx.defs.PerChannel):
        assert quantizer._encoding_type() == EncodingType.PER_CHANNEL
    if isinstance(spec.granularity, aimet_onnx.defs.LPBQ):
        assert quantizer._encoding_type() == EncodingType.LPBQ
        assert quantizer.quant_info.blockSize == spec.granularity.block_size
        assert quantizer._scale_quantizer.scale_bits == spec.granularity.scale_bits
    elif isinstance(spec.granularity, aimet_onnx.defs.Blockwise):
        assert quantizer._encoding_type() == EncodingType.PER_BLOCK
        assert quantizer.quant_info.blockSize == spec.granularity.block_size

    if not isinstance(spec.granularity, aimet_onnx.defs.LPBQ):
        assert quantizer._scale_quantizer is None


@pytest.mark.parametrize(
    "qspec",
    [
        QSpec.lpbq("int4", 64, 4),
        QSpec.blockwise("int2", 128, symmetric=True, shift_zero_point=True),
        QSpec.per_channel("int8"),
    ],
)
def test_set_retain_qspec_on_error(qspec):
    original_spec = QSpec.per_tensor("int8", symmetric=False, shift_zero_point=False)
    quant_info = libquant_info.QcQuantizeInfo()
    qc_op = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.oneShotQuantizeDequantize,
        bitwidth=original_spec.dtype.bits,
        use_symmetric_encodings=original_spec.symmetric,
        tensor_quantizer_params=None,
    )

    with pytest.raises(RuntimeError):
        qc_op.set_qspec(qspec)

    # Original quantizer state should be kept
    _assert_is_qspec(qc_op, original_spec)


@pytest.mark.parametrize(
    "qspec",
    [
        QSpec.lpbq("int4", 16, 4),
        QSpec.blockwise("int2", 8, symmetric=True, shift_zero_point=True),
        QSpec.blockwise("int8", 8, symmetric=True, shift_zero_point=False),
        QSpec.per_channel("int16", symmetric=False),
        QSpec.per_tensor("int4", symmetric=True),
        QSpec.per_tensor("int8", symmetric=False),
        QSpec.per_tensor("float16", symmetric=False),
    ],
)
def test_set_qspec(qspec):
    input_shape = (32, 32)
    orig = QSpec.per_tensor("int8", symmetric=False, shift_zero_point=False)
    quant_info = libquant_info.QcQuantizeInfo()
    tensor_quantizer_params = TensorQuantizerParams(input_shape, 0, 1)
    quantizer = QcQuantizeOp(
        quant_info=quant_info,
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.updateStats,
        bitwidth=orig.dtype.bits,
        use_symmetric_encodings=orig.symmetric,
        tensor_quantizer_params=tensor_quantizer_params,
    )

    quantizer.set_qspec(qspec)
    _assert_is_qspec(quantizer, qspec)
    quantizer.set_qspec(orig)
    _assert_is_qspec(quantizer, orig)

    # When: Set partial QSpec
    # Then: Only specified parameters should be changed

    partial_qspec = QSpec(qspec.dtype, None, None)
    quantizer.set_qspec(partial_qspec)
    _assert_is_qspec(quantizer, QSpec(qspec.dtype, orig.granularity, orig.symmetric))
    quantizer.set_qspec(orig)

    # LPBQ requires dtype and symmetry to be specified
    if not isinstance(qspec.granularity, aimet_onnx.defs.LPBQ):
        partial_qspec = QSpec(None, qspec.granularity, None)
        quantizer.set_qspec(partial_qspec)
        _assert_is_qspec(
            quantizer, QSpec(orig.dtype, qspec.granularity, orig.symmetric)
        )
        quantizer.set_qspec(orig)

    partial_qspec = QSpec(None, None, qspec.symmetric)
    quantizer.set_qspec(partial_qspec)
    _assert_is_qspec(quantizer, QSpec(orig.dtype, orig.granularity, qspec.symmetric))


def _calibrate_and_qdq(quantizer: QcQuantizeOp, tensor: np.ndarray) -> np.ndarray:
    quantizer.op_mode = OpMode.updateStats
    quantizer.update_encoding_stats(tensor)
    quantizer.compute_encodings()
    quantizer.op_mode = OpMode.quantizeDequantize
    return quantizer.quantize_dequantize(tensor)


def _new_quantizer(input_shape, **kwargs) -> QcQuantizeOp:
    return QcQuantizeOp(
        quant_info=libquant_info.QcQuantizeInfo(),
        quant_scheme=QuantScheme.post_training_tf,
        op_mode=OpMode.updateStats,
        tensor_quantizer_params=TensorQuantizerParams(input_shape, 0, 1),
        **kwargs,
    )


def test_set_qspec_per_tensor_matches_reference():
    """
    set_qspec should produce identical QDQ output to a quantizer configured
    directly through the pre-existing API.
    """
    input_shape = (32, 32)
    input_tensor = np.random.randn(*input_shape).astype(np.float32)

    configured = _new_quantizer(input_shape, bitwidth=8, use_symmetric_encodings=True)
    configured.set_qspec(QSpec.per_tensor("int8", symmetric=False))

    reference = _new_quantizer(input_shape, bitwidth=8, use_symmetric_encodings=False)

    actual = _calibrate_and_qdq(configured, input_tensor)
    expected = _calibrate_and_qdq(reference, input_tensor)

    assert np.array_equal(actual, expected)


def test_set_qspec_per_channel_output():
    input_shape = (32, 32)
    input_tensor = np.random.randn(*input_shape).astype(np.float32)

    configured = _new_quantizer(input_shape, bitwidth=8, use_symmetric_encodings=False)
    configured.set_qspec(QSpec.per_channel("int8", symmetric=True))

    reference = _new_quantizer(input_shape, bitwidth=8, use_symmetric_encodings=True)
    reference.enable_per_channel_quantization(True)

    actual = _calibrate_and_qdq(configured, input_tensor)
    expected = _calibrate_and_qdq(reference, input_tensor)

    assert np.array_equal(actual, expected)


def test_set_qspec_blockwise_output():
    input_shape = (32, 32)
    input_tensor = np.random.randn(*input_shape).astype(np.float32)

    configured = _new_quantizer(input_shape, bitwidth=8, use_symmetric_encodings=False)
    configured.set_qspec(QSpec.blockwise("int8", 8, symmetric=True))

    reference = _new_quantizer(input_shape, bitwidth=8, use_symmetric_encodings=True)
    reference.enable_per_channel_quantization(True)
    reference._enable_blockwise_quantization(8)

    actual = _calibrate_and_qdq(configured, input_tensor)
    expected = _calibrate_and_qdq(reference, input_tensor)

    assert np.array_equal(actual, expected)


def test_set_qspec_lpbq_output():
    input_shape = (32, 32)
    input_tensor = np.random.randn(*input_shape).astype(np.float32)

    configured = _new_quantizer(input_shape, bitwidth=8, use_symmetric_encodings=True)
    configured.set_qspec(QSpec.lpbq("int4", 16, 4))

    reference = _new_quantizer(input_shape, bitwidth=4, use_symmetric_encodings=True)
    reference._enable_blockwise_quantization(16)
    reference._scale_quantizer = LPBQScaleQuantizer(4)
    reference.data_type = QuantizationDataType.int

    actual = _calibrate_and_qdq(configured, input_tensor)
    expected = _calibrate_and_qdq(reference, input_tensor)

    assert np.array_equal(actual, expected)


def test_set_granularity():
    quantizer = _new_quantizer((32, 32), use_symmetric_encodings=True)
    assert not quantizer.quant_info.usePerChannelMode

    quantizer._set_granularity(PerChannel())
    assert quantizer.quant_info.usePerChannelMode

    quantizer._set_granularity(PerTensor())
    assert not quantizer.quant_info.usePerChannelMode

    quantizer._set_granularity(Blockwise(8))
    assert quantizer.quant_info.usePerChannelMode
    assert quantizer.quant_info.blockSize == 8
    assert quantizer.quant_info.blockAxis == 1

    quantizer._set_granularity(PerChannel())
    assert quantizer.quant_info.usePerChannelMode
    assert quantizer.quant_info.blockSize == 0

    quantizer._set_granularity(LPBQ(8, 4))
    assert quantizer.quant_info.usePerChannelMode
    assert quantizer.quant_info.blockSize == 8
    assert quantizer.quant_info.blockAxis == 1
    assert quantizer._scale_quantizer == LPBQScaleQuantizer(4)

    quantizer._set_granularity(Blockwise(16))
    assert quantizer._scale_quantizer is None
    assert quantizer.quant_info.blockSize == 16


def test_load_encodings_with_different_channel_axis():
    """
    Given: A per-channel quantizer with channel axis 1
    When: Loading per-channel encodings with channel axis 0
    Then: The loaded axis should be used for quantize-dequantize and export
    """
    qc_op = _new_quantizer((2, 2), bitwidth=8)
    qc_op.tensor_quantizer_params.channel_axis = 1
    qc_op.enable_per_channel_quantization()
    assert qc_op.quant_info.channelAxis == 1

    qc_op._load_encodings_dict(
        {"name": "w", "axis": 0, "output_dtype": "int8", "y_scale": [0.1, 1.0]}
    )
    assert qc_op.quant_info.channelAxis == 0
    assert qc_op.export_encodings("2.0.0")["axis"] == 0

    out = qc_op.quantize_dequantize(np.full((2, 2), 0.4, dtype=np.float32))
    assert np.allclose(out, [[0.4, 0.4], [0.0, 0.0]])


def test_load_encodings_with_different_block_axis():
    """
    Given: A blockwise quantizer with block axis 1
    When: Loading blockwise encodings with the same block size but block axis 0
    Then: The loaded axes should be used for quantize-dequantize and export
    """
    qc_op = _new_quantizer((4, 4), bitwidth=8)
    qc_op._enable_blockwise_quantization(2)
    assert qc_op._encoding_shape() == (4, 2)

    scale = np.array([[0.1, 0.1, 0.1, 0.1], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    encoding = AffineEncoding(
        scale=scale,
        offset=np.zeros_like(scale),
        dtype="int8",
        channel_axis=1,
        block_axis=0,
        block_size=2,
    )
    encoding.load_to(qc_op)
    assert qc_op.quant_info.channelAxis == 1
    assert qc_op.quant_info.blockAxis == 0
    assert qc_op._encoding_shape() == (2, 4)

    out = qc_op.quantize_dequantize(np.full((4, 4), 0.4, dtype=np.float32))
    assert np.allclose(out, [[0.4] * 4] * 2 + [[0.0] * 4] * 2)


def test_enable_per_channel_with_equivalent_axis_retains_encodings():
    """
    Given: A per-channel quantizer with computed encodings
    When: Re-enabling per-channel quantization with an equivalent negative channel axis
    Then: The existing encodings should be retained
    """
    qc_op = _new_quantizer((3, 20), bitwidth=8)
    qc_op.enable_per_channel_quantization()
    _calibrate_and_qdq(qc_op, np.random.randn(3, 20).astype(np.float32))
    scale_before = qc_op._get_scale()

    qc_op.tensor_quantizer_params.channel_axis = -2
    qc_op.enable_per_channel_quantization()
    assert qc_op.is_initialized()
    assert np.array_equal(qc_op._get_scale(), scale_before)
