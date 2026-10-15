# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


from typing import Optional
import pytest
import numpy as np

try:
    import onnx_ir
    import onnxruntime as ort

    try:
        from aimet_onnx.common.onnx.opset13 import (
            QuantizeLinear,
            DequantizeLinear,
        )
    except ImportError:
        from aimet_torch.common.onnx.opset13 import (
            QuantizeLinear,
            DequantizeLinear,
        )
except ImportError:
    pass
else:

    def _make_model(opset_version, input_value, output_value, initializers, nodes):
        graph = onnx_ir.Graph(
            inputs=[input_value],
            outputs=[output_value],
            nodes=nodes,
            initializers=initializers,
            opset_imports={"": opset_version},
            name="graph",
        )
        ir_model = onnx_ir.Model(graph, ir_version=10)
        return onnx_ir.to_proto(ir_model)

    @pytest.mark.parametrize("zero_point", [0, 1])
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", ["int8", "uint8"])
    def test_QuantizeLinear_success(dtype: str, axis: Optional[int], zero_point):
        input = np.arange(-128, 256, dtype=np.float32)
        scale = np.ones((), dtype=np.float32)
        zero_point = np.array(zero_point, dtype=np.int64)
        qmin = np.iinfo(dtype).min
        qmax = np.iinfo(dtype).max
        expected_output = (input + zero_point).clip(qmin, qmax).astype(dtype)

        if axis is not None:
            input = input.repeat(3).reshape(
                *input.shape[:axis], -1, *input.shape[axis:]
            )
            scale = scale.repeat(3)
            zero_point = zero_point.repeat(3)
            expected_output = expected_output.repeat(3).reshape(
                *expected_output.shape[:axis], -1, *expected_output.shape[axis:]
            )

        """
        When: Create onnx model with opset13 QuantizeLinear
        """
        input_value = onnx_ir.Value(
            name="input",
            type=onnx_ir.TensorType(onnx_ir.DataType.FLOAT),
            shape=onnx_ir.Shape(input.shape),
        )
        output_value = onnx_ir.Value(
            name="output",
            type=onnx_ir.TensorType(getattr(onnx_ir.DataType, dtype.upper())),
            shape=onnx_ir.Shape(expected_output.shape),
        )
        scale_value = onnx_ir.Value(
            name="scale", const_value=onnx_ir.tensor(scale, name="scale")
        )
        zero_point_value = QuantizeLinear.make_zero_point(
            zero_point, dtype=dtype, name="zero_point"
        )
        node = QuantizeLinear.make_node(
            name="QuantizeLinear",
            inputs=[input_value, scale_value, zero_point_value],
            output=output_value,
            dtype=dtype,
            axis=axis,
            block_size=None,
        )
        model = _make_model(
            13, input_value, output_value, [scale_value, zero_point_value], [node]
        )

        """
        Then: Model should be runnable on ORT and should produce correct output
        """
        sess = ort.InferenceSession(model.SerializeToString())
        (output,) = sess.run(None, {"input": input})
        assert np.all(output == expected_output)

    @pytest.mark.parametrize(
        "dtype", ["int4", "uint4", "int16", "uint16", "int32", "uint32"]
    )
    def test_QuantizeLinear_unsupported_dtype(dtype):
        """
        When: Create opset13 QuantizeLinear with unsupported dtypes
        Then: Throw runtime error
        """
        with pytest.raises(RuntimeError):
            _ = QuantizeLinear.make_zero_point(
                np.zeros(()), dtype=dtype, name="zero_point"
            )

        with pytest.raises(RuntimeError):
            _ = QuantizeLinear.make_node(
                name="QuantizeLinear",
                inputs=[
                    onnx_ir.Value(name="input"),
                    onnx_ir.Value(name="scale"),
                    onnx_ir.Value(name="zero_point"),
                ],
                output=onnx_ir.Value(name="output"),
                dtype=dtype,
                axis=None,
                block_size=None,
            )

    @pytest.mark.parametrize(
        "axis, block_size",
        [
            (0, 32),
            (None, 32),
        ],
    )
    def test_QuantizeLinear_unsupported_pcq(axis, block_size):
        """
        When: Create opset13 QuantizeLinear with axis/block_size
        Then: Throw runtime error
        """
        with pytest.raises(RuntimeError):
            _ = QuantizeLinear.make_node(
                name="QuantizeLinear",
                inputs=[
                    onnx_ir.Value(name="input"),
                    onnx_ir.Value(name="scale"),
                    onnx_ir.Value(name="zero_point"),
                ],
                output=onnx_ir.Value(name="output"),
                dtype="int8",
                axis=axis,
                block_size=block_size,
            )

    @pytest.mark.parametrize("zero_point", [0, 1])
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", ["int8", "uint8", "int32"])
    def test_DequantizeLinear_success(dtype: str, axis: Optional[int], zero_point):
        if dtype == "int32" and zero_point != 0:
            pytest.skip(reason="ORT requires zero_point=0 for int32 DequantizeLinear")

        qmin = np.iinfo(dtype).min
        qmax = np.iinfo(dtype).max
        input = np.arange(-128, 256).clip(qmin, qmax).astype(dtype)

        scale = np.ones((), dtype=np.float32)
        zero_point = np.array(zero_point, dtype=np.int64)
        expected_output = (input.astype(np.int64) - zero_point).astype(np.float32)

        if axis is not None:
            input = input.repeat(3).reshape(
                *input.shape[:axis], -1, *input.shape[axis:]
            )
            scale = scale.repeat(3)
            zero_point = zero_point.repeat(3)
            expected_output = expected_output.repeat(3).reshape(
                *expected_output.shape[:axis], -1, *expected_output.shape[axis:]
            )

        """
        When: Create onnx model with opset13 DequantizeLinear
        """
        input_value = onnx_ir.Value(
            name="input",
            type=onnx_ir.TensorType(getattr(onnx_ir.DataType, dtype.upper())),
            shape=onnx_ir.Shape(input.shape),
        )
        output_value = onnx_ir.Value(
            name="output",
            type=onnx_ir.TensorType(onnx_ir.DataType.FLOAT),
            shape=onnx_ir.Shape(expected_output.shape),
        )
        scale_value = onnx_ir.Value(
            name="scale", const_value=onnx_ir.tensor(scale, name="scale")
        )
        zero_point_value = DequantizeLinear.make_zero_point(
            zero_point, dtype=dtype, name="zero_point"
        )
        node = DequantizeLinear.make_node(
            name="DequantizeLinear",
            inputs=[input_value, scale_value, zero_point_value],
            output=output_value,
            dtype=dtype,
            axis=axis,
            block_size=None,
        )
        model = _make_model(
            13, input_value, output_value, [scale_value, zero_point_value], [node]
        )

        """
        Then: Model should be runnable on ORT and should produce correct output
        """
        sess = ort.InferenceSession(model.SerializeToString())
        (output,) = sess.run(None, {"input": input})
        assert np.all(output == expected_output)

    @pytest.mark.parametrize(
        "dtype,    zero_point",
        [
            ("int4", 0),
            ("uint4", 0),
            ("int16", 0),
            ("uint16", 0),
            ("int32", 1),
            ("uint32", 0),
        ],
    )
    def test_DequantizeLinear_unsupported_dtype1(dtype, zero_point):
        """
        When: Create opset13 DequantizeLinear.make_zero_point with unsupported dtypes
        Then: Throw runtime error
        """
        with pytest.raises(RuntimeError):
            _ = DequantizeLinear.make_zero_point(
                np.array(zero_point, dtype=np.int64), dtype=dtype, name="zero_point"
            )

    @pytest.mark.parametrize("dtype", ["int4", "uint4", "int16", "uint16", "uint32"])
    def test_DequantizeLinear_unsupported_dtype2(dtype):
        """
        When: Create opset13 DequantizeLinear.make_node with unsupported dtypes
        Then: Throw runtime error
        """
        with pytest.raises(RuntimeError):
            _ = DequantizeLinear.make_node(
                name="QuantizeLinear",
                inputs=[
                    onnx_ir.Value(name="input"),
                    onnx_ir.Value(name="scale"),
                    onnx_ir.Value(name="zero_point"),
                ],
                output=onnx_ir.Value(name="output"),
                dtype=dtype,
                axis=None,
                block_size=None,
            )

    @pytest.mark.parametrize(
        "axis, block_size",
        [
            (0, 32),
            (None, 32),
        ],
    )
    def test_DequantizeLinear_unsupported_pcq(axis, block_size):
        """
        When: Create opset13 DequantizeLinear with axis/block_size
        Then: Throw runtime error
        """
        with pytest.raises(RuntimeError):
            _ = DequantizeLinear.make_node(
                name="DequantizeLinear",
                inputs=[
                    onnx_ir.Value(name="input"),
                    onnx_ir.Value(name="scale"),
                    onnx_ir.Value(name="zero_point"),
                ],
                output=onnx_ir.Value(name="output"),
                dtype="int8",
                axis=axis,
                block_size=block_size,
            )

    @pytest.mark.parametrize("zero_point", [0, 1])
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", ["int8", "uint8"])
    def test_QuantizeDequantize_success(dtype: str, axis: Optional[int], zero_point):
        input = np.arange(-128, 256, dtype=np.float32)
        scale = np.ones((), dtype=np.float32)
        zero_point = np.array(zero_point, dtype=np.int64)
        qmin = np.iinfo(dtype).min
        qmax = np.iinfo(dtype).max
        expected_output = (input + zero_point).clip(qmin, qmax) - zero_point

        if axis is not None:
            input = input.repeat(3).reshape(
                *input.shape[:axis], -1, *input.shape[axis:]
            )
            scale = scale.repeat(3)
            zero_point = zero_point.repeat(3)
            expected_output = expected_output.repeat(3).reshape(
                *expected_output.shape[:axis], -1, *expected_output.shape[axis:]
            )

        """
        When: Create onnx model with opset21 QuantizeLinear - DequantizeLinear
        """
        input_value = onnx_ir.Value(
            name="input",
            type=onnx_ir.TensorType(onnx_ir.DataType.FLOAT),
            shape=onnx_ir.Shape(input.shape),
        )
        input_q_value = onnx_ir.Value(
            name="input_q",
            type=onnx_ir.TensorType(getattr(onnx_ir.DataType, dtype.upper())),
            shape=onnx_ir.Shape(input.shape),
        )
        output_value = onnx_ir.Value(
            name="output",
            type=onnx_ir.TensorType(onnx_ir.DataType.FLOAT),
            shape=onnx_ir.Shape(expected_output.shape),
        )
        scale_value = onnx_ir.Value(
            name="scale", const_value=onnx_ir.tensor(scale, name="scale")
        )
        zero_point_value = QuantizeLinear.make_zero_point(
            zero_point, dtype=dtype, name="zero_point"
        )
        q_node = QuantizeLinear.make_node(
            name="QuantizeLinear",
            inputs=[input_value, scale_value, zero_point_value],
            output=input_q_value,
            dtype=dtype,
            axis=axis,
            block_size=None,
        )
        dq_node = DequantizeLinear.make_node(
            name="DequantizeLinear",
            inputs=[input_q_value, scale_value, zero_point_value],
            output=output_value,
            dtype=dtype,
            axis=axis,
            block_size=None,
        )
        model = _make_model(
            21,
            input_value,
            output_value,
            [scale_value, zero_point_value],
            [q_node, dq_node],
        )

        """
        Then: Model should be runnable on ORT and should produce correct output
        """
        sess = ort.InferenceSession(model.SerializeToString())
        (output,) = sess.run(None, {"input": input})
        assert np.all(output == expected_output)
