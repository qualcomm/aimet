# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


from typing import Optional
import pytest
import numpy as np

try:
    import onnx_ir
    import onnxruntime as ort

    try:
        from aimet_onnx.common.onnx.opset21 import (
            QuantizeLinear,
            DequantizeLinear,
        )
    except ImportError:
        from aimet_torch.common.onnx.opset21 import (
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

    class TestOpset21:
        @pytest.mark.parametrize("zero_point", [0, 1])
        @pytest.mark.parametrize(
            "axis, block_size",
            [
                (None, None),
                (0, None),
                (0, 32),
                (1, 32),
            ],
        )
        @pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16"])
        def test_QuantizeLinear_success(
            self, dtype: str, axis: Optional[int], block_size: Optional[int], zero_point
        ):
            input = np.arange(-128, 256, dtype=np.float32)
            scale = np.ones((), dtype=np.float32)
            zero_point = np.array(zero_point, dtype=np.int64)
            qmin = np.iinfo(dtype).min
            qmax = np.iinfo(dtype).max
            expected_output = (input + zero_point).clip(qmin, qmax).astype(dtype)

            if axis is not None:
                block_axis = axis if block_size else None
                channel_axis = axis if block_axis is None else 1 - block_axis
                input = input.repeat(3).reshape(
                    *input.shape[:channel_axis], -1, *input.shape[channel_axis:]
                )
                scale = scale.repeat(3)
                zero_point = zero_point.repeat(3)
                expected_output = expected_output.repeat(3).reshape(
                    *expected_output.shape[:channel_axis],
                    -1,
                    *expected_output.shape[channel_axis:],
                )

                if block_axis is not None:
                    assert block_size is not None
                    assert input.shape[block_axis] % block_size == 0
                    num_blocks = input.shape[block_axis] // block_size
                    scale_shape = tuple(
                        -1 if axis == channel_axis else 1 for axis in range(input.ndim)
                    )

                    scale = scale.reshape(*scale_shape)
                    scale = scale.repeat(num_blocks, axis=block_axis)

                    zero_point = zero_point.reshape(*scale_shape)
                    zero_point = zero_point.repeat(num_blocks, axis=block_axis)

            """
            When: Create onnx model with opset21 QuantizeLinear
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
                block_size=block_size,
            )
            model = _make_model(
                21, input_value, output_value, [scale_value, zero_point_value], [node]
            )

            """
            Then: Model should be runnable on ORT and should produce correct output
            """
            sess = ort.InferenceSession(model.SerializeToString())
            (output,) = sess.run(None, {"input": input})
            assert np.all(output == expected_output)

        @pytest.mark.parametrize("dtype", ["int32", "uint32"])
        def test_QuantizeLinear_unsupported_dtype(self, dtype):
            """
            When: Create opset21 QuantizeLinear with unsupported dtypes
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
                (None, 32),
            ],
        )
        def test_QuantizeLinear_invalid_pcq(self, axis, block_size):
            """
            When: Create opset21 QuantizeLinear with invalid axis/block_size
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
        @pytest.mark.parametrize(
            "axis, block_size",
            [
                (None, None),
                (0, None),
                (0, 32),
                (1, 32),
            ],
        )
        @pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32"])
        def test_DequantizeLinear_success(
            self, dtype: str, axis: Optional[int], block_size: Optional[int], zero_point
        ):
            if dtype == "int32" and zero_point != 0:
                pytest.skip(
                    reason="ORT requires zero_point=0 for int32 DequantizeLinear"
                )

            qmin = np.iinfo(dtype).min
            qmax = np.iinfo(dtype).max
            input = np.arange(-128, 256).clip(qmin, qmax).astype(dtype)

            scale = np.ones((), dtype=np.float32)
            zero_point = np.array(zero_point, dtype=np.int64)
            expected_output = (input.astype(np.int64) - zero_point).astype(np.float32)

            if axis is not None:
                block_axis = axis if block_size else None
                channel_axis = axis if block_axis is None else 1 - block_axis
                input = input.repeat(3).reshape(
                    *input.shape[:channel_axis], -1, *input.shape[channel_axis:]
                )
                scale = scale.repeat(3)
                zero_point = zero_point.repeat(3)
                expected_output = expected_output.repeat(3).reshape(
                    *expected_output.shape[:channel_axis],
                    -1,
                    *expected_output.shape[channel_axis:],
                )

                if block_axis is not None:
                    assert block_size is not None
                    assert input.shape[block_axis] % block_size == 0
                    num_blocks = input.shape[block_axis] // block_size
                    scale_shape = tuple(
                        -1 if axis == channel_axis else 1 for axis in range(input.ndim)
                    )

                    scale = scale.reshape(*scale_shape)
                    scale = scale.repeat(num_blocks, axis=block_axis)

                    zero_point = zero_point.reshape(*scale_shape)
                    zero_point = zero_point.repeat(num_blocks, axis=block_axis)

            """
            When: Create onnx model with opset21 DequantizeLinear
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
                block_size=block_size,
            )
            model = _make_model(
                21, input_value, output_value, [scale_value, zero_point_value], [node]
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
                ("int32", 1),  # int32 DQ requires zero_point=0
                ("uint32", 0),
            ],
        )
        def test_DequantizeLinear_unsupported_dtype1(self, dtype, zero_point):
            """
            When: Create opset21 DequantizeLinear.make_zero_point with unsupported dtypes
            Then: Throw runtime error
            """
            with pytest.raises(RuntimeError):
                _ = DequantizeLinear.make_zero_point(
                    np.array(zero_point, dtype=np.int64), dtype=dtype, name="zero_point"
                )

        @pytest.mark.parametrize("dtype", ["uint32"])
        def test_DequantizeLinear_unsupported_dtype2(self, dtype):
            """
            When: Create opset21 DequantizeLinear.make_node with unsupported dtypes
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
                (None, 32),
            ],
        )
        def test_DequantizeLinear_invalid_pcq(self, axis, block_size):
            """
            When: Create opset21 DequantizeLinear with invalid axis/block_size
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
        @pytest.mark.parametrize(
            "axis, block_size",
            [
                (None, None),
                (0, None),
                (0, 32),
                (1, 32),
            ],
        )
        @pytest.mark.parametrize(
            "dtype", ["int4", "uint4", "int8", "uint8", "int16", "uint16"]
        )
        def test_QuantizeDequantize_success(
            self, dtype: str, axis: Optional[int], block_size: Optional[int], zero_point
        ):
            input = np.arange(-128, 256, dtype=np.float32)
            scale = np.ones((), dtype=np.float32)
            zero_point = np.array(zero_point, dtype=np.int64)
            qmin = (
                -8
                if dtype == "int4"
                else 0
                if dtype == "uint4"
                else np.iinfo(dtype).min
            )
            qmax = (
                7
                if dtype == "int4"
                else 15
                if dtype == "uint4"
                else np.iinfo(dtype).max
            )
            expected_output = (input + zero_point).clip(qmin, qmax) - zero_point

            if axis is not None:
                block_axis = axis if block_size else None
                channel_axis = axis if block_axis is None else 1 - block_axis
                input = input.repeat(3).reshape(
                    *input.shape[:channel_axis], -1, *input.shape[channel_axis:]
                )
                scale = scale.repeat(3)
                zero_point = zero_point.repeat(3)
                expected_output = expected_output.repeat(3).reshape(
                    *expected_output.shape[:channel_axis],
                    -1,
                    *expected_output.shape[channel_axis:],
                )

                if block_axis is not None:
                    assert block_size is not None
                    assert input.shape[block_axis] % block_size == 0
                    num_blocks = input.shape[block_axis] // block_size
                    scale_shape = tuple(
                        -1 if axis == channel_axis else 1 for axis in range(input.ndim)
                    )

                    scale = scale.reshape(*scale_shape)
                    scale = scale.repeat(num_blocks, axis=block_axis)

                    zero_point = zero_point.reshape(*scale_shape)
                    zero_point = zero_point.repeat(num_blocks, axis=block_axis)

            """
            When: Create onnx model with opset21 QuantizeLinear - DequantizeLinear
            """
            input_value = onnx_ir.Value(
                name="input",
                type=onnx_ir.TensorType(onnx_ir.DataType.FLOAT),
                shape=onnx_ir.Shape(input.shape),
            )
            input_q_value = onnx_ir.Value(name="input_q")
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
                block_size=block_size,
            )
            dq_node = DequantizeLinear.make_node(
                name="DequantizeLinear",
                inputs=[input_q_value, scale_value, zero_point_value],
                output=output_value,
                dtype=dtype,
                axis=axis,
                block_size=block_size,
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
