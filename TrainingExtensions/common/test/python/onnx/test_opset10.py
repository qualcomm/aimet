# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import numpy as np

try:
    import onnx_ir
    import onnxruntime as ort

    try:
        from aimet_onnx.common.onnx.opset10 import (
            QuantizeLinear,
            DequantizeLinear,
            pack_int8_to_int4x2,
            unpack_int4x2_to_int8,
        )
    except ImportError:
        from aimet_torch.common.onnx.opset10 import (
            QuantizeLinear,
            DequantizeLinear,
            pack_int8_to_int4x2,
            unpack_int4x2_to_int8,
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
    @pytest.mark.parametrize("dtype", ["int8", "uint8"])
    def test_QuantizeLinear_success(dtype: str, zero_point):
        input = np.arange(-128, 256, dtype=np.float32)
        scale = np.ones((), dtype=np.float32)
        zero_point = np.array(zero_point, dtype=np.int64)
        qmin = np.iinfo(dtype).min
        qmax = np.iinfo(dtype).max
        expected_output = (input + zero_point).clip(qmin, qmax).astype(dtype)

        """
        When: Create onnx model with opset10 QuantizeLinear
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
            axis=None,
            block_size=None,
        )
        model = _make_model(
            10, input_value, output_value, [scale_value, zero_point_value], [node]
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
        When: Create opset10 QuantizeLinear with unsupported dtypes
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
            (0, None),
            (0, 32),
            (None, 32),
        ],
    )
    def test_QuantizeLinear_unsupported_pcq(axis, block_size):
        """
        When: Create opset10 QuantizeLinear with axis/block_size
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
    @pytest.mark.parametrize("dtype", ["int8", "uint8", "int32"])
    def test_DequantizeLinear_success(dtype: str, zero_point):
        if dtype == "int32" and zero_point != 0:
            pytest.skip(reason="ORT requires zero_point=0 for int32 DequantizeLinear")

        qmin = np.iinfo(dtype).min
        qmax = np.iinfo(dtype).max
        input = np.arange(-128, 256).clip(qmin, qmax).astype(dtype)

        scale = np.ones((), dtype=np.float32)
        zero_point = np.array(zero_point, dtype=np.int64)
        expected_output = (input.astype(np.int64) - zero_point).astype(np.float32)

        """
        When: Create onnx model with opset10 DequantizeLinear
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
            axis=None,
            block_size=None,
        )
        model = _make_model(
            10, input_value, output_value, [scale_value, zero_point_value], [node]
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
            ("int32", 1),  # int32 DQ requires zero_point=0
            ("uint32", 0),
        ],
    )
    def test_DequantizeLinear_unsupported_dtype1(dtype, zero_point):
        """
        When: Create opset10 DequantizeLinear.make_zero_point with unsupported dtypes
        Then: Throw runtime error
        """
        with pytest.raises(RuntimeError):
            _ = DequantizeLinear.make_zero_point(
                np.array(zero_point, dtype=np.int64), dtype=dtype, name="zero_point"
            )

    @pytest.mark.parametrize("dtype", ["int4", "uint4", "int16", "uint16", "uint32"])
    def test_DequantizeLinear_unsupported_dtype2(dtype):
        """
        When: Create opset10 DequantizeLinear.make_node with unsupported dtypes
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
            (0, None),
            (0, 32),
            (None, 32),
        ],
    )
    def test_DequantizeLinear_unsupported_pcq(axis, block_size):
        """
        When: Create opset10 DequantizeLinear with axis/block_size
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
    @pytest.mark.parametrize("dtype", ["int8", "uint8"])
    def test_QuantizeDequantize_success(dtype: str, zero_point):
        input = np.arange(-128, 256, dtype=np.float32)
        scale = np.ones((), dtype=np.float32)
        zero_point = np.array(zero_point, dtype=np.int64)
        qmin = np.iinfo(dtype).min
        qmax = np.iinfo(dtype).max
        expected_output = (input + zero_point).clip(qmin, qmax) - zero_point

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
            axis=None,
            block_size=None,
        )
        dq_node = DequantizeLinear.make_node(
            name="DequantizeLinear",
            inputs=[input_q_value, scale_value, zero_point_value],
            output=output_value,
            dtype=dtype,
            axis=None,
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

    def test_pack_int8_to_int4x2():
        uint8 = np.arange(16, dtype=np.uint8)
        uint4x2 = pack_int8_to_int4x2(uint8)
        assert uint4x2[0] == 0b00010000
        assert uint4x2[1] == 0b00110010
        assert uint4x2[2] == 0b01010100
        assert uint4x2[3] == 0b01110110
        assert uint4x2[4] == 0b10011000
        assert uint4x2[5] == 0b10111010
        assert uint4x2[6] == 0b11011100
        assert uint4x2[7] == 0b11111110

        uint8_ = unpack_int4x2_to_int8(uint4x2, dtype=np.uint8)
        assert uint8_.dtype == np.uint8
        assert np.all(uint8 == uint8_)

        int8 = np.arange(-8, 8, dtype=np.int8)
        int4x2 = pack_int8_to_int4x2(int8)
        assert int4x2[0] == 0b10011000
        assert int4x2[1] == 0b10111010
        assert int4x2[2] == 0b11011100
        assert int4x2[3] == 0b11111110
        assert int4x2[4] == 0b00010000
        assert int4x2[5] == 0b00110010
        assert int4x2[6] == 0b01010100
        assert int4x2[7] == 0b01110110

        int8_ = unpack_int4x2_to_int8(int4x2, dtype=np.int8)
        assert int8_.dtype == np.int8
        assert np.all(int8 == int8_)
