# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
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
from typing import Optional
import pytest
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    from aimet_common.onnx.opset21 import QuantizeLinear, DequantizeLinear
except ImportError:
    pass
else:
    @pytest.mark.skipif(onnx is None, reason="Requires onnx")
    class TestOpset21:
        @pytest.mark.parametrize("zero_point", [0, 1])
        @pytest.mark.parametrize(
            "axis, block_size", [
            (None, None),
            (0,    None),
            (0,    32),
            (1,    32),
        ])
        @pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16"])
        def test_QuantizeLinear_success(self, dtype: str, axis: Optional[int], block_size: Optional[int], zero_point):
            input = np.arange(-128, 256, dtype=np.float32)
            scale = np.ones((), dtype=np.float32)
            zero_point = np.array(zero_point, dtype=np.int64)
            qmin = np.iinfo(dtype).min
            qmax = np.iinfo(dtype).max
            expected_output = (input + zero_point).clip(qmin, qmax).astype(dtype)

            if axis is not None:
                block_axis = axis if block_size else None
                channel_axis = axis if block_axis is None else 1 - block_axis
                input = input.repeat(3).reshape(*input.shape[:channel_axis],
                                                -1,
                                                *input.shape[channel_axis:])
                scale = scale.repeat(3)
                zero_point = zero_point.repeat(3)
                expected_output = expected_output.repeat(3).reshape(*expected_output.shape[:channel_axis],
                                                                    -1,
                                                                    *expected_output.shape[channel_axis:])

                if block_axis is not None:
                    assert block_size is not None
                    assert input.shape[block_axis] % block_size == 0
                    num_blocks = input.shape[block_axis] // block_size
                    scale_shape = tuple(-1 if axis == channel_axis else 1 for axis in range(input.ndim))

                    scale = scale.reshape(*scale_shape)
                    scale = scale.repeat(num_blocks, axis=block_axis)

                    zero_point = zero_point.reshape(*scale_shape)
                    zero_point = zero_point.repeat(num_blocks, axis=block_axis)

            """
            When: Create onnx model with opset21 QuantizeLinear
            """
            model = onnx.helper.make_model(
                ir_version=10,
                opset_imports=[onnx.OperatorSetIdProto(version=21)],
                graph=onnx.helper.make_graph(
                    name="QuantizeLinear",
                    inputs=[
                        onnx.helper.make_tensor_value_info("input",
                                                           onnx.TensorProto.FLOAT,
                                                           shape=input.shape),
                    ],
                    outputs=[
                        onnx.helper.make_tensor_value_info("output",
                                                           getattr(onnx.TensorProto, dtype.upper()),
                                                           shape=expected_output.shape),
                    ],
                    initializer=[
                        onnx.numpy_helper.from_array(scale, name="scale"),
                        QuantizeLinear.make_zero_point(zero_point,
                                                       dtype=dtype,
                                                       name="zero_point")
                    ],
                    nodes=[
                        QuantizeLinear.make_node(name="QuantizeLinear",
                                                 inputs=["input", "scale", "zero_point"],
                                                 output="output",
                                                 dtype=dtype,
                                                 axis=axis,
                                                 block_size=block_size)
                    ],
                ),
            )

            """
            Then: Model should be runnable on ORT and should produce correct output
            """
            sess = ort.InferenceSession(model.SerializeToString())
            output, = sess.run(None, {"input": input})
            assert np.all(output == expected_output)


        @pytest.mark.parametrize("dtype", ["int32", "uint32"])
        def test_QuantizeLinear_unsupported_dtype(self, dtype):
            """
            When: Create opset21 QuantizeLinear with unsupported dtypes
            Then: Throw runtime error
            """
            with pytest.raises(RuntimeError):
                _ = QuantizeLinear.make_zero_point(np.zeros(()),
                                                   dtype=dtype,
                                                   name="zero_point")

            with pytest.raises(RuntimeError):
                _ = QuantizeLinear.make_node(name="QuantizeLinear",
                                             inputs=["input", "scale", "zero_point"],
                                             output="output",
                                             dtype=dtype,
                                             axis=None,
                                             block_size=None)


        @pytest.mark.parametrize(
            "axis, block_size", [
            (None, 32),
        ])
        def test_QuantizeLinear_invalid_pcq(self, axis, block_size):
            """
            When: Create opset21 QuantizeLinear with invalid axis/block_size
            Then: Throw runtime error
            """
            with pytest.raises(RuntimeError):
                _ = QuantizeLinear.make_node(name="QuantizeLinear",
                                             inputs=["input", "scale", "zero_point"],
                                             output="output",
                                             dtype="int8",
                                             axis=axis,
                                             block_size=block_size)


        @pytest.mark.parametrize("zero_point", [0, 1])
        @pytest.mark.parametrize(
            "axis, block_size", [
            (None, None),
            (0,    None),
            (0,    32),
            (1,    32),
        ])
        @pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32"])
        def test_DequantizeLinear_success(self, dtype: str, axis: Optional[int], block_size: Optional[int], zero_point):
            if dtype == "int32" and zero_point != 0:
                pytest.skip(reason="ORT requires zero_point=0 for int32 DequantizeLinear")

            qmin = np.iinfo(dtype).min
            qmax = np.iinfo(dtype).max
            input = np.arange(-128, 256).clip(qmin, qmax).astype(dtype)

            scale = np.ones((), dtype=np.float32)
            zero_point = np.array(zero_point, dtype=np.int64)
            expected_output = (input.astype(np.int64) - zero_point).astype(np.float32)

            if axis is not None:
                block_axis = axis if block_size else None
                channel_axis = axis if block_axis is None else 1 - block_axis
                input = input.repeat(3).reshape(*input.shape[:channel_axis],
                                                -1,
                                                *input.shape[channel_axis:])
                scale = scale.repeat(3)
                zero_point = zero_point.repeat(3)
                expected_output = expected_output.repeat(3).reshape(*expected_output.shape[:channel_axis],
                                                                    -1,
                                                                    *expected_output.shape[channel_axis:])

                if block_axis is not None:
                    assert block_size is not None
                    assert input.shape[block_axis] % block_size == 0
                    num_blocks = input.shape[block_axis] // block_size
                    scale_shape = tuple(-1 if axis == channel_axis else 1 for axis in range(input.ndim))

                    scale = scale.reshape(*scale_shape)
                    scale = scale.repeat(num_blocks, axis=block_axis)

                    zero_point = zero_point.reshape(*scale_shape)
                    zero_point = zero_point.repeat(num_blocks, axis=block_axis)

            """
            When: Create onnx model with opset21 DequantizeLinear
            """
            model = onnx.helper.make_model(
                ir_version=10,
                opset_imports=[onnx.OperatorSetIdProto(version=21)],
                graph=onnx.helper.make_graph(
                    name="DequantizeLinear",
                    inputs=[
                        onnx.helper.make_tensor_value_info("input",
                                                           getattr(onnx.TensorProto, dtype.upper()),
                                                           shape=input.shape),
                    ],
                    outputs=[
                        onnx.helper.make_tensor_value_info("output",
                                                           onnx.TensorProto.FLOAT,
                                                           shape=expected_output.shape),
                    ],
                    initializer=[
                        onnx.numpy_helper.from_array(scale, name="scale"),
                        DequantizeLinear.make_zero_point(zero_point,
                                                         dtype=dtype,
                                                         name="zero_point")
                    ],
                    nodes=[
                        DequantizeLinear.make_node(name="DequantizeLinear",
                                                   inputs=["input", "scale", "zero_point"],
                                                   output="output",
                                                   dtype=dtype,
                                                   axis=axis,
                                                   block_size=block_size)
                    ],
                ),
            )


            """
            Then: Model should be runnable on ORT and should produce correct output
            """
            sess = ort.InferenceSession(model.SerializeToString())
            output, = sess.run(None, {"input": input})
            assert np.all(output == expected_output)


        @pytest.mark.parametrize(
            "dtype,    zero_point", [
            ("int32",  1), # int32 DQ requires zero_point=0
            ("uint32", 0),
        ])
        def test_DequantizeLinear_unsupported_dtype1(self, dtype, zero_point):
            """
            When: Create opset21 DequantizeLinear.make_zero_point with unsupported dtypes
            Then: Throw runtime error
            """
            with pytest.raises(RuntimeError):
                _ = DequantizeLinear.make_zero_point(np.array(zero_point, dtype=np.int64),
                                                     dtype=dtype,
                                                     name="zero_point")


        @pytest.mark.parametrize("dtype", ["uint32"])
        def test_DequantizeLinear_unsupported_dtype2(self, dtype):
            """
            When: Create opset21 DequantizeLinear.make_node with unsupported dtypes
            Then: Throw runtime error
            """
            with pytest.raises(RuntimeError):
                _ = DequantizeLinear.make_node(name="QuantizeLinear",
                                               inputs=["input", "scale", "zero_point"],
                                               output="output",
                                               dtype=dtype,
                                               axis=None,
                                               block_size=None)


        @pytest.mark.parametrize(
            "axis, block_size", [
            (None, 32),
        ])
        def test_DequantizeLinear_invalid_pcq(self, axis, block_size):
            """
            When: Create opset21 DequantizeLinear with invalid axis/block_size
            Then: Throw runtime error
            """
            with pytest.raises(RuntimeError):
                _ = DequantizeLinear.make_node(name="DequantizeLinear",
                                               inputs=["input", "scale", "zero_point"],
                                               output="output",
                                               dtype="int8",
                                               axis=axis,
                                               block_size=block_size)


        @pytest.mark.parametrize("zero_point", [0, 1])
        @pytest.mark.parametrize(
            "axis, block_size", [
            (None, None),
            (0,    None),
            (0,    32),
            (1,    32),
        ])
        @pytest.mark.parametrize("dtype", ["int4", "uint4", "int8", "uint8", "int16", "uint16"])
        def test_QuantizeDequantize_success(self, dtype: str, axis: Optional[int], block_size: Optional[int], zero_point):
            input = np.arange(-128, 256, dtype=np.float32)
            scale = np.ones((), dtype=np.float32)
            zero_point = np.array(zero_point, dtype=np.int64)
            qmin = -8 if dtype == "int4" else  0 if dtype == "uint4" else np.iinfo(dtype).min
            qmax =  7 if dtype == "int4" else 15 if dtype == "uint4" else np.iinfo(dtype).max
            expected_output = (input + zero_point).clip(qmin, qmax) - zero_point

            if axis is not None:
                block_axis = axis if block_size else None
                channel_axis = axis if block_axis is None else 1 - block_axis
                input = input.repeat(3).reshape(*input.shape[:channel_axis],
                                                -1,
                                                *input.shape[channel_axis:])
                scale = scale.repeat(3)
                zero_point = zero_point.repeat(3)
                expected_output = expected_output.repeat(3).reshape(*expected_output.shape[:channel_axis],
                                                                    -1,
                                                                    *expected_output.shape[channel_axis:])

                if block_axis is not None:
                    assert block_size is not None
                    assert input.shape[block_axis] % block_size == 0
                    num_blocks = input.shape[block_axis] // block_size
                    scale_shape = tuple(-1 if axis == channel_axis else 1 for axis in range(input.ndim))

                    scale = scale.reshape(*scale_shape)
                    scale = scale.repeat(num_blocks, axis=block_axis)

                    zero_point = zero_point.reshape(*scale_shape)
                    zero_point = zero_point.repeat(num_blocks, axis=block_axis)

            """
            When: Create onnx model with opset21 QuantizeLinear - DequantizeLinear
            """
            model = onnx.helper.make_model(
                ir_version=10,
                opset_imports=[onnx.OperatorSetIdProto(version=21)],
                graph=onnx.helper.make_graph(
                    name="QuantizeDequantize",
                    inputs=[
                        onnx.helper.make_tensor_value_info("input",
                                                           onnx.TensorProto.FLOAT,
                                                           shape=input.shape),
                    ],
                    outputs=[
                        onnx.helper.make_tensor_value_info("output",
                                                           onnx.TensorProto.FLOAT,
                                                           shape=expected_output.shape),
                    ],
                    initializer=[
                        onnx.numpy_helper.from_array(scale, name="scale"),
                        QuantizeLinear.make_zero_point(zero_point,
                                                       dtype=dtype,
                                                       name="zero_point")
                    ],
                    nodes=[
                        QuantizeLinear.make_node(name="QuantizeLinear",
                                                 inputs=["input", "scale", "zero_point"],
                                                 output="input_q",
                                                 dtype=dtype,
                                                 axis=axis,
                                                 block_size=block_size),
                        DequantizeLinear.make_node(name="DequantizeLinear",
                                                   inputs=["input_q", "scale", "zero_point"],
                                                   output="output",
                                                   dtype=dtype,
                                                   axis=axis,
                                                   block_size=block_size)
                    ],
                ),
            )

            """
            Then: Model should be runnable on ORT and should produce correct output
            """
            sess = ort.InferenceSession(model.SerializeToString())
            output, = sess.run(None, {"input": input})
            assert np.all(output == expected_output)
