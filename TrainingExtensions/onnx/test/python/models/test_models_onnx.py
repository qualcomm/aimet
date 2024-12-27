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
"""Dummy onnx models for testing"""


import onnx
import numpy as np


def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor

class ModelWithMultipleInputs:

    @staticmethod
    def get_model():
        model_input_name = "X1"
        X1 = onnx.helper.make_tensor_value_info(model_input_name,
                                                onnx.TensorProto.FLOAT,
                                                [1, 3, 32, 32])

        model_input_name = "X2"
        X2 = onnx.helper.make_tensor_value_info(model_input_name,
                                                onnx.TensorProto.FLOAT,
                                                [1, 3, 32, 32])

        conv1_output_node_name = "Conv1_Y"

        # Dummy weights for conv.
        conv1_in_channels = 3
        conv1_out_channels = 3
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.ones(shape=(conv1_out_channels, conv1_in_channels,
                                 *conv1_kernel_shape)).astype(np.float32)
        conv1_B = np.ones(shape=(conv1_out_channels)).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT)
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT)

        conv1_node = onnx.helper.make_node(
            name="Conv1",
            op_type="Conv",
            inputs=[
                "X1", conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        add_0_node_name = "ADD_0"

        add_0_node = onnx.helper.make_node(
            name=add_0_node_name,  # Name is optional.
            op_type="Add",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[
                conv1_output_node_name, "X2"
            ],
            outputs=[add_0_node_name]
        )

        conv2_output_node_name = "Conv2_Y"

        # Dummy weights for conv.
        conv2_in_channels = 3
        conv2_out_channels = 3
        conv2_kernel_shape = (3, 3)
        conv2_pads = (1, 1, 1, 1)
        conv2_W = np.ones(shape=(conv2_out_channels, conv2_in_channels,
                                 *conv2_kernel_shape)).astype(np.float32)
        conv2_B = np.ones(shape=(conv2_out_channels)).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT)
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT)

        conv2_node = onnx.helper.make_node(
            name="Conv2",  # Name is optional.
            op_type="Conv",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[
                add_0_node_name, conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name
            ],
            outputs=[conv2_output_node_name],
            # The following arguments are attributes.
            kernel_shape=conv2_kernel_shape,
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=conv2_pads,
        )

        add_1_node_name = "ADD_1"

        add_1_node = onnx.helper.make_node(
            name=add_1_node_name,  # Name is optional.
            op_type="Add",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[
                conv2_output_node_name, "X1"
            ],
            outputs=["Y"]
        )


        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [1, 3, 32, 32])


        graph_def = onnx.helper.make_graph(
            nodes=[conv1_node, add_0_node, conv2_node, add_1_node],
            name="ConvReluNet",
            inputs=[X1, X2],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor, conv2_W_initializer_tensor, conv2_B_initializer_tensor],
        )
        # Create the model (ModelProto)
        return onnx.helper.make_model(graph_def, producer_name="onnx-example")


def model_with_multiple_inputs():
    return ModelWithMultipleInputs.get_model()


class ModelWithMultipleOutputs:

    @staticmethod
    def get_model():
        model_input_name = "X1"
        X1 = onnx.helper.make_tensor_value_info(model_input_name,
                                                onnx.TensorProto.FLOAT,
                                                [1, 3, 32, 32])

        model_input_name = "X2"
        X2 = onnx.helper.make_tensor_value_info(model_input_name,
                                                onnx.TensorProto.FLOAT,
                                                [1, 3, 32, 32])

        conv1_output_node_name = "Conv1_Y"

        # Dummy weights for conv.
        conv1_in_channels = 3
        conv1_out_channels = 3
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.ones(shape=(conv1_out_channels, conv1_in_channels,
                                 *conv1_kernel_shape)).astype(np.float32)
        conv1_B = np.ones(shape=(conv1_out_channels)).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT)
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT)

        conv1_node = onnx.helper.make_node(
            name="Conv1",
            op_type="Conv",
            inputs=[
                "X1", conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        add_0_node_name = "ADD_0"

        add_0_node = onnx.helper.make_node(
            name=add_0_node_name,  # Name is optional.
            op_type="Add",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[
                conv1_output_node_name, "X2"
            ],
            outputs=[add_0_node_name]
        )

        conv2_output_node_name = "Conv2_Y"

        # Dummy weights for conv.
        conv2_in_channels = 3
        conv2_out_channels = 3
        conv2_kernel_shape = (3, 3)
        conv2_pads = (1, 1, 1, 1)
        conv2_W = np.ones(shape=(conv2_out_channels, conv2_in_channels,
                                 *conv2_kernel_shape)).astype(np.float32)
        conv2_B = np.ones(shape=(conv2_out_channels)).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT)
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT)

        conv2_node = onnx.helper.make_node(
            name="Conv2",  # Name is optional.
            op_type="Conv",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[
                add_0_node_name, conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name
            ],
            outputs=[conv2_output_node_name],
            # The following arguments are attributes.
            kernel_shape=conv2_kernel_shape,
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=conv2_pads,
        )

        add_1_node_name = "ADD_1"

        add_1_node = onnx.helper.make_node(
            name=add_1_node_name,  # Name is optional.
            op_type="Add",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[
                conv2_output_node_name, "X1"
            ],
            outputs=["Y"]
        )


        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [1, 3, 32, 32])

        conv1_output = onnx.helper.make_tensor_value_info(conv1_output_node_name,
                                                      onnx.TensorProto.FLOAT,
                                                      [1, 3, 32, 32])


        graph_def = onnx.helper.make_graph(
            nodes=[conv1_node, add_0_node, conv2_node, add_1_node],
            name="ConvReluNet",
            inputs=[X1, X2],  # Graph input
            outputs=[Y, conv1_output],  # Graph output
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor, conv2_W_initializer_tensor, conv2_B_initializer_tensor],
        )
        # Create the model (ModelProto)
        return onnx.helper.make_model(graph_def, producer_name="onnx-example")


def model_with_multiple_outputs():
    return ModelWithMultipleOutputs.get_model()
