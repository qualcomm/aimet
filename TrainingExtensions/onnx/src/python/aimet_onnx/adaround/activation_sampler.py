# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""Sample output from original module for Adaround feature"""

import copy
from typing import Tuple, List, Dict, Union

import numpy as np
import onnxruntime as ort
import onnx
from packaging import version

from aimet_common.utils import AimetLogger
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import (
    add_hook_to_get_activation,
    remove_activation_hooks,
    create_input_dict,
    disable_quantizers,
    build_session,
)

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import ModelProto
else:
    from onnx.onnx_pb import ModelProto

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


# TODO: Remove redundant session build/teardown
#       and and always use "EXHUASTIVE" (ORT default policy)
if ort.__version__ < "1.20":
    _cudnn_conv_algo_search = "DEFAULT"
else:
    _cudnn_conv_algo_search = "HEURISTIC"


class ActivationSampler:
    """
    For a module in the model, collect the module's FP output and Quantized input activation data
    """

    def __init__(
        self,
        fp_act_name: str,
        quant_act_name: str,
        quant_sim: QuantizationSimModel,
        fp32_model: ModelProto,
        use_cuda: bool,
        device: int = 0,
    ):
        """
        :param fp_act_name: FP output tensor name of the module to retrieve
        :param quant_act_name: Quant input tensor name of the module to retrieve
        :param quant_sim: QuantizationSimModel object
        :param fp32_model: Unquantized FP32 model
        :param use_cuda: If we should use cuda
        :param device: CUDA device ID
        :return: Input data to quant op, Output data from original op
        """
        self._quant_sim = quant_sim
        self._fp32_model = fp32_model
        self._fp_act_name = fp_act_name
        self._quant_act_name = quant_act_name

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            logger.warning(
                "CUDAExecutionProvider not in ort available providers. use_cuda is set to False"
            )
            use_cuda = False
        if use_cuda:
            self.providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device,
                        "cudnn_conv_algo_search": _cudnn_conv_algo_search,
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            self.providers = ["CPUExecutionProvider"]

        self.fp32_sess, self.fp32_handle = self.create_session(
            self._fp32_model, self._fp_act_name
        )
        self.qsim_sess, self.qsim_handle = self.create_session(
            self._quant_sim.model.model, self._quant_act_name
        )

    def create_session(self, model: onnx.ModelProto, activation: str):
        """
        Helper to create a session using both module's input and output tensor names

        :param model: ONNX ModelProto to create a session
        :param activation: activation to add a hook to
        """
        handle = add_hook_to_get_activation(model, activation)
        sess = build_session(
            model,
            self.providers,
            self._quant_sim._user_onnx_libs,
        )
        return sess, handle

    def restore_graph(self):
        """
        Remove all the additional model outputs added to the graph and restore its original state
        """
        remove_activation_hooks(self._fp32_model, self.fp32_handle)
        remove_activation_hooks(self._quant_sim.model.model, self.qsim_handle)

    @staticmethod
    def run_session(
        session, model_inputs: Dict[str, List[np.ndarray]], activation_name: str
    ) -> np.ndarray:
        """
        Return quantized module input and fp module outputs using the given model_inputs
        :param model_inputs: inputs to the model
        :param activation_name: list of activation names to retrieve the output
        :param session: session to run
        :return: outputs corresponding to the activation_names of the session given model inputs
        """

        if activation_name in model_inputs:
            # Workaround memory corruption bug in onnxruntime >= 1.19 when a graph output is also a graph input
            # https://github.com/microsoft/onnxruntime/issues/21922
            act_output = model_inputs[activation_name]
        else:
            act_output = session.run([activation_name], model_inputs)[0]
        return act_output

    def sample_and_place_all_acts_on_cpu(self, dataset) -> Tuple:
        """
        Given the dataset, compute the activation tensors corresponding to the tensors: fp_act_name, quant_act_name
        :param dataset: input dataset
        :return: outputs corresponding to the activation tensors registered
        """
        all_inp_data = []
        all_out_data = []

        iterator = iter(dataset)
        for batch_index in range(len(dataset)):
            model_inputs = next(iterator)
            inp_data, out_data = self.sample_acts(
                create_input_dict(self._quant_sim.model.model, model_inputs)
            )

            all_inp_data.append(inp_data)
            all_out_data.append(out_data)

        return all_inp_data, all_out_data

    def sample_acts(self, model_inputs: Dict[str, List[np.ndarray]]):
        """
        Given the model_inputs retrieve the activation tensors corresponding to the tensors: fp_act_name, quant_act_name
        :param model_inputs: inputs to the model
        :return: Tuple of module's quantized input activation and its fp activation output
        """

        module_input_act = self.run_session(
            self.qsim_sess, model_inputs, self._quant_act_name
        )
        module_output_act = self.run_session(
            self.fp32_sess, model_inputs, self._fp_act_name
        )

        return module_input_act, module_output_act
