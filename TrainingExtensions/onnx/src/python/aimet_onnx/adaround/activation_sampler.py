# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Sample output from original module for Adaround feature"""

from typing import Tuple, List, Dict

import numpy as np
import onnxruntime as ort
import onnx
from packaging import version

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import (
    add_hook_to_get_activation,
    remove_activation_hooks,
    create_input_dict,
    OrtInferenceSession,
    _run_session_to_numpy,
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
    ):
        """
        :param fp_act_name: FP output tensor name of the module to retrieve
        :param quant_act_name: Quant input tensor name of the module to retrieve
        :param quant_sim: QuantizationSimModel object
        :param fp32_model: Unquantized FP32 model
        :return: Input data to quant op, Output data from original op
        """
        self._quant_sim = quant_sim
        self._fp32_model = fp32_model
        self._fp_act_name = fp_act_name
        self._quant_act_name = quant_act_name
        self.providers = []

        for provider in quant_sim.session.get_providers():
            if (
                provider == "CUDAExecutionProvider"
                or "CUDAExecutionProvider" in provider
            ):
                device_id = (
                    quant_sim.session.get_provider_options()
                    .get("CUDAExecutionProvider", {})
                    .get("device_id", "0")
                )
                self.providers.append(
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": device_id,
                            "cudnn_conv_algo_search": _cudnn_conv_algo_search,
                        },
                    )
                )
            else:
                self.providers.append("CPUExecutionProvider")

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
        # TODO: optional: Clean all the outputs except the one we want?
        handle = add_hook_to_get_activation(model, activation)
        sess = OrtInferenceSession(
            model,
            self.providers,
            self._quant_sim._ort_session_options,
            save_as_external_data=self._quant_sim._use_external_data,
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
            act_output = _run_session_to_numpy(
                session, model_inputs, [activation_name]
            )[0]
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
