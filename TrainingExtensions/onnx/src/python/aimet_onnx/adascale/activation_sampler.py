# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Sample output from original module for Adascale feature"""

from typing import List, Dict, Union, Optional, Sequence, Tuple, Any

import numpy as np
import onnx_ir

from aimet_onnx.utils import (
    OrtInferenceSession,
)


class ActivationSampler:
    """
    For a module in the model, collect the module's FP output and Quantized input activation data
    """

    def __init__(
        self,
        activation_name: str,
        model_path: str,
        providers: Optional[Sequence[str | Tuple[str, Dict[Any, Any]]]] = None,
    ):
        """
        :param activation_name: tensor name of the module whose output we want to retrieve
        :param model_path: Path to an ONNX model file
        :param providers: List of providers to use
        :return: Input data to quant op, Output data from original op
        """
        self._activation_name = activation_name
        self._sess, self._model = self.create_session(
            model_path, activation_name, providers
        )

    @staticmethod
    def create_session(
        model_path: str,
        activation: Union[str, List[str]],
        providers,
    ):
        """
        Helper to create a session using both module's input and output tensor names

        :param model_path: Path to an ONNX model file
        :param activation: activation to add a hook to
        :param providers: List of providers to use
        """
        ir_model = onnx_ir.load(model_path)
        if isinstance(activation, str):
            activation = [activation]
        ir_model.graph = onnx_ir.convenience.extract(
            ir_model.graph, inputs=ir_model.graph.inputs, outputs=activation
        )
        subgraph_path = model_path.replace(".onnx", "_subgraph.onnx")
        onnx_ir.save(ir_model, subgraph_path)
        sess = OrtInferenceSession(subgraph_path, providers)
        return sess, ir_model

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

    def sample_and_place_all_acts_on_cpu(self, dataset) -> List:
        """
        Given the dataset, compute the activation tensors corresponding to activation_name
        :param dataset: input dataset
        :return: outputs corresponding to the activation tensors registered
        """
        all_data = []

        iterator = iter(dataset)
        for _ in range(len(dataset)):
            model_inputs = next(iterator)
            data = self.sample_acts(model_inputs)

            all_data.append(data)

        return all_data

    def sample_acts(self, model_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Given the model_inputs retrieve the activation tensors corresponding to activation_name
        :param model_inputs: inputs to the model
        :return: Activation sample for the given input
        """
        module_input_act = self.run_session(
            self._sess, model_inputs, self._activation_name
        )

        return module_input_act
