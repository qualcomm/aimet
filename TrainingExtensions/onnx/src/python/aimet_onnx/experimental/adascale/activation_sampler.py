# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Sample output from original module for Adascale feature"""

from typing import List, Dict, Union, Optional, Sequence, Tuple, Any

import numpy as np
import onnx
from packaging import version
import os

from aimet_onnx.utils import (
    add_hook_to_get_activation,
    remove_activation_hooks,
    create_input_dict,
    OrtInferenceSession,
)

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import ModelProto
else:
    from onnx.onnx_pb import ModelProto


class ActivationSampler:
    """
    For a module in the model, collect the module's FP output and Quantized input activation data
    """

    def __init__(
        self,
        activation_name: str,
        model_or_path: Union[ModelProto, str],
        providers: Optional[Sequence[str | Tuple[str, Dict[Any, Any]]]] = None,
        path: str = None,
    ):
        """
        :param activation_name: tensor name of the module whose output we want to retrieve
        :param model_or_path: ONNX ModelProto or path to an ONNX model file
        :param providers: List of providers to use
        :path: path of stored fp model
        :return: Input data to quant op, Output data from original op
        """
        self._activation_name = activation_name
        (
            self._sess,
            self._handle,
            self._model,
        ) = self.create_session(model_or_path, activation_name, providers, path)

    @staticmethod
    def create_session(
        model_or_path: Union[ModelProto, str],
        activation: Union[str, List[str]],
        providers,
        path: str,
    ):
        """
        Helper to create a session using both module's input and output tensor names

        :param model_or_path: ONNX ModelProto or path to an ONNX model file
        :param activation: activation to add a hook to
        :param providers: List of providers to use
        :path: path to store the onnx model

        Note:
        Aimet wrapper for OrtInferenceSession is saving and loading the onnx-model before creating a session.
        Each time load_external_data_for_model is called, it is allocating new memory for the weights and not reusing the existing memory
        Workaround:
        1. QuantSim: Pass a copy of the model when creating ActivationSampler
        2. FP : Save the fp model to disk with save_as_external_data = True. Load the model , add hooks and create session with model path,
            (will load the saved weights from disk)
        """
        if isinstance(model_or_path, str):
            model = onnx.load(model_or_path, load_external_data=False)
            handle = add_hook_to_get_activation(model, activation)
            model_path = os.path.join(path, "fp32_model.onnx")
            onnx.save(model, model_path)
            sess = OrtInferenceSession(model_path, providers)
        else:
            model = model_or_path
            handle = add_hook_to_get_activation(model, activation)
            sess = OrtInferenceSession(model, providers)

        return sess, handle, model

    def restore_graph(self):
        """
        Remove all the additional model outputs added to the graph and restore its original state
        """
        remove_activation_hooks(self._model, self._handle)

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

    def sample_acts(self, model_inputs: Dict[str, List[np.ndarray]]) -> List:
        """
        Given the model_inputs retrieve the activation tensors corresponding to activation_name
        :param model_inputs: inputs to the model
        :return: Tuple of module's quantized input activation and its fp activation output
        """
        model_inputs = create_input_dict(self._model, model_inputs)
        module_input_act = self.run_session(
            self._sess, model_inputs, self._activation_name
        )

        return module_input_act
