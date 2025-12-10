# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""Utility functions for ONNX"""

import copy
import itertools
import math
import platform
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Union, Tuple, Callable, Optional
from contextlib import contextmanager
import os
import pickle
import numpy as np
import torch
import onnx
from onnx import helper, numpy_helper
from onnx.utils import Extractor
from onnx.external_data_helper import (
    load_external_data_for_model,
)
from onnxruntime import SessionOptions, InferenceSession
import shutil
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from aimet_onnx.common import libquant_info
from aimet_onnx.common.utils import AimetLogger, compute_psnr, deprecated
from aimet_onnx.common.onnx._utils import (  # pylint: disable=unused-import
    _ParamUtils,
    _get_node_attribute as get_node_attribute,
    contains_tensor_type,
)
from packaging import version

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import NodeProto, TensorProto, ModelProto, GraphProto, ValueInfoProto
else:
    from onnx.onnx_pb import (
        NodeProto,
        TensorProto,
        ModelProto,
        GraphProto,
        ValueInfoProto,
    )

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

ONE_B_PARAMS = 2**28

OP_TYPES_WITH_PARAMS = [
    "Conv",
    "Gemm",
    "ConvTranspose",
    "BatchNormalization",
    "MatMul",
    "Transpose",
    "InstanceNormalization",
    "LayerNormalization",
    "GroupNormalization",
    "RNN",
    "LSTM",
    "GRU",
]


def remove_nodes_with_type(node_type: str, onnx_graph: onnx.GraphProto):
    """
    Remove specific type of nodes from graph

    :param node_type: string, type of node to be removed
    :param onnx_graph: onnx graph to modify

    """
    input_output_pairs = {}

    onnx_graph_node_list = copy.deepcopy(onnx_graph.node)

    for node in onnx_graph_node_list:
        if node.op_type == node_type:
            input_output_pairs[node.output[0]] = node.input[0]
            onnx_graph.node.remove(node)
    for node in onnx_graph.node:
        for i, _input in enumerate(node.input):
            output = input_output_pairs.get(_input, None)
            if output:
                node.input[i] = input_output_pairs[_input]
        for outputs in onnx_graph.output:
            if (
                outputs.name in input_output_pairs
                and node.output[0] == input_output_pairs[outputs.name]
            ):
                node.output[0] = outputs.name


def _prune_unused_initializer(graph: onnx.GraphProto, init_name: str):
    """
    Remove initializer from graph if it is unused
    """
    return _prune_unused_initializers(graph, [init_name])


def _prune_unused_initializers(graph: onnx.GraphProto, init_names: Iterable[str]):
    """
    Remove initializer from graph if it is unused
    """
    to_be_pruned = set(init_names)

    # Don't prune if initializer is still used
    to_be_pruned -= set(inp for node in graph.node for inp in node.input)

    initializers = [init for init in graph.initializer if init.name not in to_be_pruned]

    graph.ClearField("initializer")
    graph.initializer.extend(initializers)

    inputs = [inp for inp in graph.input if inp.name not in to_be_pruned]
    graph.ClearField("input")
    graph.input.extend(inputs)


def remove_node(node: NodeProto, onnx_graph: onnx.GraphProto):
    """
    Remove a specific node from graph along with associated initializers

    :param node: the node to be removed
    :param onnx_graph: onnx graph to modify

    """
    remove_nodes([node], onnx_graph)


def remove_nodes(nodes: Iterable[NodeProto], onnx_graph: onnx.GraphProto):
    """
    Remove a specific node from graph along with associated initializers

    :param node: the node to be removed
    :param onnx_graph: onnx graph to modify

    """

    removed_nodes = {node.name: node for node in nodes}
    remaining_nodes = [
        node for node in onnx_graph.node if node.name not in removed_nodes
    ]
    output_to_node = {node.output[0]: node for node in onnx_graph.node}

    for other_node in remaining_nodes:
        if other_node.input and other_node.output:
            # Check if other node takes input from removed node
            for idx, _input in enumerate(other_node.input):
                prev_node = output_to_node.get(_input)
                if prev_node and prev_node.name in removed_nodes:
                    other_node.input[idx] = prev_node.input[0]

    for output in onnx_graph.output:
        prev_node = output_to_node.get(output.name)
        if prev_node and prev_node.name in removed_nodes:
            prev_prev_node = output_to_node.get(prev_node.input[0])

            if not prev_prev_node:
                raise RuntimeError(
                    "Dangling output: "
                    f"Leaf output '{output.name}' is not an output of any node"
                )

            prev_prev_node.output[0] = output.name

    for node in removed_nodes.values():
        onnx_graph.node.remove(node)

    removed_node_inputs = set(
        inp for node in removed_nodes.values() for inp in node.input
    )
    _prune_unused_initializers(onnx_graph, removed_node_inputs)


def transpose_tensor(t: TensorProto, axes: Union[List, Tuple]) -> TensorProto:
    """
    Permutes the axes of a given array using numpy.transpose

    :param t: tensor to transpose
    :param axes: tuple or list containing the permuted axis ordering
    :return: t permuted according to the axis ordering in axes
    """
    t_np = numpy_helper.to_array(t)
    # Expand tensor with singleton dimensions to match axes
    while len(t_np.shape) < len(axes):
        t_np = np.expand_dims(t_np, len(t_np.shape))
    return numpy_helper.from_array(np.transpose(t_np, axes), name=t.name)


def replace_node_with_op(node_type: str, new_type: str, onnx_graph: onnx.GraphProto):
    """
    Replace the given op type of nodes to new op type

    :param node_type: string, type of node to be replaced
    :param new_type: string, type of node to substitute for
    :param onnx_graph: onnx graph to modify

    """
    for node in onnx_graph.node:
        if node.op_type == node_type:
            node.op_type = new_type


def get_weights(name: str, onnx_graph: onnx.GraphProto) -> bytes:
    """
    Return the weights by given name
    :param name, name of the weights to find
    :param onnx_graph, onnx graph to find the corresponding weight data
    :return onnx tensor
    """
    for param in onnx_graph.initializer:
        if param.name == name:
            return param.raw_data
    assert Exception("Couldn't find weights by the given name")
    return None


def get_ordered_dict_of_nodes(onnx_graph: onnx.GraphProto) -> Dict:
    """
    Return the ordered list of nodes

    :param onnx_graph: onnx graph to provide node info
    :return dict of ordered nodes with name as key

    """
    ordered_dict = {}
    for node in onnx_graph.node:
        ordered_dict[node.name] = node
    return ordered_dict


def make_dummy_input(model: ModelProto, dynamic_size: int = 1) -> Dict[str, np.ndarray]:
    """
    Create a dummy input based on the model input types and shapes

    :param model: Model to create an input for
    :param dynamic_size: Dimension size to use for dynamic axes
    :return: Dictionary of input_name : input array
    """
    input_dict = {}
    for item in model.graph.input:
        name = item.name
        dtype = item.type.tensor_type.elem_type
        shape = []
        for dim in item.type.tensor_type.shape.dim:
            if dim.dim_param:
                # Evaluates true if axis is dynamic. We set the size of dynamic axes to dynamic_size
                shape.append(dynamic_size)
            else:
                # Else, axis has a fixed dimension size stored in dim.dim_value
                shape.append(dim.dim_value)
        if shape:
            input_dict[name] = np.random.randn(*shape).astype(
                helper.tensor_dtype_to_np_dtype(dtype)
            )
        else:
            input_dict[name] = np.array(np.random.randn(*shape)).astype(
                helper.tensor_dtype_to_np_dtype(dtype)
            )
    return input_dict


def replace_relu6_with_relu(model: ModelProto):
    """
    Replace relu6 op with relu op

    :param model: ONNX model
    """
    for node in model.model.graph.node:
        if node.op_type == "Clip" and check_if_clip_node_minimum_is_zero(node, model):
            parent_node = None
            child_node = None
            for temp_node in model.model.graph.node:
                if node.input[0] in temp_node.output:
                    parent_node = temp_node
                if node.output[0] in temp_node.input:
                    child_node = temp_node
            assert parent_node, "Parent Node for Clip operation does not exist"
            if (
                parent_node.op_type in ["Conv", "ConvTranspose"]
                and child_node
                and child_node.op_type in ["Conv", "ConvTranspose"]
            ):
                name = node.name
                remove_node(node, model.model.graph)
                inputs = [parent_node.output[0]]
                model.replace_input_of_all_nodes(
                    parent_node.output[0], parent_node.output[0] + "_replaced"
                )
                relu_node = onnx.helper.make_node(
                    op_type="Relu",
                    inputs=inputs,
                    outputs=[parent_node.output[0] + "_replaced"],
                    name="Relu_" + name,
                )

                model.add_node(relu_node)
    model.topological_sort()


def check_if_clip_node_minimum_is_zero(node: NodeProto, model: ModelProto):
    """
    Check if the clip node's minimum is 0

    :param node: ONNX node
    :param model: ONNX model
    """
    if len(node.input) == 3:
        input_node = node.input[1]
        for node_graph in model.model.graph.node:
            if node_graph.output[0] == input_node:
                if (
                    hasattr(node_graph, "attribute")
                    and hasattr(node_graph.attribute[0], "t")
                    and numpy_helper.to_array(node_graph.attribute[0].t) == 0
                ):
                    return True
    elif (
        hasattr(node, "attribute")
        and node.attribute[1].name == "min"
        and node.attribute[1].f == 0.0
    ):
        return True
    return False


def add_hook_to_get_activation(model: ModelProto, name: str) -> ValueInfoProto:
    """
    Adds a given activation to the model output
    :param model: The model to add the hook to
    :param name: The name of the activation
    :return: ValueInfoProto for the given activation that has been appended to model.graph.output
    """
    val_info = onnx.helper.ValueInfoProto()
    val_info.name = name
    model.graph.output.append(val_info)
    return val_info


def remove_activation_hooks(
    model: ModelProto, hooks: Union[List[ValueInfoProto], ValueInfoProto]
):
    """
    Removes activation hooks from the model output
    :param model: The model from which to remove the hooks
    :param hooks: Value info or list of value infos to remove from the model output
    """
    if not isinstance(hooks, List):
        hooks = [hooks]
    for hook in hooks:
        model.graph.output.remove(hook)


def get_graph_intermediate_activations(graph: GraphProto) -> List[str]:
    """
    Returns the names of all activations within a graph that are used as the input to another node
    :param graph: The graph for which to retrieve the activations
    :return: A list containing the names of all found activations
    """
    param_names = []
    for param in graph.initializer:
        if param.name not in param_names and param.name:
            param_names.append(param.name)
    activation_names = []
    for node in graph.node:
        for name in node.input:
            if name not in activation_names and name not in param_names and name:
                activation_names.append(name)
    return activation_names


ParamUtils = _ParamUtils


def get_product_name_from_quantized_name(quantized_name: str):
    """
    Gets product's name from quantized name
    :param quantized_name: Quantized name
    """
    if "_updated" in quantized_name:
        return quantized_name[: quantized_name.index("_updated")]
    if "_qdq" in quantized_name:
        return quantized_name[: quantized_name.index("_qdq")]
    # If there is no quantizer added then return None
    return None


def retrieve_constant_input(
    node: NodeProto, model: ModelProto, index: int
) -> Tuple[TensorProto, bool]:
    """
    Retrieves node input at the specified index if the input has a corresponding initializer in model.graph.initializer
    and is separated from node by no more than one Transpose operation.
    :param node: The node to find the input for
    :param model: The model to which the node belongs
    :param index: The index of the desired input within node.input
    :return: Tuple containing the input parameter and a bool specifying whether the param is transposed before entering
             the node
    """
    weight_input = node.input[index]
    transposed = False
    weight = ParamUtils.get_param(model, node, index)
    if not weight:
        # Check if the weight is transposed before entering the node
        for other_node in model.graph.node:
            if weight_input in other_node.output and other_node.op_type == "Transpose":
                weight = ParamUtils.get_param(model, other_node, 0)
                transposed = True
    return weight, transposed


def save_model_with_external_weights(model: onnx.ModelProto, f: str, **kwargs):
    """
    Saves an onnx model with external weights without mutating the original model

    :param model: ONNX ModelProto object to save
    :param f: filename to save the model to
    :param kwargs: Additional keyword arguments to pass to :func:`onnx.save_model`
    """
    onnx.save_model(model, f, save_as_external_data=True, **kwargs)
    # Load back weights which are removed when saving as external data
    onnx.load_external_data_for_model(model, os.path.dirname(f))


@contextmanager
def disable_quantizers(sim, quantizer_names: set[str]):
    """Disables all quantizers in quantizer_names inside the context"""
    if not isinstance(quantizer_names, set):
        quantizer_names = set(quantizer_names)

    if not quantizer_names.issubset(sim.qc_quantize_op_dict.keys()):
        raise RuntimeError(
            f"quantizer_names contains non-existent quantizers: {quantizer_names - sim.qc_quantize_op_dict.keys()}"
        )

    is_enabled = {
        name: sim.qc_quantize_op_dict[name].enabled for name in quantizer_names
    }

    try:
        for name in quantizer_names:
            sim.qc_quantize_op_dict[name].enabled = False

        yield

    finally:
        for name in quantizer_names:
            sim.qc_quantize_op_dict[name].enabled = is_enabled[name]


class CachedDataset:
    """
    Cache number of batches from the data loader at given path location and
    provide interface to fetch single batch of model inputs.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, data_loader, num_batches: int, path: str):
        """
        :param data_loader: Data loader
        :param num_batches: Number of batches to fetch from data loader
        :param path: Path to save model inputs
        """
        if len(data_loader) < num_batches:
            raise ValueError(
                f"Can not fetch {num_batches} batches from "
                f"a data loader of length {len(data_loader)}."
            )

        self._num_batches = num_batches
        self._path = path

        self._cache_model_inputs(itertools.islice(data_loader, num_batches))

    def __len__(self):
        return self._num_batches

    def __getitem__(self, index: int):
        path = os.path.join(self._path, "model_inputs_" + str(index))

        with open(path, "rb") as file:
            batch = pickle.load(file)

        return batch

    def _cache_model_inputs(self, data_loader):
        """
        Function to cache number of batches individually in separate file at provided path location
        """
        if not os.path.exists(self._path):
            os.makedirs(self._path)

        for i, batch in enumerate(data_loader):
            path = os.path.join(self._path, f"model_inputs_{i}")
            with open(path, "wb") as file:
                pickle.dump(batch, file)

        logger.info(
            "Caching %d batches from data loader at path location: %s",
            self._num_batches,
            self._path,
        )


def create_input_dict(
    model: ModelProto,
    input_batch: Union[Dict, np.ndarray, List[np.ndarray], Tuple[np.ndarray]],
) -> Dict:
    """
    Creates input dictionary (input name to input value map) for session.run

    :param model: ONNX model
    :param input_batch: either a dict, single numpy array, list or tuple of numpy array
    :return: input dictionary
    """
    if isinstance(input_batch, dict):
        return input_batch

    input_names = [input.name for input in model.graph.input]

    # single input
    if isinstance(input_batch, (np.ndarray, torch.Tensor)):
        input_batch_list = [input_batch]

    # list of multiple inputs
    elif isinstance(input_batch, list):
        input_batch_list = input_batch

    # tuple of multiple inputs
    elif isinstance(input_batch, tuple):
        input_batch_list = list(input_batch)

    else:
        raise ValueError(
            "Input batch should be either dict, numpy array, list or tuple"
        )

    if not len(input_names) == len(input_batch_list):
        raise ValueError(
            "There is mismatch between number of input names and input tensors"
        )

    return dict(zip(input_names, input_batch_list))


def create_ort_session_options_with_aimet_custom_ops() -> SessionOptions:
    """
    Returns onnxruntime session options with aimet custom ops registered
    :return: onnxruntime session options
    """

    session_options = SessionOptions()

    shared_library = os.path.join(
        os.path.dirname(libquant_info.__file__),
        "libaimet_onnxrt_ops.dll"
        if platform.system() == "Windows"
        else "libaimet_onnxrt_ops.so",
    )

    session_options.register_custom_ops_library(shared_library)
    return session_options


class OrtInferenceSession(InferenceSession):
    """
    onnxruntime inference session with automatic handling of model cleanup.

    If session_options are not provided,
        creates session options with aimet custom ops registered.
    If path is not provided,
        creates a temp directory to store model external data and cleans it up on object deletion.
    """

    def __init__(
        self,
        model: onnx.ModelProto,
        providers: List,
        session_options: SessionOptions = None,
        path: str = None,
    ):
        """
        Build and return onnxruntime inference session

        :param model: onnx model
        :param providers: providers to execute onnxruntime
        :param session_options: onnxruntime session options
        :param path: path where to store model external data
        """

        self.model_dir: Optional[str] = None

        if path is None:
            self.model_dir = tempfile.mkdtemp()
            path = self.model_dir

        model_path = os.path.join(path, "model.onnx")
        save_model_with_external_weights(
            model, model_path, location=Path(model_path).name + ".data"
        )

        if session_options is None:
            session_options = create_ort_session_options_with_aimet_custom_ops()

        super().__init__(
            path_or_bytes=model_path,
            sess_options=session_options,
            providers=providers,
        )

    def __del__(self):
        if self.model_dir is None:
            return

        try:
            shutil.rmtree(self.model_dir)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Error while cleaning up temp dir at %s: %s", self.model_dir, e
            )


@deprecated("Use `aimet_onnx.utils.OrtInferenceSession` instead")
def build_session(
    model: onnx.ModelProto,
    providers: List,
    user_onnx_libs: List[str] = None,
    path: str = None,
) -> OrtInferenceSession:
    """
    Build and return onnxruntime inference session

    :param model: onnx model
    :param providers: providers to execute onnxruntime
    :param user_onnx_libs: list of paths to user custom ONNX op libraries
    :param path: path where to store model external data
    """
    sess_options = create_ort_session_options_with_aimet_custom_ops()
    for lib in user_onnx_libs or []:
        sess_options.register_custom_ops_library(lib)

    return OrtInferenceSession(
        model, providers, session_options=sess_options, path=path
    )


class LazyExtractor(Extractor):
    """
    Wrapper Extractor to handle models with external data

    Wraps the original Extractor to keep data on disk during extraction and load weights on extracted model.
    """

    def __init__(self, model: onnx.ModelProto):
        """
        :param model: ONNX model to set up extractor for
        """
        self.model_dir: Optional[str] = None
        self.model_with_no_data: Optional[onnx.ModelProto] = None
        self.lazy_load_data: bool = False

        for init in model.graph.initializer:
            total_params = math.prod(init.dims)
            # Having more than 1B parameters is a good enough heuristic for large models
            # Goal is to avoid loading large models fully into memory
            if total_params > ONE_B_PARAMS:
                self.lazy_load_data = True
                break

        # If model is small enough, no need to lazy load data
        if not self.lazy_load_data:
            self.model_with_no_data = model
            super().__init__(self.model_with_no_data)
            return

        # Initialize extractor to lazily load data from disk post extraction
        self.model_dir = tempfile.mkdtemp()
        model_path = os.path.join(self.model_dir, "model.onnx")

        # Writes the .onnx file and external weight files to disk and
        # the same model object is restored with weights embedded so that it is fully self-contained again.
        save_model_with_external_weights(
            model,
            model_path,
            location=Path(model_path).name + ".data",
            size_threshold=1024**2,
            all_tensors_to_one_file=False,
            convert_attribute=True,
        )

        # Load model without external data for extraction
        self.model_with_no_data = onnx.load_model(model_path, load_external_data=False)
        super().__init__(self.model_with_no_data)

    def extract_model(
        self, input_names: List[str], output_names: List[str]
    ) -> onnx.ModelProto:
        """
        Extracts a sub-model from the original model given input and output names.

        :param input_names: input names of split graph
        :param output_names: output names of split graph
        :return: extracted model with external data loaded
        """
        # pylint: disable=protected-access
        model = super().extract_model(input_names, output_names)
        if self.lazy_load_data:
            # Load external data for extracted model
            load_external_data_for_model(model, self.model_dir)

        return model

    def __del__(self):
        """
        Clean up temp directory created to store model with external data
        """
        if self.model_dir is None:
            return

        try:
            shutil.rmtree(self.model_dir)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Error while cleaning up temp dir at %s: %s", self.model_dir, e
            )


class ModuleData:
    """
    Collect activation tensor for the given model and model input
    """

    def __init__(
        self,
        model: ModelProto,
        activation_name: str,
        providers: List,
        session_options: SessionOptions = None,
    ):
        """
        :param model: ONNX model
        :param activation_name: tensor corresponding to activation name to fetch
        :param providers: CPU/GPU execution providers
        :param session_options: ONNX Runtime session options
        """
        self._model = model
        self._activation_name = activation_name
        self._providers = providers
        self._session_options = session_options

    def collect_activation(self, model_input: Dict[str, List[np.ndarray]]) -> List:
        """
        Collect activation using the model_input

        :param model_input: Input to model
        :return: Activation corresponding to the model_input passed
        """

        handle = add_hook_to_get_activation(self._model.model, self._activation_name)
        sess = OrtInferenceSession(
            self._model.model, self._providers, self._session_options
        )
        if self._activation_name in model_input:
            # Workaround memory corruption bug in onnxruntime >= 1.19 when a graph output is also a graph input
            # https://github.com/microsoft/onnxruntime/issues/21922
            outputs = [model_input[self._activation_name]]
        else:
            outputs = sess.run([self._activation_name], model_input)
        remove_activation_hooks(self._model.model, handle)

        return outputs


def make_psnr_eval_fn(
    fp_session: InferenceSession,
    inputs: Iterable[Dict[str, np.ndarray]],
    output_indices: Union[int, List[int]] = 0,
) -> Callable[[InferenceSession], float]:
    """
    NOTE: To run inference through ORT, we need inputs in following format:
     [{input1: sample1, ...}, {input1: sample2, ...}, ...]

     Based on empirical evidence, PSNR of the primary output (i.e. bounding boxes
     in object detection models) is the most useful for analyzing per-layer sensitivity,
     Subsequent outputs (e.g., classification scores, class labels) may be misleading.

    :param fp_session: ORT inference session for floating-point model
    :param inputs: The model input samples
    :param output_indices: Index or list of indices of output tensors to compare (default is 0).
     if None, uses all float32 outputs
    :return: PSNR callback
    """
    inputs = list(inputs)
    fp_outputs = [fp_session.run(None, inp) for inp in inputs]

    if isinstance(output_indices, int):
        output_indices = [output_indices]

    if output_indices is None:
        output_indices = [
            idx for idx, out in enumerate(fp_outputs[0]) if out.dtype == np.float32
        ]

    def psnr_eval_fn(session: InferenceSession):
        """
        Callback function to compute PSNR between floating-point and quantized outputs.

        :param session: ORT inference session
        :return: PSNR value
        """
        sim_outputs = [session.run(None, inp) for inp in inputs]
        psnrs = []

        for idx in output_indices:
            fp_output = np.array([out[idx] for out in fp_outputs])
            sim_output = np.array([out[idx] for out in sim_outputs])
            psnrs.append(compute_psnr(fp_output, sim_output))

        return min(psnrs)

    return psnr_eval_fn


def get_torch_device(session: InferenceSession) -> torch.device:
    """
    Given the onnx session object, return corresponding torch device to use for optimization

    :param session: Onnx inference session
    :return: torch device
    """
    if "CUDAExecutionProvider" in session.get_providers():
        device_id = int(
            session.get_provider_options()
            .get("CUDAExecutionProvider", {})
            .get("device_id", "0")
        )
        return torch.device("cuda:" + str(device_id))
    return torch.device("cpu")


def map_np_dtype_to_torch(np_dtype: np.dtype) -> torch.dtype:
    """
    Maps a numpy dtype to torch.dtype

    :param np_dtype: Numpy dtype object
    :return: Corresponding torch dtype
    """
    dtype_map = {
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.int32,
        np.int64: torch.int64,
    }

    try:
        return dtype_map[np_dtype.type]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {np_dtype}") from exc


def map_torch_dtype_to_np(torch_dtype: torch.dtype) -> np.dtype:
    """
    Maps a PyTorch dtype to the corresponding NumPy dtype.

    :param torch_dtype: PyTorch dtype
    :return: Corresponding NumPy dtype
    """
    dtype_map = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }

    try:
        return dtype_map[torch_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}") from exc


@contextmanager
def _remove_initializer_data(model: onnx.ModelProto):
    # Hacky way to get around onnx.shape_inference.infer_shapes call as it doesn't work for model >2GB
    raw_data = {}

    try:
        # Store and clear raw_data from initializers
        for initializer in model.graph.initializer:
            if initializer.HasField("raw_data"):
                raw_data[initializer.name] = initializer.raw_data
                initializer.ClearField("raw_data")

        yield

    finally:
        for initializer in model.graph.initializer:
            if initializer.name in raw_data:
                initializer.raw_data = raw_data[initializer.name]


@contextmanager
def add_value_info(model: onnx.ModelProto):
    """
    Context manager to add value_info to model graph by performing shape inference.
    :param model: ONNX ModelProto
    """
    initial_value_info = model.graph.value_info

    # Remove weight data to allow shape inference (fails for models > 2GB)
    with _remove_initializer_data(model):
        model_copy = onnx.ModelProto()
        model_copy.CopyFrom(model)

    # Replace quantizers with Identity ops to allow shape inference
    for node in model_copy.graph.node:
        if node.op_type != "QcQuantizeOp":
            continue

        node.op_type = "Identity"
        node.ClearField("attribute")
        node.ClearField("domain")

    # Model must be topologically sorted prior to shape inference
    ONNXModel(model_copy).topological_sort()
    inferred_model = onnx.shape_inference.infer_shapes(model_copy)

    value_info = inferred_model.graph.value_info
    del model_copy, inferred_model

    try:
        # Update model's value info
        model.graph.ClearField("value_info")
        model.graph.value_info.extend(value_info)
        yield
    finally:
        # Restore original value info
        model.graph.ClearField("value_info")
        model.graph.value_info.extend(initial_value_info)
