# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utility functions for ONNX"""

import copy
import itertools
import platform
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Set, Union, Tuple, Callable, Optional, Sequence
from contextlib import contextmanager
import os
import pickle
import ml_dtypes
import numpy as np
import torch
import onnx
from onnx import helper, numpy_helper
from onnxruntime import SessionOptions, InferenceSession, OrtValue
import shutil
from aimet_onnx.common import libquant_info
from aimet_onnx.common import libpymo
from aimet_onnx.common.quantsim import compute_min_max_given_delta_offset
from aimet_onnx.common.defs import qtype, Int
from aimet_onnx.common.utils import AimetLogger, compute_psnr, deprecated
from aimet_onnx.common.onnx._utils import (  # pylint: disable=unused-import
    _ParamUtils,
    _get_node_attribute as get_node_attribute,
    contains_tensor_type,
    to_array,
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

# Not helper.tensor_dtype_to_np_dtype(BFLOAT16): onnx < 1.19 maps that to float32.
_BFLOAT16 = np.dtype(ml_dtypes.bfloat16)


class _NOT_SPECIFIED:
    pass


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


def find_shared_param_names(
    connected_graph, param_types: Optional[Tuple[str, ...]] = None
) -> set:
    """
    Find parameter names shared by multiple nodes.

    :param connected_graph: Connected graph to analyze
    :param param_types: If provided, only consider parameters matching these types (e.g. ("weight",)).
                        If None, consider all parameter types.
    :return: Set of parameter names used by more than one node
    """
    usage_count: Dict[str, int] = {}
    for op in connected_graph.ordered_ops:
        for product, ptype in op.parameters.values():
            if param_types is not None and ptype not in param_types:
                continue
            usage_count[product.name] = usage_count.get(product.name, 0) + 1

    return {name for name, count in usage_count.items() if count > 1}


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
    t_np = to_array(t)
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
        np_dtype = (
            _BFLOAT16
            if dtype == TensorProto.BFLOAT16
            else helper.tensor_dtype_to_np_dtype(dtype)
        )
        if shape:
            input_dict[name] = np.random.randn(*shape).astype(np_dtype)
        else:
            input_dict[name] = np.array(np.random.randn(*shape)).astype(np_dtype)
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
                    and to_array(node_graph.attribute[0].t) == 0
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


def register_aimet_custom_ops(session_options: SessionOptions) -> SessionOptions:
    """
    Registers aimet's custom-ops library onto ``session_options`` in place and returns it.

    :param session_options: Session options to register aimet's custom ops library onto.
    :return: ``session_options``, with aimet's custom ops registered
    """
    system = platform.system()
    if system == "Windows":
        lib_name = "libaimet_onnxrt_ops.dll"
    elif system == "Darwin":
        lib_name = "libaimet_onnxrt_ops.dylib"
    else:
        lib_name = "libaimet_onnxrt_ops.so"
    shared_library = os.path.join(
        os.path.dirname(libquant_info.__file__),
        lib_name,
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
        model: Union[ModelProto, str],
        providers: List,
        session_options: SessionOptions = None,
        path: str = None,
        save_as_external_data: bool = True,
    ):
        """
        Build and return onnxruntime inference session

        :param model: onnx model or path to onnx model
        :param providers: providers to execute onnxruntime
        :param session_options: onnxruntime session options
        :param path: path where to store model external data
        :param save_as_external_data: Whether to save model with external data.
        """

        self.model_dir: Optional[str] = None
        if isinstance(model, ModelProto):
            if path is None:
                self.model_dir = tempfile.mkdtemp()
                path = self.model_dir
            model_path = os.path.join(path, "model.onnx")
            # Use a deep copy to avoid memory accumulation in the original model.
            # Protobuf doesn't release memory when fields are cleared - it allocates new memory on reload.
            # This causes the ModelProto to grow by the weight size on each call. By operating on a
            # disposable copy, we take a temporary memory hit but avoid permanent growth in the original.
            model_copy = copy.deepcopy(model)
            onnx.save_model(
                model_copy,
                model_path,
                save_as_external_data=save_as_external_data,
                location=Path(model_path).name + ".data",
                # Prevents Split's split_sizes input from being externalized
                size_threshold=1024**2,
            )
            del model_copy
        else:
            model_path = model

        if session_options is None:
            session_options = register_aimet_custom_ops(SessionOptions())

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
    sess_options = register_aimet_custom_ops(SessionOptions())
    for lib in user_onnx_libs or []:
        sess_options.register_custom_ops_library(lib)

    return OrtInferenceSession(
        model, providers, session_options=sess_options, path=path
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
    output_indices: Union[int, List[int], None, type[_NOT_SPECIFIED]] = _NOT_SPECIFIED,
) -> Callable[[InferenceSession], float]:
    """
    NOTE: To run inference through ORT, we need inputs in following format:
     [{input1: sample1, ...}, {input1: sample2, ...}, ...]

     By default, all float32 and float16 outputs are compared and the minimum PSNR
     is returned.

    :param fp_session: ORT inference session for floating-point model
    :param inputs: The model input samples
    :param output_indices: Index or list of indices of output tensors to compare.
     If omitted or None, uses all float32 and float16 outputs
    :return: PSNR callback
    """
    inputs = list(inputs)

    if isinstance(output_indices, int):
        output_indices = [output_indices]

    if output_indices in (None, _NOT_SPECIFIED):
        float_output_indices = [
            idx
            for idx, out in enumerate(fp_session.get_outputs())
            if out.type in {"tensor(float)", "tensor(float16)"}
        ]
        if output_indices is _NOT_SPECIFIED and float_output_indices != [0]:
            warnings.warn(
                "The default output indices used by make_psnr_eval_fn changed in "
                "version 2.40 to consider all float outputs. To retain previous "
                "behavior, pass output_indices=0.",
                stacklevel=2,
            )
        if not float_output_indices:
            raise RuntimeError(
                "Cannot infer appropriate output indices for PSNR analysis for model "
                "with no float outputs. Provide output_indices explicitly."
            )

        output_indices = float_output_indices

    fp_outputs = [fp_session.run(None, inp) for inp in inputs]

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
        ml_dtypes.bfloat16: torch.bfloat16,
    }

    try:
        return dtype_map[np_dtype.type]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {np_dtype}") from exc


def _onnx_element_type(np_dtype: np.dtype) -> Union[int, np.dtype]:
    """
    ``element_type`` to pass to ``IOBinding.bind_input``/``bind_output`` for ``np_dtype``.

    ORT cannot map ``ml_dtypes.bfloat16`` to a tensor type, so bfloat16 is given as its
    ONNX TensorProto code instead.
    """
    if np_dtype == _BFLOAT16:
        return TensorProto.BFLOAT16
    return np_dtype


def _run_session_to_numpy(
    session: InferenceSession, inputs: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """
    ``session.run(None, inputs)`` that also accepts and returns bfloat16 arrays.

    ``InferenceSession.run`` cannot convert bfloat16 tensors to or from numpy, so they are
    passed as OrtValues and read back through DLPack as ``ml_dtypes.bfloat16`` arrays.
    """
    ort_inputs = {
        name: OrtValue.ortvalue_from_numpy_with_onnx_type(
            np.ascontiguousarray(value).view(np.uint16), TensorProto.BFLOAT16
        )
        if value.dtype == _BFLOAT16
        else OrtValue.ortvalue_from_numpy(np.ascontiguousarray(value))
        for name, value in inputs.items()
    }
    outputs = []
    for output in session.run_with_ort_values(None, ort_inputs):
        if output.data_type() == "tensor(bfloat16)":
            bits = torch.from_dlpack(output._ortvalue).view(torch.int16)  # pylint: disable=protected-access
            outputs.append(bits.cpu().numpy().view(_BFLOAT16))
        else:
            outputs.append(output.numpy())
    return outputs


def _numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    """
    ``torch.from_numpy(array)`` that also accepts ``ml_dtypes.bfloat16`` arrays.

    ``torch.from_numpy`` rejects bfloat16 arrays, so their raw bits are viewed as
    ``torch.bfloat16`` instead. Shares memory with ``array``, like ``torch.from_numpy``.
    """
    if array.dtype == _BFLOAT16:
        bits = np.ascontiguousarray(array).view(np.int16)
        return torch.from_numpy(bits).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    ``tensor.numpy()`` that also accepts ``torch.bfloat16`` tensors.

    numpy has no bfloat16, so the raw bits of a bfloat16 tensor are viewed as an
    ``ml_dtypes.bfloat16`` array instead. ``tensor`` must be on the CPU.
    """
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(_BFLOAT16)
    return tensor.numpy()


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


def _get_qdq_pair_initializers(graph: GraphProto) -> Set[str]:
    """
    Return the set of initializer names that serve as scale or zero_point for
    at least one QuantizeLinear–DequantizeLinear pair.

    A QDQ pair is identified by ``QuantizeLinear.output[0] == DequantizeLinear.input[0]``.
    Scale (input[1]) and zero_point (input[2]) of such a pair must remain shared
    between Q and DQ; duplicating them would decouple the encode/decode parameters
    and break quantization correctness.

    :param graph: ONNX GraphProto to inspect
    :return: Set of initializer names that must not be duplicated
    """
    producers = {out: node for node in graph.node for out in node.output}
    init_names = {init.name for init in graph.initializer}
    protected: Set[str] = set()
    for node in graph.node:
        if node.op_type != "DequantizeLinear" or not node.input:
            continue
        q_node = producers.get(node.input[0])
        if q_node is None or q_node.op_type != "QuantizeLinear":
            continue
        for i in range(1, max(len(q_node.input), len(node.input))):
            if i < len(q_node.input) and q_node.input[i] in init_names:
                protected.add(q_node.input[i])
            if i < len(node.input) and node.input[i] in init_names:
                protected.add(node.input[i])
    return protected


def duplicate_shared_initializers(graph: GraphProto) -> int:
    """
    Duplicate initializers that are shared across multiple nodes.

    For each initializer referenced by more than one node, create an independent
    copy for every usage beyond the first and update the corresponding node inputs
    to point to the new copies.  This ensures each node owns a unique tensor,
    which is required by some downstream passes (e.g. per-node quantization).

    :param graph: ONNX GraphProto to modify in-place
    :return: Number of duplicate initializers created
    """
    if not graph.node:
        return 0

    # Exclude scale/zp initializers that belong to a QDQ pair — they must stay
    # shared between QuantizeLinear and its paired DequantizeLinear.
    qdq_protected = _get_qdq_pair_initializers(graph)
    initializer_map = {
        init.name: init for init in graph.initializer if init.name not in qdq_protected
    }
    if not initializer_map:
        return 0

    # Map each tensor name (initializer or intermediate) to its direct consumers
    direct_consumers: Dict[str, List[Tuple[NodeProto, int]]] = {}
    for node in graph.node:
        for input_idx, input_name in enumerate(node.input):
            if input_name:
                direct_consumers.setdefault(input_name, []).append((node, input_idx))

    # For each initializer, collect its leaf (non-Identity) consumer usages by
    # traversing Identity passthrough chains. Shape consumers only read tensor
    # metadata and are skipped to avoid unnecessary duplication.
    usage_map: Dict[str, List[Tuple[NodeProto, int]]] = {}
    for init_name in initializer_map:
        stack: List[str] = [init_name]
        while stack:
            tensor_name = stack.pop()
            for consumer, idx in direct_consumers.get(tensor_name, []):
                if consumer.op_type == "Shape":
                    continue
                if consumer.op_type == "Identity":
                    stack.append(consumer.output[0])
                else:
                    usage_map.setdefault(init_name, []).append((consumer, idx))

    duplicate_count = 0
    existing_names = (
        {init.name for init in graph.initializer}
        | {out for node in graph.node for out in node.output}
        | {inp.name for inp in graph.input}
    )

    shared_tensor_count = 0
    for tensor_name, usages in usage_map.items():
        if len(usages) <= 1:
            continue
        shared_tensor_count += 1

        original = initializer_map[tensor_name]
        logger.debug(
            "Duplicating shared initializer '%s' (shape=%s) for %d additional usages",
            tensor_name,
            list(original.dims),
            len(usages) - 1,
        )

        counter = 0
        for node, input_idx in usages[1:]:
            counter += 1
            new_name = f"{tensor_name}_copy_{counter}"
            while new_name in existing_names:
                counter += 1
                new_name = f"{tensor_name}_copy_{counter}"
            existing_names.add(new_name)

            # add() + CopyFrom() merges fields directly into an already-parented slot.
            new_init = graph.initializer.add()
            new_init.CopyFrom(original)
            new_init.name = new_name
            node.input[input_idx] = new_name
            duplicate_count += 1

    logger.info(
        "duplicate_shared_initializers: created %d duplicate initializer(s) for %d shared tensor(s)",
        duplicate_count,
        shared_tensor_count,
    )
    return duplicate_count


def numpy_to_TfEncoding(
    scale: np.ndarray, offset: np.ndarray, dtype: qtype
) -> list[libpymo.TfEncoding]:
    """
    Converts scale offset arrays to a list of TfEncoding objects
    """
    dtype = qtype.as_qtype(dtype)
    if not isinstance(dtype, Int):
        raise NotImplementedError(
            f"Converting {dtype} encodings to TfEncoding is not supported"
        )
    _, bitwidth = dtype.to_legacy_repr()

    # Flatten arrays
    scales_flat = scale.flatten()
    offsets_flat = offset.flatten()

    # Vectorized min/max computation
    min_vals, max_vals = compute_min_max_given_delta_offset(
        scales_flat, offsets_flat, bitwidth, False, False
    )

    # Construct TfEncoding objects
    encodings = []
    for scale, offset, min_val, max_val in zip(
        scales_flat, offsets_flat, min_vals, max_vals
    ):
        encoding = libpymo.TfEncoding()

        encoding.bw = bitwidth
        encoding.min = min_val
        encoding.max = max_val
        encoding.delta = scale
        encoding.offset = offset
        encodings.append(encoding)

    return encodings


def numpy_from_TfEncoding(
    encodings: List[libpymo.TfEncoding], shape: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts list of TfEncoding objects to scale & offset numpy arrays with the given shape
    """
    num_encodings = np.prod(shape)
    assert len(encodings) == num_encodings
    scale = np.array([enc.delta for enc in encodings]).reshape(shape)
    offset = np.array([enc.offset for enc in encodings]).reshape(shape)
    return scale, offset
