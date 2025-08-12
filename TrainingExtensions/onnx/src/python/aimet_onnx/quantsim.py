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
"""Implementation for simulating models running on Quantized hardware"""

# pylint: disable=wrong-import-order
import contextlib
import tempfile
from pathlib import Path
import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    overload,
    Tuple,
    TypeVar,
    Union,
    Set,
    Sequence,
    Iterable,
)
from functools import wraps
import itertools
import json
import warnings
import numpy as np
import onnx

from onnx import helper
from onnx.numpy_helper import to_array
import onnxruntime as ort
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from packaging import version

from aimet_common import libpymo, quantsim
from aimet_common import libquant_info
from aimet_common.defs import (
    QuantScheme,
    QuantizationDataType,
    qtype,
    QTYPE_ALIASES,
    Float,
    int8,
    EncodingType,
    _quant_scheme_aliases,
)
from aimet_common.onnx._utils import (
    _add_onnx_qdq_nodes,
    _remove_onnx_qdq_nodes,
    _is_grid_preserving_op,
)
from aimet_common.quantsim import (
    extract_global_quantizer_args,
    VALID_ENCODING_VERSIONS,
    _INT32_MINIMUM_SCALE,
    _is_bias_out_of_int32_range,
    _get_adjusted_weight_scale,
)
from aimet_common.utils import save_json_yaml, AimetLogger, _red, deprecated
from aimet_common.quant_utils import _convert_encoding_format_0_6_1_to_1_0_0
from aimet_common.quantsim_config.quantsim_config import _config_file_aliases
from aimet_common.connected_graph.product import Product
from aimet_common.onnx._utils import _convert_version
from aimet_onnx import utils
from aimet_onnx.meta.operations import Op
from aimet_onnx.meta.utils import (
    get_op_given_param_name,
    get_param_shape_using_connected_graph,
)
from aimet_onnx.meta.connectedgraph import (
    ConnectedGraph,
    _get_matmul_add_bias_idx,
    WEIGHT_INDEX,
)
from aimet_onnx.qc_quantize_op import (
    QcQuantizeOp,
    OpMode,
    TensorQuantizerParams,
    GroupedBlockQuantizeDequantize,
    _EncodingMismatchInfo,
    _2_0_0_json_encoding_to_TfEncoding_list,
)
from aimet_onnx.quantsim_config.quantsim_config import QuantSimConfigurator
from aimet_onnx.utils import (
    make_dummy_input,
    save_model_with_external_weights,
    add_hook_to_get_activation,
    remove_activation_hooks,
    build_session,
)

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# pylint: disable=no-name-in-module, ungrouped-imports, too-many-lines
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import ModelProto
else:
    from onnx.onnx_pb import ModelProto

# List of ops whose outputs are not to be quantized
op_outputs_to_ignore = [
    "branch",
    "Flatten",
    "Gather",
    "Reshape",
    "Shape",
    "Unsqueeze",
    "Squeeze",
    "Split",
    "Compress",
    "Tile",
    "Transpose",
    "Identity",
]

# List of ops whose params are not to be quantized
op_params_to_ignore = ["Resize"]

allowed_op_type_for_per_channel = ["Conv", "Gemm", "MatMul", "ConvTranspose"]

# List of op types whose input and output quantizers to be tied
op_types_to_tie_qtzrs = [
    "Concat",
    "CropAndResize",
    "MaxPool",
    "AveragePool",
    "Resize",
    "Max",
    "ReduceMax",
    "Min",
    "ReduceMin",
    "ScatterElements",
    "Upsample",
]
_tie_qtzrs = False

data_types_to_quantize = [np.float32]

_DEPRECATED_ARGS = {
    "rounding_mode",
    "default_param_bw",
    "default_activation_bw",
    "use_symmetric_encodings",
    "default_data_type",
    "use_cuda",
    "device",
}


def _allow_deprecated_args(func):
    @wraps(func)
    def init_wrapper(self, model, *args, **kwargs):
        # Quantsim constructor called using old function signature
        if args or (kwargs.keys() & _DEPRECATED_ARGS):
            warnings.warn(
                _red(
                    f"{func.__qualname__}() was called using a deprecated function signature. This will raise an error in future releases."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs = _parse_deprecated_args(*args, **kwargs)

        return func(self, model, **kwargs)

    return init_wrapper


def _parse_deprecated_args(
    dummy_input: Optional[Dict[str, np.ndarray]] = None,
    quant_scheme: QuantScheme = QuantScheme.min_max,
    rounding_mode: str = None,
    default_param_bw: int = None,
    default_activation_bw: int = None,
    use_symmetric_encodings: bool = None,  # pylint:disable = unused-argument
    use_cuda: bool = None,
    device: int = None,
    config_file: Optional[str] = None,
    default_data_type: QuantizationDataType = None,
    user_onnx_libs: List[str] = None,
    providers: Optional[Sequence[str | Tuple[str, Dict[Any, Any]]]] = None,
    path: Optional[str] = None,
    **kwargs,
):
    # Args which are now keyword-only
    kwargs["dummy_input"] = dummy_input
    kwargs["quant_scheme"] = quant_scheme
    kwargs["config_file"] = config_file
    kwargs["user_onnx_libs"] = user_onnx_libs
    kwargs["providers"] = providers
    kwargs["path"] = path

    # Unused argument
    kwargs.pop("use_symmetric_encodings", None)

    # Legacy behavior for already-deprecated rounding
    if rounding_mode and rounding_mode != "nearest":
        raise TypeError("'rounding_mode' parameter is no longer supported.")

    # Providers is not compatible with `use_cuda` or `device`
    if providers and (use_cuda is not None or device is not None):
        raise RuntimeError(
            f"Cannot provide `providers` and { {'use_cuda', 'device'} } at the same time."
        )

    # If user has explicitly passed use_cuda=True, allow it
    if use_cuda:
        kwargs["providers"] = [
            ("CUDAExecutionProvider", {"device_id": device or 0}),
            "CPUExecutionProvider",
        ]

    # Deprecated args related to dtype/bitwidth
    deprecated_dtype_args = {
        "default_param_bw": default_param_bw,
        "default_activation_bw": default_activation_bw,
        "default_data_type": default_data_type,
    }
    deprecated_dtype_args = {
        key: value for key, value in deprecated_dtype_args.items() if value is not None
    }
    new_dtype_args = kwargs.keys() & {"param_type", "activation_type"}

    # Don't allow old and new dtype arguments
    if deprecated_dtype_args and new_dtype_args:
        raise RuntimeError(
            f"Received deprecated keyword arguments {set(deprecated_dtype_args.keys())} which are incompatible with keyword arguments {new_dtype_args}"
        )

    # Convert legacy dtype specification to qtype
    if deprecated_dtype_args:
        param_bw = deprecated_dtype_args.pop("default_param_bw", 8)
        act_bw = deprecated_dtype_args.pop("default_activation_bw", 8)
        dtype = deprecated_dtype_args.pop("default_data_type", QuantizationDataType.int)
        kwargs["param_type"] = qtype.from_legacy_repr(dtype, param_bw)
        kwargs["activation_type"] = qtype.from_legacy_repr(dtype, act_bw)

    return kwargs


@contextlib.contextmanager
def _apply_constraints(flag: bool):
    """
    Apply runtime specific constraints.
    For certain ``op_types_to_tie_qtzrs``, runtime has constraints to have same encodings for
     input and output quantizers.

    NOTE: Default setting doesn't apply these constraints.
    """
    global _tie_qtzrs  # pylint: disable=global-statement
    orig_flag = _tie_qtzrs
    try:
        _tie_qtzrs = flag
        yield
    finally:
        _tie_qtzrs = orig_flag


class _NOT_SPECIFIED:
    pass


@contextlib.contextmanager
def compute_encodings(sim: "QuantizationSimModel"):
    r"""
    Computes encodings for all quantizers in the model.

    Under this context manager, :class:`QuantizationSimModel` will
    observe all inputs that run through the model to calibrate
    the quantization encoding of each quantizer.

    Example:

        >>> sim = QuantizationSimModel(...)
        >>> with compute_encodings(sim):
        ...     for input in dataset:
        ...         _ = sim.session.run(None, {"input": input})
    """
    enabled_quantizers = {
        name: q for name, q in sim.qc_quantize_op_dict.items() if q.enabled
    }
    for op_name, qc_op in enabled_quantizers.items():
        qc_op.reset_encoding_stats()
        if op_name in sim.activation_names:
            qc_op.op_mode = OpMode.updateStats
        else:
            qc_op.op_mode = OpMode.oneShotQuantizeDequantize
            if qc_op.is_encoding_frozen():
                qc_op.op_mode = OpMode.quantizeDequantize

    yield

    for op_name, qc_op in enabled_quantizers.items():
        if (
            qc_op.data_type == QuantizationDataType.int
            and not qc_op.is_encoding_frozen()
        ):
            qc_op.compute_encodings()
        qc_op.op_mode = OpMode.quantizeDequantize


# pylint: disable=missing-class-docstring, too-many-arguments, too-many-locals, too-many-instance-attributes
class QuantizationSimModel:
    __doc__ = f"""
    Class that simulates the quantized model execution on a target hardware backend.

    Args:
        model (onnx.ModelProto): ONNX ModelProto to quantize
        param_type (qtype | str): quantized type to use for parameter tensors.
            Can be {{ {", ".join(QTYPE_ALIASES.keys())} }} or :class:`aimet_onnx.qtype`
        activation_type (qtype | str): quantized type to use for activation tensors.
            Can be {{ {", ".join(QTYPE_ALIASES.keys())} }} or :class:`aimet_onnx.qtype`
        quant_scheme (QuantScheme | str): Quantization scheme to use for calibration.
            Can be {{ {", ".join(_quant_scheme_aliases.keys() - {"tf", "percentile"})} }} or :class:`QuantScheme`
        config_file (str, optional): File path or alias of the configuration file.
            Alias can be one of {{ {", ".join(_config_file_aliases.keys())} }} (Default: `"default"`)
        dummy_input (Dict[str, np.ndarray], optional): Sample input to the model. Only needed for non shape-inferable models with parameterized shapes
        user_onnx_libs (List[str], optional): List of paths to all compiled ONNX custom ops libraries
        providers (List, optional): Onnxruntime execution providers to use when building InferenceSession.
            If `None`, default provider is "CPUExecutionProvider"
        path (str, optional): Directory to save temporary artifacts.
    """

    @_allow_deprecated_args
    def __init__(
        self,
        model: ModelProto,
        *,
        param_type: Union[str, qtype] = int8,
        activation_type: Union[str, qtype] = int8,
        quant_scheme: Union[str, QuantScheme] = QuantScheme.min_max,
        config_file: Optional[str] = None,
        dummy_input: Optional[Dict[str, np.ndarray]] = None,
        user_onnx_libs: Optional[List[str]] = None,
        providers: Optional[Sequence[str | Tuple[str, Dict[Any, Any]]]] = None,
        path: Optional[str] = None,
    ):
        if isinstance(quant_scheme, str):
            quant_scheme = QuantScheme.from_str(quant_scheme)

        if isinstance(model, ModelProto):
            model = ONNXModel(model)

        if isinstance(param_type, str):
            param_type = qtype.from_string(param_type)

        if isinstance(activation_type, str):
            activation_type = qtype.from_string(activation_type)

        for dtype in (param_type, activation_type):
            if dtype in QTYPE_ALIASES.values():
                continue

            # Only aliased float types (fp16, fp32) are supported float types
            if isinstance(dtype, Float):
                raise RuntimeError(f"Simulating {dtype} quantization is not supported.")

            logger.warning(
                "Exporting {dtype} quantization to onnx graph is not supported"
            )

        if providers is None:
            providers = ["CPUExecutionProvider"]

        op_domain = "aimet.customop.cpu"
        for provider in providers:
            if (
                provider == "CUDAExecutionProvider"
                or provider[0] == "CUDAExecutionProvider"
            ):
                op_domain = "aimet.customop.cuda"

        self.model = model
        self._op_domain = op_domain
        self.providers = providers

        if not dummy_input:
            dummy_input = make_dummy_input(self.model.model)

        self.qc_quantize_op_dict = {}
        self.connected_graph = ConnectedGraph(self.model)
        self._quant_scheme = quant_scheme
        self._param_type = param_type
        self._activation_type = activation_type
        self._user_onnx_libs = user_onnx_libs
        self.param_names = []
        self.input_quantizers_name = []
        self.activation_names = []
        self.activation_dtypes = {}
        self._path = path
        if self._path:
            os.makedirs(self._path, exist_ok=True)

        # Get names of parameters and activations to quantize
        self._get_param_names()
        self._get_activations_to_quantize(dummy_input)

        self._add_quantization_nodes()

        # Apply configurations based on provided config file.
        quantsim_configurator = self._add_configuration_(config_file)
        self._hw_version = quantsim_configurator._get_hw_version()
        self._supported_kernels = quantsim_configurator.get_supported_kernels()
        self._op_to_supported_kernel = (
            quantsim_configurator.get_op_to_supported_kernels()
        )
        self.quant_args = extract_global_quantizer_args(
            quant_scheme, quantsim_configurator
        )
        self._apply_param_symmetry_to_inputs(quantsim_configurator)
        self._apply_exception_rules()
        if _tie_qtzrs:
            self._tie_quantizers_for_op_types(op_types_to_tie_qtzrs)

        # Build onnxruntime inference session
        self.session = build_session(
            self.model.model,
            self.providers,
            user_onnx_libs=self._user_onnx_libs,
            path=self._path,
        )

    @classmethod
    def _from_onnx_qdq(cls, model: ModelProto, **kwargs) -> "QuantizationSimModel":
        """
        Create sim from onnx QDQ model with following strategy

        1. Remove Q/DQ nodes from model
        2. Extract encodings from the removed Q/DQ nodes
        3. Create QuantizationSimModel
        4. Load extracted encodings to sim

        Args:
            model: ONNX model that contains QuantizeLinear/DequantizeLinear
            **kwargs: same as QuantizationSimModel.__init__
        """
        # pylint: disable=protected-access

        # Removes Q/DQ node from model and extract them into 2.0.0 json encoding
        encodings = _remove_onnx_qdq_nodes(model)

        # Create sim
        sim = QuantizationSimModel(model, **kwargs)

        quantizable_tensor_names = set(
            name
            for name, qtzr in sim.qc_quantize_op_dict.items()
            if qtzr and qtzr.enabled
        )
        bias_names = set(
            bias.name
            for op in sim.connected_graph.get_all_ops().values()
            for _, bias in [sim._get_weight_and_bias(op)]
            if bias is not None
        )
        encoding_names = set(enc["name"] for enc in encodings)
        excess_encodings = encoding_names - (quantizable_tensor_names | bias_names)

        if excess_encodings:
            raise NotImplementedError(
                "Unexpected QuantizeLinear/DequantizeLinear nodes were found "
                f"for the following tensors: {excess_encodings}"
            )

        lpbq_weights = {
            enc["name"]: enc for enc in encodings if "per_channel_float_scale" in enc
        }

        def get_lpbq_params(op: Op):
            for inp in op.inputs:
                if inp.name in lpbq_weights:
                    enc = lpbq_weights[inp.name]
                    *_, bitwidth = enc["output_dtype"].split("int")
                    bitwidth = int(bitwidth)
                    decompressed_bw = bitwidth * 2
                    block_size = enc["block_size"]
                    return bitwidth, decompressed_bw, block_size
            return None, None, None

        if lpbq_weights:
            _set_grouped_blockwise_quantization_for_weights(
                sim,
                get_lpbq_params,
                strict=True,
            )

        # Load encodings to sim
        for enc in encodings:
            qtzr = sim.qc_quantize_op_dict[enc["name"]]

            if enc["name"] in bias_names and enc["output_dtype"] == "int32":
                qtzr.enabled = False
                continue

            channel_axis = block_axis = None
            block_size = enc.get("block_size")
            if "axis" in enc:
                if block_size is None:
                    channel_axis = enc["axis"]
                else:
                    param = utils.ParamUtils.get_param_by_name(model, enc["name"])
                    if not param:
                        raise RuntimeError(
                            "Creating QuantizationSimModel from onnx models with "
                            "blockwise QuantizeLinear/DequantizeLinear is supported "
                            f"with static 2D weights. Got dynamic input {enc['name']}"
                        )
                    if len(param.dims) != 2:
                        raise RuntimeError(
                            "Creating QuantizationSimModel from onnx models with "
                            "blockwise QuantizeLinear/DequantizeLinear is supported "
                            f"with 2D weights. Got {len(param.dims)}D weight {param.name}"
                        )
                    block_axis = enc["axis"]
                    channel_axis = 0 if block_axis in (1, -1) else 1

            if channel_axis is not None:
                if not qtzr.tensor_quantizer_params:
                    raise RuntimeError(
                        f"Per-channel quantization for tensor {enc['name']} is not supported"
                    )

                qtzr.tensor_quantizer_params.channel_axis = channel_axis
                qtzr.enable_per_channel_quantization()

                if block_axis is not None:
                    qtzr.tensor_quantizer_params.block_axis = block_axis
                    qtzr._enable_blockwise_quantization(block_size)

            qtzr.load_encodings(_2_0_0_json_encoding_to_TfEncoding_list(enc))
            qtzr.freeze_encodings()

        return sim

    def get_supported_kernels(self) -> Dict:
        """
        Return _supported_kernels parsed from the config file
        :return: Dictionary containing supported_kernels
        """
        return self._supported_kernels

    def _add_configuration_(self, config_file: str):
        """
        Add configuration based on config file

        :param config_file: Path to Configuration file for model quantizers
        """
        quantsim_configurator = QuantSimConfigurator(
            self.model,
            self.connected_graph,
            config_file,
            self._param_type,
            self._activation_type,
        )
        quantsim_configurator.configure_quantizers(
            self.qc_quantize_op_dict,
            self.param_names,
            self.activation_names,
            self.input_quantizers_name,
        )

        return quantsim_configurator

    def _get_param_names(self):
        """
        Get the names of params
        """
        valid_ops = self._get_ops_with_parameter()
        for op in valid_ops:
            for param_info in op.parameters.values():
                param, _ = param_info
                if param.name and param.name not in self.param_names:
                    self.param_names.append(param.name)

    def _get_ops_with_parameter(self) -> List[Op]:
        """
        Gets ops with parameters to add quantization nodes for

        :return: Connected graph ops
        """
        valid_ops = list(self.connected_graph.get_all_ops().values())
        return valid_ops

    def _get_activations_to_quantize(self, dummy_input: Dict[str, np.ndarray]):
        """
        Get the names of activations to quantize

        :param dummy_input: Sample input to be run through the model
        """
        try:
            self.activation_dtypes = self._infer_activation_dtypes()
        except onnx.shape_inference.InferenceError:
            self.activation_dtypes = self._observe_activation_dtypes(dummy_input)

        self.input_name_to_nodes = self.model.input_name_to_nodes()
        self.output_name_to_node = self.model.output_name_to_node()

        # Capture model inputs
        for node in self.model.graph().input:
            name = node.name
            if (
                name not in self.activation_names
                and name not in self.param_names
                and self._is_tensor_quantizable(name)
            ):
                self.activation_names.append(name)

        # Capture intermediate activations and model outputs
        for node in self.model.nodes():
            for name in node.input:
                if (
                    name not in self.activation_names
                    and name not in self.param_names
                    and self._is_tensor_quantizable(name)
                ):
                    self.activation_names.append(name)
                    self.input_quantizers_name.append(name)

            for name in node.output:
                if (
                    name not in self.activation_names
                    and name not in self.param_names
                    and self._is_tensor_quantizable(name)
                ):
                    self.activation_names.append(name)

        # Rename model output node
        for node in self.model.graph().output:
            if node.name in self.activation_names:
                node.name += "_updated"

    def _is_quantizable_dtype(self, name: str) -> bool:
        # Check if the tensor data-type can be quantized
        if name in self.model.get_initializer_name_set():
            np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
                self.model.get_initializer(name).data_type
            )
            if np_dtype not in data_types_to_quantize:
                return False
        else:  # dynamic activation
            if (
                name not in self.activation_dtypes
                or self.activation_dtypes[name] not in data_types_to_quantize
            ):
                return False

        return True

    def _is_tensor_quantizable(self, name: str) -> bool:
        """
        Checks whether the given tensor should be quantized

        :param name: Name of the tensor
        :return: True if the tensor should be quantized
        """
        if not self._is_quantizable_dtype(name):
            return False

        # Check if the tensor is param to certain ops (eg: Resize)
        consumer_nodes = self.input_name_to_nodes.get(name)
        if consumer_nodes:
            for consumer_node in consumer_nodes:
                if (
                    consumer_node.op_type in op_params_to_ignore
                    and consumer_node.input[0] != name
                ):  # except first input rest are params (only valid for unary ops)
                    return False

        # Check if the tensor is output of certain ops
        producer_node = self.output_name_to_node.get(name)
        if producer_node and producer_node.op_type in op_outputs_to_ignore:
            return False

        return True

    def _infer_activation_dtypes(self):
        """
        Get the data type for each activation through shape inference
        """
        with tempfile.TemporaryDirectory(dir=self._path) as tempdir:
            save_path = os.path.join(tempdir, "inferred_model.onnx")
            save_model_with_external_weights(
                self.model.model, save_path, location=Path(save_path).name + ".data"
            )
            onnx.shape_inference.infer_shapes_path(save_path)
            # Do not load the weights for the shape inference model, we only need to access the graph's `value_info`
            inferred_model = onnx.load(save_path, load_external_data=False)

        activation_dtypes = {}
        for val_info in itertools.chain(
            inferred_model.graph.value_info,
            inferred_model.graph.input,
            inferred_model.graph.output,
        ):
            act_name = val_info.name
            dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[
                val_info.type.tensor_type.elem_type
            ]
            activation_dtypes[act_name] = dtype
        return activation_dtypes

    def _observe_activation_dtypes(self, dummy_input: Dict[str, np.ndarray]):
        """
        Get the data type for each activation by returning all activations

        :param dummy_input: Sample input to run through the model
        """
        activations = utils.get_graph_intermediate_activations(self.model.graph())
        hooks = []
        for name in activations:
            hooks.append(add_hook_to_get_activation(self.model.model, name))
        sess = build_session(
            self.model.model,
            ["CPUExecutionProvider"],
            user_onnx_libs=self._user_onnx_libs,
            path=self._path,
        )
        outputs = sess.run(None, dummy_input)

        activation_dtypes = {}
        for idx, node in enumerate(self.model.graph().output):
            act_name = node.name
            dtype = outputs[idx].dtype
            activation_dtypes[act_name] = dtype
        remove_activation_hooks(self.model.model, hooks)
        return activation_dtypes

    def _add_quantization_nodes(self):
        """
        Call insert functions for quantization nodes
        """
        self._insert_param_quantization_nodes()
        self._insert_activation_quantization_nodes()

    def _replace_input_of_all_nodes(self, old_name, new_name):
        if old_name not in self.connected_graph.get_all_products():
            raise ValueError(
                f"Tensor name {old_name} was not found in graph tensors "
                f"{self.connected_graph.get_all_products().keys()}."
            )

        product = self.connected_graph.get_all_products()[old_name]
        for consumer in product.consumers:
            node = consumer.get_module()
            for idx, tensor in enumerate(node.input):
                if tensor == old_name:
                    node.input[idx] = new_name

    def _insert_param_quantization_nodes(self):
        """
        Insert quantization node for each param tensor
        """
        for name in self.param_names:
            self._insert_quantizer(name, is_param=True)

    def _create_tensor_quantizer_params(self, param_name: str):
        """
        Creates TensorQuantizerParams object for QcQuantizeOp and QDQ node

        :param param_name: Name of the parameter for which the quant info object will be created
        :return: TensorQuantizerParams object
        """
        op = get_op_given_param_name(self.connected_graph, param_name)
        if not op:
            return None

        param_shape = get_param_shape_using_connected_graph(
            self.connected_graph, param_name
        )
        tensor_quantizer_params = TensorQuantizerParams(param_shape)

        if len(param_shape) == 1:
            tensor_quantizer_params.channel_axis = 0
            tensor_quantizer_params.block_axis = None
        else:
            channel_axis, block_axis = self._get_quantization_axes(op)
            tensor_quantizer_params.channel_axis = channel_axis
            tensor_quantizer_params.block_axis = block_axis

        return tensor_quantizer_params

    @staticmethod
    def _get_quantization_axes(op: Op) -> Tuple[Optional[int], Optional[int]]:
        """
        Gets quantization axes for per-channel and blockwise quantization

        :param op: Connected graph op
        :return: (channel axis, block axis)
        """
        if op.type in ["Conv"]:
            return 0, 1
        if op.type in ["ConvTranspose"]:
            return 1, 0
        if op.type in ["Gemm"]:
            if op.transposed_params:
                return 0, 1
            return 1, 0
        if op.type in ["MatMul"]:
            return -1, -2

        return None, None

    def _insert_activation_quantization_nodes(self):
        """
        Insert quantization node for each activation tensor
        """
        for name in self.activation_names:
            self._insert_quantizer(name, is_param=False)

    def _insert_quantizer(self, input_name: str, is_param: bool):
        """
        Inserts a quantizer for tensor `input_name` in the graph and adds it to `self.qc_quantize_op_dict`

        self.session must be rebuilt after calling this for changes to take effect.
        """
        if input_name in self.qc_quantize_op_dict:
            raise RuntimeError(f"Quantizer already exists for tensor {input_name}")

        # TODO: Revisit all tensor/node naming
        node_name = "QcQuantizeOp_" + input_name
        if is_param:
            output_name = input_name + "_qdq"
            op_mode = OpMode.oneShotQuantizeDequantize
            dtype, bitwidth = self._param_type.to_legacy_repr()
            tensor_quantizer_params = self._create_tensor_quantizer_params(input_name)
        else:
            output_name = input_name + "_updated"
            op_mode = OpMode.updateStats
            dtype, bitwidth = self._activation_type.to_legacy_repr()
            tensor_quantizer_params = None

        quant_info = libquant_info.QcQuantizeInfo()
        self._replace_input_of_all_nodes(input_name, output_name)
        custom_node = helper.make_node(
            op_type="QcQuantizeOp",
            inputs=[input_name],
            outputs=[output_name],
            name=node_name,
            domain=self._op_domain,
            op_name=input_name,
            quant_info=libpymo.PtrToInt64(quant_info),
        )
        self.model.add_node(custom_node)
        self.qc_quantize_op_dict[input_name] = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=self._quant_scheme,
            op_mode=op_mode,
            bitwidth=bitwidth,
            tensor_quantizer_params=tensor_quantizer_params,
        )
        self.qc_quantize_op_dict[input_name].data_type = dtype

    @staticmethod
    @deprecated("Use `aimet_onnx.utils.build_session` instead")
    def build_session(
        model: onnx.ModelProto,
        providers: List,
        user_onnx_libs: List[str] = None,
        path: str = None,
    ):
        """
        Build and return onnxruntime inference session
        :param model: onnx model
        :param providers: providers to execute onnxruntime
        :param user_onnx_libs: list of paths to user custom ONNX op libraries
        :param path: path where to store model external data
        """
        return build_session(model, providers, user_onnx_libs=user_onnx_libs, path=path)

    def get_qc_quantize_op(self):
        """
        Return dict of qc quantize ops
        """
        return self.qc_quantize_op_dict

    def get_op_quantizers(self, op: Op) -> Tuple[List, List, Dict]:
        """
        This function returns the input, output and param quantizers of the given connected graph op.

        :param op: Connected Graph Op
        :return: list of input quantizers, list of output quantizers and dictionary of param quantizers
        """
        input_quantizers = []
        output_quantizers = []
        param_quantizers = {}

        # Capture as input quantizer if tensor is not a layer output or parameter
        for cg_product in op.inputs:
            if not cg_product.producer and not cg_product.is_parm:
                input_name = cg_product.name
                if input_name in self.qc_quantize_op_dict:
                    input_quantizers.append(self.qc_quantize_op_dict[input_name])

        # Capture output quantizers of the op
        for cg_product in op.outputs:
            if cg_product.name in self.qc_quantize_op_dict:
                output_quantizers.append(self.qc_quantize_op_dict[cg_product.name])

        # Capture param quantizers of the op
        for param_name, (_, param_type) in op.parameters.items():
            if param_name in self.qc_quantize_op_dict:
                param_quantizers[param_type] = self.qc_quantize_op_dict[param_name]

        return input_quantizers, output_quantizers, param_quantizers

    def _apply_param_symmetry_to_inputs(
        self, quantsim_configurator: QuantSimConfigurator
    ):
        """
        Apply Param symmetry to it's respective input quantizer when weights are not constant.

        Currently this is applicable to the following operations:
            Conv, ConvTranspose, Gemm, MatMul
        """

        # Get default symmetry from config
        default_symmetry = (
            quantsim_configurator.quantsim_configs.get("defaults", {})
            .get("params", {})
            .get("is_symmetric", False)
        )
        op_specific_config = quantsim_configurator.quantsim_configs.get("op_type", {})

        for op in self.connected_graph.ordered_ops:
            if op.type not in ("Conv", "ConvTranspose", "Gemm", "MatMul"):
                continue

            op_weights = op.inputs[WEIGHT_INDEX]
            # Check if weights are constant
            if op_weights.name in self.param_names:
                continue

            # If `op_type` overrides symmetry, use that. Otherwise, use default symmetry from config
            expected_op_symmetry = (
                op_specific_config.get(op.type, {})
                .get("params", {})
                .get("weight", {})
                .get("is_symmetric", default_symmetry)
            )
            input_weight_quantizer = self._get_enabled_quantizer(op_weights.name)

            if input_weight_quantizer is None:
                logger.warning(
                    "Quantizer for weights input not found for Op: %s. Unable to override symmetry for input weights.",
                    op.name,
                )
                continue

            # Override symmetry for input weights
            input_weight_quantizer.use_symmetric_encodings = expected_op_symmetry

    def _apply_exception_rules(self):
        """
        Apply exception rules to specific op. For example, a rule can override high bitwidth to GroupNorm op.
        """
        # pylint:disable = too-many-branches
        for op in self.connected_graph.get_all_ops().values():
            _, output_quantizers, param_quantizers = self.get_op_quantizers(op)

            if op.type == "GroupNormalization":
                if self._hw_version is None or self._hw_version in {
                    "V66",
                    "V68",
                    "V69",
                }:
                    continue
                if "weight" in param_quantizers:
                    output_quantizer = output_quantizers[0]
                    for _, param_quantizer in param_quantizers.items():
                        param_quantizer.bitwidth = output_quantizer.bitwidth
                        param_quantizer.use_symmetric_encodings = (
                            output_quantizer.use_symmetric_encodings
                        )

            elif op.type == "MatMul":
                # Apply exception rule only to dynamic matmuls
                if op.inputs[1].name in self.param_names:
                    continue
                target_quantizer_for_first_input = self._get_enabled_quantizer(
                    op.inputs[0].name
                )
                target_quantizer_for_second_input = self._get_enabled_quantizer(
                    op.inputs[1].name
                )

                # According to opdef for Matmul in HTP:
                # 16bit Weight(second input for dynamic MatMul) must have 16bit Activation(first input for dynamic MatMul).
                # 16bit Activation and 16bit Weight require minimum arch V73.
                # 16bit Weight must be symmetric quantized.

                # Below are the possible combinations for MatMul with 8/16 bitwidth:
                # If version is V73/V75: {input0->8, input1->8 symm/asymm} {input0->16 , input1->8 symm/asymm} {input0->16, input1->16 symmetric}
                # If version is lesser than V73: {input0->8, input1->8 symmetric} {input0->16, input1->8 symmetric}
                if self._hw_version is None:
                    continue
                if self._hw_version in {"V66", "V68", "V69"}:
                    if target_quantizer_for_second_input is None:
                        logger.warning(
                            "The target quantizer for second input could not be found. MatMul exception rule does not apply for op: %s.",
                            op.name,
                        )
                    elif (
                        target_quantizer_for_second_input.data_type
                        == QuantizationDataType.int
                    ):
                        target_quantizer_for_second_input.use_symmetric_encodings = True
                        target_quantizer_for_second_input.set_bitwidth(8)
                else:
                    if (
                        target_quantizer_for_first_input is None
                        or target_quantizer_for_second_input is None
                    ):
                        logger.warning(
                            "The target quantizers could not be found. MatMul exception rule does not apply for op: %s.",
                            op.name,
                        )
                    elif target_quantizer_for_second_input.bitwidth == 16:
                        target_quantizer_for_second_input.use_symmetric_encodings = True
                        target_quantizer_for_first_input.set_bitwidth(16)

            else:
                bias_idx = _get_matmul_add_bias_idx(op, self.model.model)

                if bias_idx is not None:
                    bias = op.inputs[bias_idx]
                    matmul_output = op.inputs[1 - bias_idx]
                    matmul = matmul_output.producer
                    weight = next(
                        param
                        for param, param_type in matmul.parameters.values()
                        if param_type == "weight"
                    )

                    matmul_output_qtzr = self.qc_quantize_op_dict[matmul_output.name]
                    bias_qtzr = self.qc_quantize_op_dict[bias.name]
                    weight_qtzr = self.qc_quantize_op_dict[weight.name]

                    # Disable intermediate output quantization and bias quantization
                    matmul_output_qtzr.enabled = False
                    bias_qtzr.enabled = False

                    # Let bias quantizers follow the same granularity as weight quantizer
                    bias_qtzr.enable_per_channel_quantization(
                        weight_qtzr.quant_info.usePerChannelMode
                    )

    @deprecated("Use _get_enabled_quantizer instead")
    def _get_closest_enabled_quantizer(self, tensor: Product):
        """
        Deprecated. Use :meth:`_get_enabled_quantizer` to get the quantizer instead.

        Returns closest enabled quantizer to `tensor` traversing upwards

        :param tensor: Tensor for which to find quantizer
        """
        quantizer = self.qc_quantize_op_dict.get(tensor.name, None)
        if quantizer and quantizer.enabled:
            return quantizer
        if not tensor.producer:
            return None
        if not tensor.producer.inputs:
            return None
        # Assume first input to parent op is the relevant upstream activation
        upstream_tensor = tensor.producer.inputs[0]
        return self._get_closest_enabled_quantizer(upstream_tensor)

    def save_model_graph(self, filename_prefix: str):
        """
        Save model to given path

        :param filename_prefix: filename to save the onnx model
        """
        self.model.save_model_to_file(
            os.path.join(self._path, filename_prefix) + ".onnx"
        )

    @overload
    def compute_encodings(self, inputs: Iterable[Dict[str, np.ndarray]]):  # pylint: disable=arguments-differ
        ...

    @overload
    def compute_encodings(
        self, forward_pass_callback: Callable[[ort.InferenceSession], Any]
    ):  # pylint: disable=arguments-differ
        ...

    T = TypeVar("T")

    @overload
    def compute_encodings(
        self,  # pylint: disable=arguments-differ
        forward_pass_callback: Callable[[ort.InferenceSession, T], Any],
        forward_pass_callback_args: T,
    ): ...

    del T

    def compute_encodings(self, *args, **kwargs):
        r"""
        Computes encodings for all quantizers in the model.

        This API will invoke `forward_pass_callback`, a function written by the user that runs
        forward pass(es) of the quantized model with a small, representative subset of the training dataset.
        By doing so, the quantizers in the quantized model will observe the inputs and initialize
        their quantization encodings according to the observed input statistics.

        This function is overloaded with the following signatures:

        .. function:: compute_encodings(inputs)
           :noindex:

           :param inputs: The set of model input samples to use during calibration
           :type inputs: Iterable[Dict[str, np.ndarray]]

        .. function:: compute_encodings(forward_pass_callback)
           :noindex:

           :param forward_pass_callback_: A function that takes a quantized model and runs forward passes
               with a small, representative subset of training dataset
           :type forward_pass_callback_: Callable[[ort.InferenceSession], Any]

        .. function:: compute_encodings(forward_pass_callback, forward_pass_callback_args)
           :noindex:

           :param forward_pass_callback_: A function that takes a quantized model and runs forward passes
               with a small, representative subset of training dataset
           :type forward_pass_callback_: Callable[[ort.InferenceSession, T], Any]
           :param T forward_pass_callback_args: The second argument to `forward_pass_callback`.

        Example:

            >>> sim = QuantizationSimModel(...)
            >>> def run_forward_pass(session: ort.InferenceSession):
            ...     for input in dataset:
            ...         _ = sess.run(None, {"input": input})
            ...
            >>> sim.compute_encodings(run_forward_pass)
        """
        inputs, forward_pass_callback, forward_pass_calback_args = (
            _parse_compute_encodings_args(*args, **kwargs)
        )
        if forward_pass_callback:
            return self._compute_encodings_from_callback(
                forward_pass_callback, forward_pass_calback_args
            )

        with compute_encodings(self):
            for item in inputs:
                self.session.run(None, item)

    def _compute_encodings_from_callback(
        self, forward_pass_callback, forward_pass_callback_args=_NOT_SPECIFIED
    ):
        if forward_pass_callback_args is _NOT_SPECIFIED:
            args = (self.session,)
        else:
            warnings.warn(
                _red(
                    "Support for calling compute_encodings() with forward_pass_callback_args is deprecated and will be removed in the future. "
                ),
                DeprecationWarning,
                stacklevel=3,
            )
            args = (self.session, forward_pass_callback_args)

        with compute_encodings(self):
            forward_pass_callback(*args)

    def _compute_param_encodings(
        self,
        *,
        dummy_input: Optional[Dict[str, np.ndarray]] = None,
        overwrite: bool = True,
    ):
        """
        Computes param encodings for the sim.

        Args:
            dummy_input: Input to pass during calibration. If None, input is randomly generated
            overwrite: If true, overwrites all existing param encodings. Otherwise, only computes non-initialized param encodings
        """
        if dummy_input is None:
            dummy_input = make_dummy_input(self.model.model)

        quantizers_to_calibrate = {
            name for name in self.param_names if self.qc_quantize_op_dict[name].enabled
        }

        # If not overwrite, exclude already-initialized quantizers
        if not overwrite:
            quantizers_to_calibrate -= {
                name
                for name in self.param_names
                if self.qc_quantize_op_dict[name].is_initialized()
            }

        # Early exit if there's nothing to calibrate
        if not quantizers_to_calibrate:
            return

        quantizers_to_disable = (
            self.qc_quantize_op_dict.keys() - quantizers_to_calibrate
        )
        with utils.disable_quantizers(self, quantizers_to_disable):
            self.compute_encodings([dummy_input])

    def _get_encodings(self, quantizer_names, enc_version):
        encoding_dict = {}
        for name in quantizer_names:
            encoding = self.qc_quantize_op_dict[name].export_encodings(enc_version)
            if encoding is None:
                continue
            encoding_dict[name] = encoding

        if version.parse(enc_version) < version.parse("1.0.0"):
            return encoding_dict

        for name, encoding in encoding_dict.items():
            encoding["name"] = name
        return list(encoding_dict.values())

    def _export_encodings(self, encoding_file_path):
        """
        Export encodings to json file

        :param encoding_file_path: path to save the encoding file
        """
        enc_version = quantsim.encoding_version
        if enc_version not in VALID_ENCODING_VERSIONS:
            raise NotImplementedError(
                f"Encoding version {enc_version} not in set of valid encoding "
                f"versions {VALID_ENCODING_VERSIONS}."
            )

        encodings_dict = {"version": enc_version}

        if quantsim.encoding_version >= "2.0.0":
            encodings = self._get_encodings(
                self.qc_quantize_op_dict.keys(), enc_version
            )

            encodings_dict.update(
                {
                    "encodings": encodings,
                }
            )
        else:
            param_encodings = self._get_encodings(self.param_names, enc_version)
            activation_encodings = self._get_encodings(
                self.activation_names, enc_version
            )

            encodings_dict.update(
                {
                    "activation_encodings": activation_encodings,
                    "param_encodings": param_encodings,
                    "quantizer_args": self.quant_args,
                }
            )

        save_json_yaml(encoding_file_path, encodings_dict)

    @contextlib.contextmanager
    def _remove_quantization_nodes(self):
        """
        Remove quantization nodes
        """
        sim_outputs = [out.name for out in self.model.graph().output]
        sim_nodes = list(self.model.nodes())
        try:
            self.model = self.remove_quantizers(self.model)

            yield

        finally:
            self.model.model.graph.ClearField("node")
            self.model.model.graph.node.extend(sim_nodes)
            for output, name in zip(self.model.graph().output, sim_outputs):
                output.name = name

    @staticmethod
    def remove_quantizers(model: Union[ONNXModel, ModelProto]):
        """
        Removes all QcQuantizeOp layers from model
        """
        if isinstance(model, ONNXModel):
            QuantizationSimModel.remove_quantizers(model.model)
            return model

        original_nodes = [
            node for node in model.graph.node if node.op_type != "QcQuantizeOp"
        ]
        tensor_name_map = {
            node.output[0]: node.input[0]
            for node in model.graph.node
            if node.op_type == "QcQuantizeOp"
        }

        model.graph.ClearField("node")
        model.graph.node.extend(original_nodes)

        for node in model.graph.node:
            for i, tensor in enumerate(node.input):
                if tensor not in tensor_name_map:
                    continue
                node.input[i] = tensor_name_map[tensor]

            for i, tensor in enumerate(node.output):
                if tensor not in tensor_name_map:
                    continue
                node.output[i] = tensor_name_map[tensor]

        for i, tensor in enumerate(model.graph.output):
            if tensor.name in tensor_name_map:
                model.graph.output[i].name = tensor_name_map[tensor.name]

        return model

    def _adjust_weight_scales_for_int32_bias(self):
        """
        Given, bias_scale = weight_scale * input_scale and If max(input) * max(weight) << bias,
        bias_scale becomes very small and dividing a large bias_float by a very small bias_scale
        results in very large bias_int32, potentially exceeding the int32 range (-2147483648 to 2147483647)
        during W16A16 quantization.

        Adjusting weight_scale and bias_scale when the bias_float value exceeds the int32 range,
        reduce the risk of saturation in activations (input @ weight + bias).

        NOTE: Increasing the input_scale can reduce precision for all activations, while increasing
              weight_scale only reduce the precision for the affected weights.
        """
        # pylint: disable=redefined-builtin, protected-access

        ops_with_analytic_bias_scale = {
            "Conv",
            "Gemm",
            "MatMul",
            "ConvTranspose",
        }

        ops_with_bias = {
            op: self._get_weight_and_bias(op)
            for op in self.connected_graph.get_all_ops().values()
            if op.type in ops_with_analytic_bias_scale
        }

        for op, (weight, bias) in ops_with_bias.items():
            if bias is None:
                continue

            input, *_ = op.inputs
            input_qtzr = self._get_enabled_quantizer(input.name)

            if not (input_qtzr and input_qtzr.enabled and input_qtzr.is_initialized()):
                continue

            weight_qtzr = self.qc_quantize_op_dict.get(weight.name, None)
            if not (
                weight
                and weight_qtzr
                and weight_qtzr.enabled
                and weight_qtzr.data_type == QuantizationDataType.int
                and weight_qtzr.is_initialized()
            ):
                # Weight quantizer wasn't created, enabled, or initialized.
                # Since weight_scale isn't available, exclude bias from quantization.
                continue

            if weight_qtzr.quant_info.blockSize > 0:
                # Handle weight adjustment for BQ and LPBQ quantizers
                continue

            bias_proto = self.model.get_initializer(bias.name)
            if not bias_proto:
                try:
                    bias_proto = next(
                        attr.t
                        for node in self.model.graph().node
                        if bias.name in node.output
                        for attr in node.attribute
                        if attr.type == onnx.AttributeProto.TENSOR
                    )
                except StopIteration:
                    logger.info(
                        "Bias tensor %s not found for op: %s", bias.name, op.name
                    )
                    continue

            bias_float = onnx.numpy_helper.to_array(bias_proto)

            weight_scale = weight_qtzr._get_scale()
            input_scale = input_qtzr._get_scale()

            if weight_scale is None or input_scale is None:
                continue

            bias_scale = self._get_analytic_bias_scale(op)

            if not np.any(_is_bias_out_of_int32_range(bias_float, bias_scale)):
                continue

            encodings = weight_qtzr.get_encodings()
            if encodings is None:
                continue

            adjusted_weight_scale = _get_adjusted_weight_scale(
                bias_float, input_scale, weight_scale
            )
            assert len(adjusted_weight_scale) == len(encodings), (
                "Weight scale adjustment only supported for per-tensor and per-channel scales."
            )
            for new_scale, enc in zip(adjusted_weight_scale, encodings):
                enc.delta = new_scale
            weight_qtzr.load_encodings(encodings)
            logger.info(
                "Adjusted weight scale for %s to prevent bias overflow.", op.name
            )

    def _get_statistical_bias_scale(self, op: Op) -> np.ndarray:
        r"""
        Compute int32 bias scale statistically, such that

        :math:`scale = abs(max(bias)) / 2**31`

        Note that using statistical bias scale isn't ideal for runtime performance
        on integer accelerators.
        For better runtime performance, bias encodings should be derived analytically
        whenever possible. (See ``get_analytic_bias_scale``)
        """
        _, bias = self._get_weight_and_bias(op)
        bias_proto = utils.ParamUtils.get_param_by_name(self.model.model, bias.name)

        if bias_proto is None:
            raise RuntimeError(
                "Failed to calibrate encoding of bias. "
                f'Couldn\'t find the value of "{bias.name}" statically from the graph.'
            )

        bias_float = to_array(bias_proto)

        bias_scale = np.maximum(abs(bias_float) / 2**31, _INT32_MINIMUM_SCALE)

        bias_qtzr = self.qc_quantize_op_dict[bias.name]
        if not bias_qtzr.quant_info.usePerChannelMode:
            bias_scale = bias_scale.max()

        return bias_scale

    def _get_analytic_bias_scale(self, op: Op) -> np.ndarray:
        """
        Derive int32 bias scale analytically from input and weight encodings, such that

        :math:`bias_scale = weight_scale * input_scale`

        This analytic formula is friendly for integer hardware/runtime
        since bias-add operation ``(input @ weight) + bias`` becomes trivial when
        both terms share the same quantization scale
        """
        # pylint: disable=redefined-builtin, protected-access, too-many-statements
        input, *_ = op.inputs
        weight, bias = self._get_weight_and_bias(op)
        assert bias is not None

        weight_qtzr = self.qc_quantize_op_dict.get(weight.name)
        input_qtzr = self._get_enabled_quantizer(input.name)

        if not (input_qtzr and input_qtzr.enabled and input_qtzr.is_initialized()):
            return self._get_statistical_bias_scale(op)

        channel_axis = None
        num_channels = None
        block_axis = None
        block_size = None
        if weight_qtzr.quant_info.usePerChannelMode:
            channel_axis = weight_qtzr.quant_info.channelAxis
            num_channels = weight_qtzr.tensor_quantizer_params.tensor_shape[
                channel_axis
            ]
            block_size = weight_qtzr.quant_info.blockSize or None
            block_axis = weight_qtzr.quant_info.blockAxis if block_size else None

            expected_channel_axis, expected_block_axis = self._get_quantization_axes(op)
            ndim = len(weight_qtzr.tensor_quantizer_params.tensor_shape)

            if channel_axis < 0:
                channel_axis = ndim + channel_axis
            if block_axis is not None and block_axis < 0:
                block_axis = ndim + block_axis
            if expected_channel_axis is not None and expected_channel_axis < 0:
                expected_channel_axis = ndim + expected_channel_axis
            if expected_block_axis is not None and expected_block_axis < 0:
                expected_block_axis = ndim + expected_block_axis

            if channel_axis != expected_channel_axis or block_axis not in (
                expected_block_axis,
                None,
            ):
                # For example:
                #   * Conv with channel_axis=1
                #   * ConvTranspose with channel_axis=0
                #   * Gemm with channel_axis=1
                return self._get_statistical_bias_scale(op)

        if isinstance(weight_qtzr, GroupedBlockQuantizeDequantize):
            # NOTE: In LPBQ, bias encodings should be derived from per-channel weight scale
            weight_scale = weight_qtzr._get_per_channel_scale()
        else:
            weight_scale = weight_qtzr._get_scale()

        input_scale = input_qtzr._get_scale()

        if weight_scale is None or input_scale is None:
            return self._get_statistical_bias_scale(op)

        bias_scale = input_scale * weight_scale

        if block_size is not None:
            bias_scale = bias_scale.max(axis=block_axis)

        if channel_axis is not None:
            bias_scale = bias_scale.reshape([num_channels])

        return bias_scale

    def _concretize_int32_bias_quantizers(self):
        # pylint: disable=protected-access
        switcher = {
            "Conv": self._get_analytic_bias_scale,
            "Gemm": self._get_analytic_bias_scale,
            "MatMul": self._get_analytic_bias_scale,
            "ConvTranspose": self._get_analytic_bias_scale,
            "BatchNormalization": self._get_statistical_bias_scale,
            "InstanceNormalization": self._get_statistical_bias_scale,
            "LayerNormalization": self._get_statistical_bias_scale,
            "GroupNormalization": self._get_statistical_bias_scale,
        }

        ops_with_bias = {
            op: self._get_weight_and_bias(op)
            for op in self.connected_graph.get_all_ops().values()
            if op.type in switcher
        }

        for op, (weight, bias) in ops_with_bias.items():
            if bias is None:
                continue

            bias_qtzr = self.qc_quantize_op_dict.get(bias.name)

            weight_qtzr = self.qc_quantize_op_dict.get(weight.name)
            encoding_type = weight_qtzr._encoding_type().name

            if bias_qtzr.data_type == QuantizationDataType.float:
                # Float16 quantizers are not exported to onnx QDQ graph
                continue

            if bias_qtzr and bias_qtzr.enabled and bias_qtzr.is_initialized():
                # Edge case: bias encoding already exists.
                # Always honor the existing bias encoding
                continue

            if not (
                weight_qtzr
                and weight_qtzr.enabled
                and weight_qtzr.data_type == QuantizationDataType.int
                and weight_qtzr.is_initialized()
            ):
                # Weight quantizer wasn't created, enabled, or initialized.
                # Since weight_scale isn't available, exclude bias from quantization.
                continue

            if encoding_type == EncodingType.PER_TENSOR.name:
                bias_qtzr.enable_per_channel_quantization(False)
            elif encoding_type in [
                EncodingType.PER_CHANNEL.name,
                EncodingType.LPBQ.name,
                EncodingType.PER_BLOCK.name,
            ]:
                bias_qtzr.enable_per_channel_quantization()
            else:
                raise RuntimeError(
                    f"Unknown encoding type {encoding_type}, cannot concretize bias quantizers."
                )

            if weight is None:
                # Edge case: Op has no weight. Fall back to statistical bias scale
                get_bias_scale = self._get_statistical_bias_scale
            else:
                get_bias_scale = switcher.get(op.type, self._get_statistical_bias_scale)

            bias_scale = get_bias_scale(op)

            encodings = [libpymo.TfEncoding() for _ in range(bias_scale.size)]

            for enc, scale in zip(encodings, bias_scale.flatten()):
                enc.bw = 32
                enc.delta = scale
                enc.offset = -(2**31)
                enc.min = scale * -(2**31)
                enc.max = scale * (2**31 - 1)

            bias_qtzr.load_encodings(encodings)
            bias_qtzr.enabled = True

    def _get_weight_and_bias(
        self, op: Op
    ) -> Tuple[Optional[Product], Optional[Product]]:
        weight = None
        bias = None

        for inp in op.inputs:
            _, param_type = op.parameters.get(inp.name, (None, None))
            if param_type == "weight":
                weight = inp
            elif param_type == "bias":
                bias = inp

        if op.type == "MatMul":
            # Fetch weight from the previous MatMul node
            bias_idx = _get_matmul_add_bias_idx(op, self.model.model)
            if bias_idx is not None:
                (add,) = op.outputs[0].consumers
                _, bias = self._get_weight_and_bias(add)

        return weight, bias

    def export(self, path: str, filename_prefix: str, export_model: bool = True):
        """
        Compute encodings and export to files

        :param path: dir to save encoding files
        :param filename_prefix: filename to save encoding files
        :param export_model: If True, then ONNX model is exported. When False, only encodings are exported.
        """
        if quantsim.encoding_version == "0.6.1":
            msg = _red(
                "Encoding version 0.6.1 was deprecated in favor of 1.0.0 since aimet-onnx==2.1. "
                "If your code depends on parsing the exported encodings file, ensure that it is "
                "updated to be able to parse 1.0.0 format"
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
        self._export_encodings(os.path.join(path, filename_prefix) + ".encodings")

        if export_model:
            with self._remove_quantization_nodes():
                if self.model.model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF:
                    # Note: Saving as external data mutates the saved model, removing all initializer data
                    save_model_with_external_weights(
                        self.model.model,
                        os.path.join(path, filename_prefix) + ".onnx",
                        location=filename_prefix + ".data",
                        all_tensors_to_one_file=True,
                    )
                else:
                    self.model.save_model_to_file(
                        os.path.join(path, filename_prefix) + ".onnx"
                    )

    def set_and_freeze_param_encodings(self, encoding_path: str):
        """
        Set and freeze parameter encodings from encodings JSON file

        :param encoding_path: path from where to load parameter encodings file
        """

        # Load encodings file
        with open(encoding_path) as json_file:
            encodings = json.load(json_file)

        # TODO: handle this more cleanly
        if isinstance(encodings, dict):
            encodings = _convert_encoding_format_0_6_1_to_1_0_0(encodings)

        encodings_dict = {encoding["name"]: encoding for encoding in encodings}
        for quantizer_name in encodings_dict:
            if quantizer_name in self.qc_quantize_op_dict:
                # pylint: disable=protected-access
                self.qc_quantize_op_dict[quantizer_name]._load_encodings_dict(
                    encodings_dict[quantizer_name]
                )
                self.qc_quantize_op_dict[quantizer_name].freeze_encodings()

    def get_all_quantizers(self) -> Tuple[List, List]:
        """
        Returns all QcQuantizeOps through which TensorQuantizer's attributes can be accessed.
        """
        param_quantizers = []
        activation_quantizers = []

        for param in self.param_names:
            param_quantizers.append(self.qc_quantize_op_dict[param])

        for activation in self.activation_names:
            activation_quantizers.append(self.qc_quantize_op_dict[activation])

        return param_quantizers, activation_quantizers

    def _rebuild_session(self):
        """
        Rebuilds `self.session` object to reflect any changes in the source model.
        """
        self.session = build_session(
            self.model.model,
            self.providers,
            user_onnx_libs=self._user_onnx_libs,
            path=self._path,
        )

    def set_quantizers(self, quantizer_dict: Dict[str, QcQuantizeOp]):
        """
        Updates `self.qc_quantize_op_dict` with the entries in `quantizer_dict`

        :param quantizer_dict: Dictionary mapping tensor names to QcQuantizeOp objects
        """
        for tensor, quantizer in quantizer_dict.items():
            self._set_quantizer(tensor, quantizer)

        self._rebuild_session()

    def _set_quantizer(self, tensor_name: str, quantizer: QcQuantizeOp):
        """
        Places `quantizer` at `tensor_name` and updates the onnx graph.

        :param tensor_name: Name of the tensor at which to place the source quantizer
        :param quantizer: Quantizer to place at tensor_name
        """
        if not isinstance(quantizer, QcQuantizeOp):
            raise TypeError(
                f"Quantizer object {quantizer} is not of type {QcQuantizeOp.__qualname__}"
            )
        if tensor_name not in self.qc_quantize_op_dict:
            raise ValueError(f"Tensor {tensor_name} is not an input to a quantize node")

        self._set_quant_info(tensor_name, quantizer)
        self.qc_quantize_op_dict[tensor_name] = quantizer

    def _set_quant_info(self, dst_qtzr_tensor_name: str, src_qtzr: QcQuantizeOp):
        """
        Set quant_info attribute (pointer to the libquant_info object)

        :param dst_qtzr_tensor_name: destination quantizer node name in graph.
        :param src_qtzr: source quantizer.
        """
        for node in self.model.graph().node:
            if node.op_type == "QcQuantizeOp" and node.input[0] == dst_qtzr_tensor_name:
                for atr in node.attribute:
                    if atr.name == "quant_info":
                        atr.i = libpymo.PtrToInt64(src_qtzr.quant_info)
                        # Session is now invalid and must be rebuilt
                        self.session = None
                        return

        raise RuntimeError(f"Did not find quantizer for tensor {dst_qtzr_tensor_name}")

    def _tie_quantizers_for_op_types(self, op_types_to_tie: List[str]):
        """
        Tie the input and output quantizers for given op types.

        :param op_types_to_tie: List of onnx ops for which to tie quantizers
        """
        self._propagate_input_encodings({x for x in op_types_to_tie if x != "Concat"})

        if "Concat" in op_types_to_tie:
            self._propagate_output_encodings({"Concat"})

    def _propagate_output_encodings(self, op_types_to_tie: Set[str]):
        """
        Let input quantizers inherit output encodings

        :param op_types_to_tie: List of onnx ops for which to tie quantizers
        """
        qtzr_to_name = {qtzr: name for name, qtzr in self.qc_quantize_op_dict.items()}

        for op in reversed(self.connected_graph.ordered_ops):
            if op.type not in op_types_to_tie:
                continue

            if not op.outputs:
                continue

            output_name = op.outputs[0].name
            output_qtzr = self.qc_quantize_op_dict.get(output_name)

            if not output_qtzr:
                continue

            if len(op.outputs) != 1:
                msg = (
                    "Encoding propagation is only supported for ops with exactly "
                    f"1 output, but {op.name} (type: {op.type}) has {len(op.outputs)} "
                    "outputs"
                )
                raise RuntimeError(msg)

            src_qtzrs = {}

            for inp in op.inputs:
                src_qtzr = self._get_enabled_quantizer(inp.name)
                if src_qtzr:
                    src_name = qtzr_to_name[src_qtzr]
                else:
                    src_name = inp.name

                src_qtzrs[src_name] = src_qtzr

            # If all inputs are quantized and have the same fixed quantization range,
            # output quantizer will inherit the fixed range.
            # In practice, this will be most relevant when
            # Concat takes all its inputs from Softmax/Sigmoid.
            #            [0, 1]          [0, 1]
            #   Softmax -------> Concat ------>
            #   Softmax -----------^
            #            [0, 1]
            if all(qtzr is not None for qtzr in src_qtzrs.values()):
                src_min_max_ranges = set(
                    src_qtzr._encoding_min_max_fixed_vals  # pylint: disable=protected-access
                    for src_qtzr in src_qtzrs.values()
                )
                if len(src_min_max_ranges) == 1:
                    output_qtzr.set_fixed_encoding_range(src_min_max_ranges.pop())

            for src_name, src_qtzr in src_qtzrs.items():
                if src_qtzr:
                    self._set_quantizer(src_name, output_qtzr)

    def _propagate_input_encodings(self, op_types_to_tie: Set[str]):
        """
        Let output quantizers inherit input encodings

        :param op_types_to_tie: List of onnx ops for which to tie quantizers
        """
        for op in self.connected_graph.ordered_ops:
            if op.type not in op_types_to_tie:
                continue

            if not op.inputs:
                msg = (
                    "Encoding propagation is only supported for ops with at least "
                    f"1 input, but {op.name} (type: {op.type}) has no input"
                )
                raise RuntimeError(msg)

            input_qtzr = self._get_enabled_quantizer(op.inputs[0].name)

            if not input_qtzr:
                continue

            for out in op.outputs:
                output_qtzr = self.qc_quantize_op_dict.get(out.name)

                if output_qtzr and output_qtzr.enabled:
                    # If output quantizer already exists,
                    # "merge" the quantization constraints into single quantizer
                    #
                    # This logic was added specifically to resolve conflicting
                    # constraints between Softmax output and the second input of MatMul.
                    #   Softmax ------------> ... ------------> MatMul
                    #             [0, 1]            symmetric
                    #
                    # Ideally, this conflict should be resolved by
                    # simulating & exporting two independent quantizers like this:
                    #   Softmax ---> QDQ ------> QDQ ---> MatMul
                    #               [0, 1]     symmetric
                    #
                    # However, this is currently not allowed by QAIRT.
                    # As an ad-hoc workaround, we "merge" the two configurations as below:
                    #   Softmax ---------> QDQ ---------> MatMul
                    #                    [-1, 1]
                    #                    symmetric
                    input_qtzr._merge_constraints(output_qtzr)  # pylint: disable=protected-access

                self._set_quantizer(out.name, input_qtzr)

    def to_onnx_qdq(self) -> onnx.ModelProto:
        """
        Return a copy of ModelProto with all QcQuantizeOp nodes replaced with
        QuantizeLinear and/or DequantizeLinear.

        Example:

            >>> len([qc_op for qc_op in sim.model.nodes() if dq.op_type == "QcQuantizeOp"])
            10
            >>> onnx_qdq = sim.to_onnx_qdq()
            >>> len([qc_op for qc_op in sim.model.nodes() if dq.op_type == "QcQuantizeOp"])
            0
            >>> len([dq for dq in onnx_qdq.graph.node if dq.op_type == "DequantizeLinear"])
            10
        """
        return self._to_onnx_qdq(prequantize_constants=False)

    def _to_onnx_qdq(self, prequantize_constants: bool) -> onnx.ModelProto:
        try:
            invalid_bitwidth = next(
                qtzr.bitwidth
                for qtzr in self.qc_quantize_op_dict.values()
                if qtzr.data_type == QuantizationDataType.int
                and qtzr.bitwidth not in (4, 8, 16, 32)
            )
        except StopIteration:
            invalid_bitwidth = None

        if invalid_bitwidth is not None:
            raise RuntimeError(
                f"Invalid bitwidth {invalid_bitwidth};"
                " expected standard ONNX integer data types such as [U]INT{4, 8, 16, 32}"
            )

        onnx_opset_version = next(
            opset.version for opset in self.model.opset_import() if opset.domain == ""
        )

        desired_onnx_opset_version = onnx_opset_version

        if onnx_opset_version < 10:
            desired_onnx_opset_version = 10

            logger.info(
                "onnx::QuantizeLinear and DequantizeLinear are only supported in opset >= 10;"
                " got opset=%d",
                onnx_opset_version,
            )

        if onnx_opset_version < 13 and any(
            qtzr.quant_info.usePerChannelMode
            and qtzr.tensor_quantizer_params
            and qtzr.tensor_quantizer_params.channel_axis is not None
            for qtzr in self.qc_quantize_op_dict.values()
        ):
            desired_onnx_opset_version = 13
            logger.info(
                "onnx::QuantizeLinear and DequantizeLinear with per-channel are only supported in opset >= 13;"
                " got opset=%d",
                onnx_opset_version,
            )

        if onnx_opset_version < 21 and any(
            qtzr.quant_info.usePerChannelMode
            and qtzr.tensor_quantizer_params
            and qtzr.quant_info.blockSize > 0
            for qtzr in self.qc_quantize_op_dict.values()
        ):
            desired_onnx_opset_version = 21
            logger.info(
                "onnx::QuantizeLinear and DequantizeLinear with per-block are only supported in opset >= 21;"
                " got opset=%d",
                onnx_opset_version,
            )

        if onnx_opset_version < 21 and any(
            qtzr.data_type == QuantizationDataType.int and qtzr.bitwidth not in (8, 32)
            for qtzr in self.qc_quantize_op_dict.values()
        ):
            desired_onnx_opset_version = 21
            logger.info(
                "onnx::QuantizeLinear and DequantizeLinear with INT4/INT16 are only supported in opset >= 21;"
                " got opset=%d",
                onnx_opset_version,
            )

        model_copy = onnx.ModelProto()
        model_copy.CopyFrom(self.model.model)

        self._overwrite_parameters(model_copy, self._get_qdq_parameters())

        aimet_qc_quantize_nodes = [
            node
            for node in model_copy.graph.node
            if node.op_type == "QcQuantizeOp"
            and node.domain in ("aimet.customop.cpu", "aimet.customop.cuda")
        ]

        qdq_node_info = {
            "input_names": [],
            "output_names": [],
            "node_name_prefixes": [],
            "encodings": [],
        }
        param_names = set(self.param_names)

        for aimet_node in aimet_qc_quantize_nodes:
            input_name = aimet_node.input[0]
            qtzr = self.qc_quantize_op_dict[input_name]
            encodings = qtzr.export_encodings("2.0.0")

            if encodings:
                if input_name not in param_names:
                    # Always cast activation encoding to unsigned encoding.
                    # This takes care of edge case where qtzr could be a symmetric quantizer
                    # for dynamic weight of Conv/ConvTranspose/Gemm/Matmul.
                    # This is a workaround for QNN converter limitation
                    encodings = _to_unsigned_encoding(encodings)

                # Affine quantizer
                # Replace QcQuantizeOp with onnx::QuantizeLinear and DequantizeLinear
                qdq_node_info["input_names"].append(aimet_node.input[0])
                qdq_node_info["output_names"].append(aimet_node.output[0])
                qdq_node_info["node_name_prefixes"].append(aimet_node.name)
                qdq_node_info["encodings"].append(encodings)

        self.remove_quantizers(model_copy)

        if onnx_opset_version < desired_onnx_opset_version:
            model_copy = _convert_version(model_copy, desired_onnx_opset_version)

        _add_onnx_qdq_nodes(
            model_copy,
            **qdq_node_info,
            onnx_opset=desired_onnx_opset_version,
            prequantize_constants=prequantize_constants,
        )

        # Restore original model's output names
        #
        #   ORIGINAL MODEL:
        #     ... -> last_node -------------->
        #                       (out)
        #
        #   ONNX QDQ (BEFORE RENAMING):
        #     ... -> last_node --------------> Q ---------> DQ -------------->
        #                       (out)             (out_q)      (out_updated)
        #
        #   ONNX QDQ (AFTER RENAMING):
        #     ... -> last_node --------------> Q ---------> DQ -------------->
        #                       (out_updated)     (out_q)      (out)
        consumers = {
            node.input[i]: node
            for node in model_copy.graph.node
            for i in range(len(node.input))
        }
        producers = {
            node.output[i]: node
            for node in model_copy.graph.node
            for i in range(len(node.output))
        }
        for graph_out in model_copy.graph.output:
            last_node = producers[graph_out.name]
            q = consumers.get(graph_out.name)

            if not (q and q.op_type == "QuantizeLinear"):
                continue

            dq = consumers.get(q.output[0])

            if not (dq and dq.op_type == "DequantizeLinear"):
                continue

            i = list(last_node.output).index(graph_out.name)
            last_node.output[i], dq.output[0] = dq.output[0], last_node.output[i]

            # Redirect "out" and "out_updated" to the right consumer
            for node in model_copy.graph.node:
                for j, inp in enumerate(node.input):
                    if inp == last_node.output[i]:
                        node.input[j] = dq.output[0]
                    elif inp == dq.output[0]:
                        node.input[j] = last_node.output[i]

        ONNXModel(model_copy).topological_sort()
        return model_copy

    def _get_qdq_parameters(self):
        param_names = {
            product.name
            for op in self.connected_graph.get_all_ops().values()
            for product, _ in op.parameters.values()
            if self.qc_quantize_op_dict[product.name].bitwidth <= 32
        }
        qdq_params = {
            f"{product.name}_qdq": product
            for op in self.connected_graph.get_all_ops().values()
            for product, _ in op.parameters.values()
            if self.qc_quantize_op_dict[product.name].bitwidth <= 32
        }

        partial_model = onnx.helper.make_model(
            ir_version=self.model.model.ir_version,
            opset_imports=self.model.model.opset_import,
            graph=onnx.helper.make_graph(
                name="partial",
                inputs=[],
                outputs=[
                    onnx.helper.make_tensor_value_info(
                        qdq_param_name, onnx.TensorProto.FLOAT, shape=p.shape
                    )
                    for qdq_param_name, p in qdq_params.items()
                ],
                initializer=[
                    init
                    for init in self.model.model.graph.initializer
                    if init.name in param_names
                ],
                nodes=[
                    node
                    for node in self.model.model.graph.node
                    if any(inp in param_names for inp in node.input)
                    or (
                        node.op_type == "Constant"
                        and any(out in param_names for out in node.output)
                    )
                ],
            ),
        )

        if not partial_model.graph.output:
            return {}

        sess = build_session(partial_model, ["CPUExecutionProvider"])
        out = sess.run(list(qdq_params.keys()), {})
        return {
            qdq_param_name: qdq_param
            for qdq_param_name, qdq_param in zip(qdq_params.keys(), out)
        }

    @staticmethod
    def _overwrite_parameters(
        model: onnx.ModelProto, parameters: Dict[str, np.ndarray]
    ):
        initializers = [
            (init, parameters.pop(f"{init.name}_qdq"))
            for init in model.graph.initializer
            if f"{init.name}_qdq" in parameters
        ]
        constants = [
            (node, parameters.pop(f"{node.output[0]}_qdq"))
            for node in model.graph.node
            if node.op_type == "Constant" and f"{node.output[0]}_qdq" in parameters
        ]

        found = set(init.name for init, _ in initializers) | set(
            const.output[0] for const, _ in constants
        )

        not_found = parameters.keys() - found

        if not_found:
            raise RuntimeError(f"Couldn't find parameters: {list(not_found)}")

        for const, _ in constants:
            if any(
                attr.name in ("value_string", "value_strings")
                for attr in const.attribute
            ):
                raise RuntimeError(f"String constant {const.name} can't be quantized")

        for init, qdq_param in initializers:
            init.raw_data = qdq_param.tobytes()

        for const, qdq_param in constants:
            for attr in const.attribute:
                if attr.name == "value":
                    attr.t.raw_data = qdq_param.tobytes()
                    break
                if attr.name == "value_float":
                    attr.float = float(qdq_param)
                    break
                if attr.name == "value_floats":
                    attr.ClearField("floats")
                    attr.floats.extend(qdq_param.astype(np.float32).tolist())
                    break
                if attr.name == "value_int":
                    attr.int = int(qdq_param)
                    break
                if attr.name == "value_ints":
                    attr.ClearField("ints")
                    attr.floats.extend(qdq_param.astype(np.int64).tolist())
                    break

    def _insert_data_movement_op_output_quantizers(self):
        """
        Insert data moevement op output quantizers.
        The newly inserted output quantizers will inherit the input encodings.
        This function is useful for export; encouraged to call this function right before export

        Example:

            >>> onnx_qdq = sim._to_onnx_qdq()
            >>> len([dq for dq in onnx_qdq.graph.node if dq.op_type == "DequantizeLinear"])
            10
            >>> sim._insert_data_movement_op_output_quantizers()
            >>> onnx_qdq = sim._to_onnx_qdq()
            >>> len([dq for dq in onnx_qdq.graph.node if dq.op_type == "DequantizeLinear"])
            15
        """
        data_movement_ops = [
            op
            for op in self.connected_graph.ordered_ops
            if _is_grid_preserving_op(op.type)
        ]

        def propogate_quantizer(op: Op):
            input_qtzr = self.qc_quantize_op_dict.get(op.inputs[0].name)
            output_qtzr = self.qc_quantize_op_dict.get(op.outputs[0].name)

            input_encoding = output_encoding = None

            if input_qtzr:
                input_encoding = input_qtzr.get_encodings()

            if output_qtzr:
                output_encoding = output_qtzr.get_encodings()

            if not input_encoding and not output_encoding:
                # No input/output encoding to inherit; skip
                return

            if input_encoding and output_encoding:
                # Both input and output encoding already exists; skip
                return

            if input_encoding:
                # Reuse input encoding for output quantization
                for output in op.outputs:
                    if output.name not in self.qc_quantize_op_dict:
                        self._insert_quantizer(output.name, is_param=False)
                    output_qtzr = self.qc_quantize_op_dict[output.name]
                    output_qtzr.enabled = True
                    output_qtzr.load_encodings(input_encoding)

                    # Rename model output node
                    for graph_output in self.model.model.graph.output:
                        if graph_output.name in output.name:
                            graph_output.name += "_updated"
                            break
            else:
                if len(op.inputs[0].consumers) > 1 or len(op.outputs) > 1:
                    # If input has more than one consumer or if there are more than one output,
                    # it is NOT safe to reuse output encoding for input quantization
                    return

                # Reuse output encoding for input quantization
                if not input_qtzr:
                    self._insert_quantizer(op.inputs[0].name, is_param=False)
                input_qtzr = self.qc_quantize_op_dict[op.inputs[0].name]
                input_qtzr.enabled = True
                input_qtzr.load_encodings(output_encoding)

        for op in data_movement_ops:
            propogate_quantizer(op)

        # Repeat in reverse-DFS order
        for op in reversed(data_movement_ops):
            propogate_quantizer(op)

    def _get_enabled_quantizer(self, tensor_name: str) -> QcQuantizeOp:
        """
        Returns closest enabled quantizer to tensor traversing upwards only through invariant ops

        :param tensor_name: Name of tensor for which to find quantizer
        """
        quantizer = self.qc_quantize_op_dict.get(tensor_name, None)
        if quantizer and quantizer.enabled:
            return quantizer

        prod_dict = self.connected_graph.get_all_products()
        product = prod_dict.get(tensor_name, None)

        if product == None:
            if tensor_name.endswith(("_updated", "_qdq")):
                raise KeyError(
                    f"Could not find quantizer for tensor {tensor_name}. Input tensor_name must be the name of a tensor in the original (unquantized) graph"
                )
            else:
                raise KeyError(
                    f"Could not find quantizer for tensor {tensor_name}. Tensor name does not exist in the graph"
                )

        producer = product.producer

        if producer == None:
            return None

        if not (_is_grid_preserving_op(producer.type)):
            return None

        if len(producer.inputs) == 0:
            return None

        upstream_tensor = producer.inputs[0]
        return self._get_enabled_quantizer(upstream_tensor.name)


def _to_unsigned_encoding(encoding: dict) -> dict:
    if ("output_dtype" not in encoding) or ("y_scale" not in encoding):
        raise RuntimeError(
            f"Expected 2.0.0 encoding format. Got unexpected keys: {list(encoding.keys())}"
        )

    encoding = encoding.copy()
    output_dtype = encoding["output_dtype"]

    if output_dtype.startswith("uint"):
        return encoding

    bw = int(output_dtype[3:])

    if "y_zero_point" in encoding:
        y_zero_point = np.array(encoding["y_zero_point"], dtype=np.int64)
    else:
        y_scale = np.array(encoding["y_scale"])
        y_zero_point = np.zeros(y_scale.shape, dtype=np.int64)

    # Update dtype from int to uint and shift zero_point accordingly
    encoding["output_dtype"] = "u" + output_dtype
    encoding["y_zero_point"] = (y_zero_point + 2 ** (bw - 1)).tolist()
    return encoding


# pylint: disable=too-many-locals, too-many-branches
def load_encodings_to_sim(
    quant_sim_model: QuantizationSimModel,
    onnx_encoding_path: str,
    strict=True,
    *,
    allow_overwrite=True,
    disable_missing_quantizers=True,
) -> List[_EncodingMismatchInfo]:
    """
    Loads the saved encodings to quant sim model. The encoding filename to load should end in .encodings,
    generated as part of quantsim export.

    :param quant_sim_model: Quantized model to load encodings for. Note: The model configuration should be the same as
        when encodings were exported.
    :param onnx_encoding_path: Path of the encodings file to load.
    :param strict: If set to True and encoding settings between encodings to load do not line up with Quantsim
        initialized settings, an assertion will be thrown. If set to False, quantizer settings will update to align with
        encodings to load.
    :param allow_overwrite: If true, loaded encodings will be overwritten by subsequent compute_encodings calls
        If false, loaded quantizer encodings will be frozen.
    :param diable_missing_quantizers: If true, quantizers which do not have encodings will be disabled.
    :return: List of EncodingMismatchInfo objects containing quantizer names and mismatched settings
    """
    mismatched_encodings = []

    # Load encodings file
    with open(onnx_encoding_path) as json_file:
        encodings = json.load(json_file)

    encoding_version = encodings.get("version", None)
    if encoding_version not in VALID_ENCODING_VERSIONS:
        raise NotImplementedError(
            f"Encoding version should be one of {VALID_ENCODING_VERSIONS}; "
            f"got {encoding_version}"
        )

    if encoding_version not in ("0.6.1", "1.0.0"):
        raise NotImplementedError(
            "load_encodings_to_sim only supports encoding version 0.6.1 and 1.0.0; "
            f"got {encoding_version}"
        )

    if encoding_version == "0.6.1":
        encodings["activation_encodings"] = _convert_encoding_format_0_6_1_to_1_0_0(
            encodings["activation_encodings"]
        )
        encodings["param_encodings"] = _convert_encoding_format_0_6_1_to_1_0_0(
            encodings["param_encodings"]
        )

    validate_encodings_to_load(encodings, quant_sim_model)

    # First pass through quantizers to check for mismatched encodings
    param_encodings = {
        encoding["name"]: encoding for encoding in encodings["param_encodings"]
    }
    activation_encodings = {
        encoding["name"]: encoding for encoding in encodings["activation_encodings"]
    }

    # If quantizer not in qc_quantize_op_dict, that is equivalent to being disabled
    missing_quantizers = (
        param_encodings.keys() | activation_encodings.keys()
    ) - quant_sim_model.qc_quantize_op_dict.keys()
    for name in missing_quantizers:
        mismatched_encodings.append(
            _EncodingMismatchInfo(name, enabled_mismatch=(False, True))
        )

    for quantizer_name, quantizer in quant_sim_model.qc_quantize_op_dict.items():
        if (
            quantizer_name not in param_encodings
            and quantizer_name not in activation_encodings
        ):
            mismatched_info = get_encoding_mismatch_info(
                quantizer_name, quantizer, None
            )
            if mismatched_info.has_mismatch():
                mismatched_encodings.append(mismatched_info)
            continue

        if quantizer_name in activation_encodings:
            encodings_to_load = activation_encodings[quantizer_name]
        else:
            encodings_to_load = param_encodings[quantizer_name]

        mismatched_info = get_encoding_mismatch_info(
            quantizer_name, quantizer, encodings_to_load
        )
        if mismatched_info.has_mismatch():
            mismatched_encodings.append(mismatched_info)

    log_and_catch_mismatched_encodings(mismatched_encodings, strict)
    if missing_quantizers and not strict:
        _add_missing_quantizers(encodings, quant_sim_model)

    # Second pass through quantizers to set quantizer settings
    for quantizer_name, quantizer in quant_sim_model.qc_quantize_op_dict.items():
        if (
            quantizer_name not in activation_encodings
            and quantizer_name not in param_encodings
        ):
            if disable_missing_quantizers:
                quantizer.enabled = False
            continue

        if quantizer_name in activation_encodings:
            encodings_to_load = activation_encodings[quantizer_name]
        else:
            encodings_to_load = param_encodings[quantizer_name]

        # pylint: disable=protected-access
        quant_sim_model.qc_quantize_op_dict[quantizer_name]._load_encodings_dict(
            encodings_to_load, allow_overwrite=allow_overwrite
        )
    return mismatched_encodings


def validate_encodings_to_load(
    encodings_to_load: Dict, quant_sim_model: QuantizationSimModel
):
    """
    Validate that all names of encodings to load correspond to quantizable tensors in the model.

    :param encodings_to_load: Encodings to load
    :param quant_sim_model: Quantsim model to check for encoding names.
    """
    # Check that all encoding names in the encodings to load are found in the model. This check only works for verifying
    # that names in encodings_to_load are valid. The reverse check will not work, since quantizers which are disabled
    # will not show up in encodings_to_load.
    encoding_names_not_found = []
    non_quantizable_tensors_found = set()
    for encoding in itertools.chain(
        encodings_to_load["activation_encodings"], encodings_to_load["param_encodings"]
    ):
        name = encoding["name"]
        # If quantizer already exists, continue
        if name in quant_sim_model.qc_quantize_op_dict:
            continue
        # If name not in connected_graph.get_all_products(), it is not a tensor in the model
        if name not in quant_sim_model.connected_graph.get_all_products():
            encoding_names_not_found.append(name)
        # Check if encoding corresponds to non-quantizable tensor type
        if not quant_sim_model._is_quantizable_dtype(name):  # pylint:disable = protected-access
            non_quantizable_tensors_found.add(name)

    if encoding_names_not_found:
        logger.error(
            "The following encoding names were present in the encodings to load but not found in the model: "
            "%s",
            str(encoding_names_not_found),
        )
        raise AssertionError(
            "The following encoding names were present in the encodings to load but not found in the "
            "model: " + str(encoding_names_not_found)
        )

    if non_quantizable_tensors_found:
        msg = (
            "The following encoding names were present in the encodings to load but are of a data-type not supported for quantization "
            f"in aimet_onnx:\n{non_quantizable_tensors_found}"
        )
        logger.error(msg)
        raise RuntimeError(msg)


def _add_missing_quantizers(
    encodings_to_load: Dict[str, List], sim: QuantizationSimModel
):
    """
    Add quantizers for any tensors which are present in encodings_to_load but are not present in
    sim.qc_quantize_op_dict
    """
    # pylint:disable = protected-access
    act_encodings, param_encodings = (
        encodings_to_load["activation_encodings"],
        encodings_to_load["param_encodings"],
    )
    added_quantizers = set()
    # Insert any missing activation quantizers as disabled act quantizers
    for enc in act_encodings:
        tensor_name = enc["name"]
        if tensor_name not in sim.qc_quantize_op_dict:
            sim._insert_quantizer(tensor_name, is_param=False)
            sim.qc_quantize_op_dict[tensor_name].enabled = False
            sim.activation_names.append(tensor_name)
            added_quantizers.add(tensor_name)

    # Insert any missing param quantizers as disabled param quantizers
    for enc in param_encodings:
        tensor_name = enc["name"]
        if tensor_name not in sim.qc_quantize_op_dict:
            sim._insert_quantizer(tensor_name, is_param=True)
            sim.qc_quantize_op_dict[tensor_name].enabled = False
            sim.param_names.append(tensor_name)
            added_quantizers.add(tensor_name)

    if added_quantizers:
        logger.info(
            "Added new quantizers to graph for tensors: %s", str(added_quantizers)
        )
        sim._rebuild_session()


def log_and_catch_mismatched_encodings(
    mismatched_encodings: List[_EncodingMismatchInfo], strict: bool
):
    """
    If mismatched_encodings is not empty, log details for each entry. If strict is True, raise an AssertionError.

    :param mismatched_encodings: List of mismatched quantizer names and encoding settings
    :param strict: If True, raise an AssertionError if there are mismatched settings
    """
    if mismatched_encodings:
        logging_strings = [
            "The following quantizers had settings not matching with provided encodings to load:"
        ]
        for mismatched_encoding_info in mismatched_encodings:
            logging_strings.append(mismatched_encoding_info.quantizer_name + ":")
            if mismatched_encoding_info.enabled_mismatch:
                logging_strings.append(
                    f"\tenabled: {mismatched_encoding_info.enabled_mismatch[0]}, "
                    f"loaded encoding enabled: "
                    f"{mismatched_encoding_info.enabled_mismatch[1]}"
                )

            if mismatched_encoding_info.dtype_mismatch:
                logging_strings.append(
                    f"\tdtype: {mismatched_encoding_info.dtype_mismatch[0]}, "
                    f"loaded encoding dtype: "
                    f"{mismatched_encoding_info.dtype_mismatch[1]}"
                )

            if mismatched_encoding_info.bitwidth_mismatch:
                logging_strings.append(
                    f"\tbitwidth: "
                    f"{mismatched_encoding_info.bitwidth_mismatch[0]}, loaded encoding bitwidth:"
                    f"{mismatched_encoding_info.bitwidth_mismatch[1]}"
                )

            if mismatched_encoding_info.is_symmetric_mismatch:
                logging_strings.append(
                    f"\tsymmetric: "
                    f"{mismatched_encoding_info.is_symmetric_mismatch[0]}, "
                    f"loaded encoding symmetric: "
                    f"{mismatched_encoding_info.is_symmetric_mismatch[1]}"
                )

            if mismatched_encoding_info.is_strict_symmetric_mismatch:
                logging_strings.append(
                    f"\tstrict symmetric: "
                    f"{mismatched_encoding_info.is_strict_symmetric_mismatch[0]}, "
                    f"loaded encoding strict symmetric: "
                    f"{mismatched_encoding_info.is_strict_symmetric_mismatch[1]}"
                )

            if mismatched_encoding_info.is_unsigned_symmetric_mismatch:
                logging_strings.append(
                    f"\tunsigned symmetric: "
                    f"{mismatched_encoding_info.is_unsigned_symmetric_mismatch[0]}, "
                    f"loaded encoding unsigned symmetric: "
                    f"{mismatched_encoding_info.is_unsigned_symmetric_mismatch[1]}"
                )

            if mismatched_encoding_info.enc_type_mismatch:
                logging_strings.append(
                    f"\tencoding type: "
                    f"{mismatched_encoding_info.enc_type_mismatch[0]}, "
                    f"loaded encoding encoding type: "
                    f"{mismatched_encoding_info.enc_type_mismatch[1]}"
                )
        log_message = "\n".join(logging_strings)
        if strict:
            logger.error(log_message)
            raise AssertionError(log_message)
        logger.info(log_message)


def _parse_compute_encodings_args(*args, **kwargs):
    # Default error message to display for unsupported argument combinations
    msg = (
        f"compute_encodings() supports the following function signatures:\n\n"
        " * (inputs: Iterable[Dict[str, np.ndarray]])\n"
        " * (forward_pass_callback: Callable[[InferenceSession], Any])\n"
        " * (forward_pass_callback: Callable[[InferenceSession, T], Any], forward_pass_callback_args: T)\n"
        f"but receieved: args={[type(arg) for arg in args]}, kwargs={ {key: type(val) for key, val in kwargs.items()} }"
    )

    inputs = kwargs.pop("inputs", None)
    forward_pass_callback = kwargs.pop("forward_pass_callback", None)
    forward_pass_callback_args = kwargs.pop(
        "forward_pass_callback_args", _NOT_SPECIFIED
    )

    if kwargs:
        raise TypeError(msg)
    if args and (inputs or forward_pass_callback):
        raise TypeError(msg)
    if len(args) > 2:
        raise TypeError(msg)
    if len(args) == 2:
        if forward_pass_callback_args is not _NOT_SPECIFIED:
            raise TypeError(msg)
        forward_pass_callback, forward_pass_callback_args = args
    elif len(args) == 1:
        if isinstance(args[0], Iterable):
            inputs = args[0]
        elif callable(args[0]):
            forward_pass_callback = args[0]
        else:
            raise TypeError(
                f"First positional argument to compute_encodings() must be callable or iterable, received {type(args[0])}"
            )

    if inputs and (
        forward_pass_callback or forward_pass_callback_args is not _NOT_SPECIFIED
    ):
        raise TypeError(msg)
    if not (inputs or forward_pass_callback):
        raise TypeError(msg)

    return inputs, forward_pass_callback, forward_pass_callback_args


# pylint: disable=protected-access
def get_encoding_mismatch_info(
    quantizer_name: str,
    quantizer: QcQuantizeOp,
    encodings_to_load: Optional[List[Dict]],
) -> _EncodingMismatchInfo:
    """
    Check that quantizer settings align with the settings in encodings_to_load. If settings do not align, track the
    mismatching settings in a EncodingMismatchInfo object and add it to mismatched_encodings_info list.

    :param quantizer_name: Name of quantizer to check
    :param quantizer: Quantizer to check
    :param encodings_to_load: Encodings to check
    """
    encoding_mismatch_info = _EncodingMismatchInfo(quantizer_name)
    # pylint: disable=protected-access
    quantizer._fill_mismatching_encoding_settings_info(
        encodings_to_load, encoding_mismatch_info
    )
    return encoding_mismatch_info


def set_blockwise_quantization_for_weights(
    sim: QuantizationSimModel,
    op_types: Union[str, Tuple],
    bitwidth: int,
    symmetric: bool,
    block_size: int,
    strict: bool = False,
    excluded_nodes: List[str] = None,
):
    """
    Set weight quantizers for the given operator types to use blockwise affine quantization.

    :param sim: Quantsim object to configure weight quantizers for
    :param op_types: Operator types for which to enable blockwise weight quantizaiton
    :param bitwidth: Bitwidth for quantization
    :param symmetric: True if quantization is symmetric, False otherwise
    :param block_size: Block size for affine quantization. The block size will be applied to the weight's input features
        dimension, while per-channel will be used for the weight's output features dimension
    :param strict: If False, only enable blockwise quant for layers with dimensions evenly divisible by block_size.
        If True, throw an error for layers with incompatible shapes.
    :param excluded_nodes: List of onnx node names to exclude from blockwise weight quantization. It can be empty if no nodes are excluded


    Examples:

        >>> # Assume 'sim' is a QuantizationSimModel object
        >>> # Allows setting of all Linear and Conv weight quantizers to block_size 64 in the input_channels dimension:
        >>> set_blockwise_quantization_for_weights(sim=sim,
        ...                                        op_types=("Gemm", "MatMul", "Conv"),
        ...                                        bitwidth=4,
        ...                                        symmetric=True,
        ...                                        block_size=64,
    ...                                            excluded_nodes = ['conv1'])
    """

    if isinstance(op_types, str):
        op_types = (op_types,)

    if not excluded_nodes:
        excluded_nodes = []

    for op in sim.connected_graph.ordered_ops:
        if op.type not in op_types:
            continue

        if op.name in excluded_nodes:
            continue

        _, _, param_quantizers = sim.get_op_quantizers(op)

        weight_quantizer: QcQuantizeOp = param_quantizers.get("weight")
        bias_quantizer: QcQuantizeOp = param_quantizers.get("bias")

        if not weight_quantizer:
            continue

        try:
            weight_quantizer._enable_blockwise_quantization(block_size)  # pylint:disable = protected-access
        except ValueError as e:
            if strict:
                raise e
        else:
            weight_quantizer.set_bitwidth(bitwidth)
            weight_quantizer.use_symmetric_encodings = symmetric
            weight_quantizer.data_type = QuantizationDataType.int

            if bias_quantizer:
                # Enable per-channel quantization of bias to derive bias_scale analytically as
                # :math:`bias_scale = weight_scale * input_scale` in export time.
                # ``bias_scale`` should be per-channel if ``weight_scale`` is per-channel or per-block
                # to match the shape
                bias_quantizer.enable_per_channel_quantization()
                bias_quantizer.use_symmetric_encodings = symmetric
                bias_quantizer.data_type = QuantizationDataType.int


def set_grouped_blockwise_quantization_for_weights(
    sim: QuantizationSimModel,
    op_types: Union[str, Tuple],
    bitwidth: int,
    decompressed_bw: int,
    block_size: int,
    strict: bool = False,
    excluded_nodes: List[str] = None,
):
    """
    Set weight parameter quantizers of modules to grouped blockwise quantization.

    :param sim: Quantsim to set weight quantizers for
    :param op_types: Operator types for which to enable grouped blockwise weight quantizaiton
    :param bitwidth: Bitwidth for affine quantization
    :param decompressed_bw: Decompressed bw for grouped block quantization
    :param block_size: Block size for affine quantization. The block size will be applied to the weight's input features
        dimension, while per-channel will be used for the weight's output features dimension
    :param excluded_nodes: List of onnx node names to exclude from blockwise weight quantization. It can be empty if no nodes are excluded

    Examples:

        >>> # Assume 'sim' is a QuantizationSimModel object
        >>> # Sets of all Gemm, MatMul, and Conv weight quantizers to block_size 64 in the input_channels dimension:
        >>> set_grouped_blockwise_quantization_for_weights(sim=sim,
        ...                                                op_types=("Gemm", "MatMul", "Conv"),
        ...                                                bitwidth=4,
        ...                                                decompressed_bw=8,
        ...                                                block_size=64,
        ...                                                excluded_nodes = ['conv1'])
    """
    if isinstance(op_types, str):
        op_types = (op_types,)

    if not excluded_nodes:
        excluded_nodes = []

    def get_lpbq_params(op: Op):
        if op.type in op_types and op.name not in excluded_nodes:
            return bitwidth, decompressed_bw, block_size
        return None, None, None

    return _set_grouped_blockwise_quantization_for_weights(sim, get_lpbq_params, strict)


def _set_grouped_blockwise_quantization_for_weights(
    sim: QuantizationSimModel,
    get_lpbq_params: Callable[[Op], Tuple[Optional[int], Optional[int], Optional[int]]],
    strict: bool = False,
):
    for op in sim.connected_graph.ordered_ops:
        bitwidth, decompressed_bw, block_size = get_lpbq_params(op)

        if None in (bitwidth, decompressed_bw, block_size):
            continue

        _, _, param_quantizers = sim.get_op_quantizers(op)

        weight_quantizer: QcQuantizeOp = param_quantizers.get("weight")
        bias_quantizer: QcQuantizeOp = param_quantizers.get("bias")

        if not weight_quantizer:
            continue

        try:
            grouped_quantizer = GroupedBlockQuantizeDequantize(
                weight_quantizer.quant_info,
                bitwidth,
                decompressed_bw,
                block_size,
                weight_quantizer.quant_scheme,
                weight_quantizer.op_mode,
                weight_quantizer.tensor_quantizer_params,
            )
        except ValueError as e:
            if strict:
                raise e
        else:
            if bias_quantizer:
                # Enable per-channel quantization of bias to derive bias_scale analytically as
                # :math:`bias_scale = weight_scale * input_scale` in export time.
                # ``bias_scale`` should be per-channel if ``weight_scale`` is per-channel or per-block
                # to match the shape
                bias_quantizer.enable_per_channel_quantization()
                bias_quantizer.use_symmetric_encodings = True
                bias_quantizer.data_type = QuantizationDataType.int

            for name, quantizer in sim.qc_quantize_op_dict.items():
                if quantizer is weight_quantizer:
                    sim.qc_quantize_op_dict[name] = grouped_quantizer


# pylint: disable=protected-access
def clamp_activation_encodings(quant_sim: QuantizationSimModel, clamp_val: float):
    """
    Clamp activations to specific range if out of bound.

    :param quant_sim: quantsim object
    :param clamp_val: positive float value
    :return:
    """
    for act_name in quant_sim.activation_names:
        quantizer = quant_sim.qc_quantize_op_dict.get(act_name)
        is_clipped = quantizer.clip_and_recompute_encodings(clamp_val)
        if is_clipped:
            logger.info("Clamped tensor %s", act_name)
