# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Implementation for simulating models running on Quantized hardware"""

# pylint: disable=wrong-import-order
from collections import defaultdict
import contextlib
import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
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
import json
import re
import warnings
import numpy as np
import onnx
import onnx_ir
from onnx_ir.passes.common import ShapeInferencePass

from onnx import helper
import onnxruntime as ort
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from packaging import version
import google.protobuf.message

from aimet_onnx.common import libpymo, quantsim
from aimet_onnx.common import libquant_info
from aimet_onnx.common.defs import (
    QuantScheme,
    QuantizationDataType,
    qtype,
    QTYPE_ALIASES,
    Float,
    int2,
    int8,
    int16,
    EncodingType,
    _quant_scheme_aliases,
)
from aimet_onnx.common.onnx._utils import (
    _add_onnx_qdq_nodes,
    _remove_onnx_qdq_nodes,
    _is_grid_preserving_op,
    _is_grid_equivariant_op,
    _derive_data_movement_op_encodings,
    _is_htp_interpolation_op,
    _get_all_constants,
    _iterate_graphs_recursive,
    contains_tensor_type,
    to_array,
)
from aimet_onnx.graph_passes.cleanup import remove_duplicate_qdq_pairs
from aimet_onnx.common.quantsim import (
    extract_global_quantizer_args,
    VALID_ENCODING_VERSIONS,
    _INT32_MINIMUM_SCALE,
    _adjust_weight_scale_against_bias_overflow,
    _adjust_weight_scale_against_export_dtype_underflow,
    _adjust_weight_scale_against_scale_underflow,
    compute_min_max_given_delta_offset,
)
from aimet_onnx.common.utils import (
    save_json_yaml,
    AimetLogger,
    _red,
    deprecated,
    Handle,
    docstring,
)
from aimet_onnx.common.quantsim_config.quantsim_config import _config_file_aliases
from aimet_onnx.common.connected_graph.product import Product
from aimet_onnx.common.onnx._utils import _convert_version
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
    _EncodingMismatchInfo,
)
from aimet_onnx.quantsim_config.quantsim_config import QuantSimConfigurator
from aimet_onnx.utils import (
    build_session,
    make_dummy_input,
    add_hook_to_get_activation,
    remove_activation_hooks,
    register_aimet_custom_ops,
    OrtInferenceSession,
)
from aimet_onnx.graph_passes.fusions import (
    inline_all_supergroups,
    is_fused_supergroup,
)
from aimet_onnx.batch_norm_fold import _has_unfolded_batchnorms
import aimet_onnx
from aimet_onnx import ir_utils
from ._encoding import EncodingBase, FloatEncoding, _QDQ_FLOAT_TYPES
from .defs import QSpec

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
    "AveragePool",
    "Relu",
]
_tie_qtzrs = True

_fuse_supergroups = True

# Experimental: also quantize tensors inside Scan bodies.
# NOTE: export() and to_onnx_qdq() are not supported yet with this flag.
_quantize_subgraphs = False

data_types_to_quantize = [np.float32, np.float16, np.dtype("bfloat16")]

_DEPRECATED_ARGS = {
    "rounding_mode",
    "default_param_bw",
    "default_activation_bw",
    "use_symmetric_encodings",
    "default_data_type",
    "use_cuda",
    "device",
}

_NORM_OP_TYPES = {
    "BatchNormalization",
    "LayerNormalization",
    "GroupNormalization",
    "RMSNormalization",
    "InstanceNormalization",
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


@contextlib.contextmanager
def _quantize_subgraphs_enabled(flag: bool = True):
    """
    Quantize tensors inside Scan bodies in QuantizationSimModels created within this context.
    """
    global _quantize_subgraphs  # pylint: disable=global-statement
    orig_flag = _quantize_subgraphs
    try:
        _quantize_subgraphs = flag
        yield
    finally:
        _quantize_subgraphs = orig_flag


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
    if getattr(sim, "_is_computing_encodings", False):
        yield
        return

    enabled_quantizers = {
        name: q for name, q in sim.qc_quantize_op_dict.items() if q.enabled
    }
    original_states = {
        name: (q.op_mode, q.get_encodings()) for name, q in enabled_quantizers.items()
    }
    sim._is_computing_encodings = True  # pylint: disable=protected-access
    try:
        for op_name, qc_op in enabled_quantizers.items():
            qc_op.reset_encoding_stats()
            if op_name in sim.activation_names:
                qc_op.op_mode = OpMode.updateStats
            else:
                qc_op.op_mode = OpMode.oneShotQuantizeDequantize
                if qc_op.is_encoding_frozen():
                    qc_op.op_mode = OpMode.quantizeDequantize

        yield
    except BaseException:
        for name, qc_op in enabled_quantizers.items():
            op_mode, encodings = original_states[name]
            if encodings:
                qc_op.load_encodings(encodings)
            qc_op.op_mode = op_mode
        raise
    else:
        for op_name, qc_op in enabled_quantizers.items():
            if not qc_op.is_encoding_frozen():
                qc_op.compute_encodings()
            qc_op.op_mode = OpMode.quantizeDequantize

        sim._adjust_weight_scales_against_overflow()  # pylint: disable=protected-access
    finally:
        delattr(sim, "_is_computing_encodings")


def _fill_missing_node_names(model: onnx.ModelProto):
    """
    Fill missing node names in the ONNX model with unique names.

    :param model: ONNX model
    """
    seen: Set[str] = set()
    idx_factory = defaultdict(int)

    # The top-level graph is walked first, so its nodes keep their names on a clash.
    nodes = (
        node for graph in _iterate_graphs_recursive(model.graph) for node in graph.node
    )
    for node in nodes:
        if node.name and node.name not in seen:
            seen.add(node.name)
            continue

        new_name = f"/{node.op_type}"
        i = idx_factory[node.op_type]
        while new_name in seen:
            i += 1
            new_name = f"/{node.op_type}_{i}"

        node.name = new_name
        seen.add(new_name)
        idx_factory[node.op_type] = i + 1


# pylint: disable=missing-class-docstring, too-many-arguments, too-many-locals, too-many-instance-attributes
class QuantizationSimModel:
    __doc__ = f"""
    Class that simulates the quantized model execution on a target hardware backend.

    Args:
        model (onnx.ModelProto): ONNX ModelProto to quantize
        param_type (qtype | str): quantized type to use for parameter tensors.
            Can be {{ {", ".join(QTYPE_ALIASES)} }} or :class:`aimet_onnx.qtype`
        activation_type (qtype | str): quantized type to use for activation tensors.
            Can be {{ {", ".join(QTYPE_ALIASES)} }} or :class:`aimet_onnx.qtype`
        quant_scheme (QuantScheme | str): Quantization scheme to use for calibration.
            Can be {{ {", ".join(_quant_scheme_aliases.keys() - {"tf", "percentile"})} }} or :class:`QuantScheme`
        config_file (str, optional): File path or alias of the configuration file.
            Alias can be one of {{ {", ".join(_config_file_aliases.keys())} }} (Default: `"default"`)
        dummy_input (Dict[str, np.ndarray], optional): Sample input to the model. Only needed for non shape-inferable models with parameterized shapes
        user_onnx_libs (List[str], optional): List of paths to all compiled ONNX custom ops libraries
        providers (List, optional): Onnxruntime execution providers to use when building InferenceSession.
            If `None`, default provider is "CPUExecutionProvider"
        path (str, optional): Directory to save temporary artifacts.
        session_options_kwargs (Dict[str, Any], optional): Attributes to set on the sim's
            ``onnxruntime.SessionOptions`` (e.g. ``{{"intra_op_num_threads": 2}}``). An unknown
            attribute name raises `AttributeError`.
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
        session_options_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(quant_scheme, str):
            quant_scheme = QuantScheme.from_str(quant_scheme)

        if isinstance(model, ModelProto):
            model = ONNXModel(model)

        if any(node.op_type == "QcQuantizeOp" for node in model.nodes()):
            raise RuntimeError(
                "Model already contains QcQuantizeOp nodes. Reload the original model to instantiate QuantizationSimModel."
            )

        if any(
            node.op_type in {"QuantizeLinear", "DequantizeLinear"}
            for node in model.nodes()
        ):
            raise RuntimeError(
                "Model contains QuantizeLinear/DequantizeLinear nodes. Use `QuantizationSimModel.from_onnx_qdq()` to create sim from ONNX QDQ model."
            )

        _fill_missing_node_names(model.model)

        if isinstance(param_type, str):
            param_type = qtype.from_string(param_type)

        if isinstance(activation_type, str):
            activation_type = qtype.from_string(activation_type)

        for dtype in (param_type, activation_type):
            if dtype in QTYPE_ALIASES.values():
                continue

            # Only aliased float types are supported.
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

        # Note: bfloat16 I/O is not supported via session.run and will fail during calibration
        bf16_io = [
            io.name
            for io in (*model.model.graph.input, *model.model.graph.output)
            if io.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
        ]
        if bf16_io:
            raise RuntimeError(
                f"BFLOAT16 model inputs/outputs are not supported by "
                f"QuantizationSimModel. Offending tensors: {bf16_io}. "
                f"Only intermediate BFLOAT16 tensors are supported."
            )

        self._op_domain = op_domain
        self.providers = providers

        self.qc_quantize_op_dict = {}
        self._quant_scheme = quant_scheme
        self._param_type = param_type
        self._activation_type = activation_type
        base_session_options = ort.SessionOptions()
        for attr, value in (session_options_kwargs or {}).items():
            setattr(base_session_options, attr, value)
        self._ort_session_options = register_aimet_custom_ops(base_session_options)
        self.param_names = []
        # Param quantizers that have been folded into their parameters via
        # fold_param_quantizers(). Their QcQuantizeOp nodes are removed from the
        # graph, but the quantizer objects are retained here so that their
        # encodings can still be exported.
        self._folded_param_quantizers = {}
        self.input_quantizers_name = []
        self.activation_names = []
        self.activation_dtypes = {}
        self._path = path
        if self._path:
            os.makedirs(self._path, exist_ok=True)

        # Register user provided custom libs into ORT session options
        for lib in user_onnx_libs or []:
            self._ort_session_options.register_custom_ops_library(lib)

        quantsim_configurator = QuantSimConfigurator(
            config_file,
            self._param_type,
            self._activation_type,
        )
        if _fuse_supergroups:
            model = quantsim_configurator.apply_fusions(model)

        self.model = model
        self._quantize_subgraphs = _quantize_subgraphs
        self.connected_graph = ConnectedGraph(self.model)

        if _has_unfolded_batchnorms(self.model.model, self.connected_graph):
            logger.warning(
                "Model contains unfolded BatchNormalization layers. To accurately simulate quantization behavior, "
                "please call aimet_onnx.batch_norm_fold.fold_all_batch_norms_to_weight(model) before creating QuantizationSimModel."
            )

        # Get names of parameters and activations to quantize
        self._get_param_names()
        self._get_activations_to_quantize(dummy_input)

        self._add_quantization_nodes()
        self._producers: dict[str, onnx.NodeProto] = {
            output: node
            for graph in self._graphs_to_quantize()
            for node in graph.node
            for output in node.output
        }

        # Apply configurations based on provided config file.
        quantsim_configurator.configure_quantizers(
            self.model,
            self.connected_graph,
            self.qc_quantize_op_dict,
            self.param_names,
            self.activation_names,
            self.input_quantizers_name,
        )
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
            op_types = {
                node.op_type
                for graph in self._graphs_to_quantize()
                for node in graph.node
            }
            op_types_to_tie = op_types_to_tie_qtzrs + [
                t
                for t in op_types
                if _is_htp_interpolation_op(t)
                # Multi-input grid-equivariant ops are safe to tie encodings
                or _is_grid_equivariant_op(t, include_unary=False)
            ]
            self._tie_quantizers_for_op_types(op_types_to_tie)

        # Always tie RNN hidden state quantizers regardless of _tie_qtzrs flag
        self._tie_rnn_hidden_state_quantizers()

        try:
            self._use_external_data = (
                self.model.model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF
            )
        except google.protobuf.message.EncodeError:
            self._use_external_data = True

        self._sort_quantizer_nodes()
        self.session = OrtInferenceSession(
            self.model.model,
            self.providers,
            session_options=self._ort_session_options,
            path=self._path,
            save_as_external_data=self._use_external_data,
        )

    @classmethod
    def from_onnx_qdq(
        cls, model: ModelProto, strict: bool = False, **kwargs
    ) -> "QuantizationSimModel":
        """
        Create sim from an ONNX QDQ model that contains QuantizeLinear/DequantizeLinear nodes.

        This method is designed to construct a fully quantized model
        based on a partially quantized ONNX QDQ model exported from
        aimet-torch (:func:`aimet_torch.onnx.export`) or other 3rd party tools.

        Args:
            model: ONNX model that contains QuantizeLinear/DequantizeLinear
            strict:
                If True, raises error if there are QuantizeLinear/DequantizeLinear nodes
                in the model that cannot be mapped to any quantizer in the sim.
                If False, ignores incompatible QuantizeLinear/DequantizeLinear nodes and
                only loads the rest of the encodings. (Default: False)
            **kwargs: same as QuantizationSimModel.__init__
        Returns:
            QuantizationSimModel: QuantizationSimModel created from ONNX QDQ model

        Example:
            >>> sim = aimet_onnx.QuantizationSimModel.from_onnx_qdq(
            ...     onnx.load("model_qdq.onnx"),
            ...     config_file="htp_v81",
            ... )
            Quant - INFO - Loaded 26 out of 63 encodings from QuantizeLinear/DequantizeLinear nodes
        """
        sim = cls._from_onnx_qdq(model, strict=strict, **kwargs)

        loaded = [
            q
            for q in sim.qc_quantize_op_dict.values()
            if q.enabled
            and q.data_type == QuantizationDataType.int
            and q.is_initialized()
        ]
        all_ = [
            q
            for q in sim.qc_quantize_op_dict.values()
            if q.enabled and q.data_type == QuantizationDataType.int
        ]
        # pylint: disable=logging-fstring-interpolation
        logger.info(
            f"Loaded {len(loaded)} out of {len(all_)} encodings from QuantizeLinear/DequantizeLinear nodes"
        )
        return sim

    @classmethod
    def _from_onnx_qdq(
        cls, model: ModelProto, strict: bool = False, **kwargs
    ) -> "QuantizationSimModel":
        """
        Create sim from onnx QDQ model with following strategy

        1. Remove Q/DQ nodes from model
        2. Extract encodings from the removed Q/DQ nodes
        3. Create QuantizationSimModel
        4. Load extracted encodings to sim

        Args:
            model: ONNX model that contains QuantizeLinear/DequantizeLinear
            strict:
                If True, raises error if there are QuantizeLinear/DequantizeLinear nodes
                in the model that cannot be mapped to any quantizer in the sim.
                If False, ignores incompatible QuantizeLinear/DequantizeLinear nodes and
                only loads the rest of the encodings. (Default: False)
            **kwargs: same as QuantizationSimModel.__init__
        """
        # pylint: disable=protected-access

        # Optimize Q->DQ->Q->DQ patterns by removing duplicate Q->DQ pairs
        removed_count, model = remove_duplicate_qdq_pairs(model)
        if removed_count > 0:
            logger.info(
                "Removed %d duplicate Q->DQ pairs from QDQ model", removed_count
            )

        # Removes Q/DQ node from model and extract them into 2.0.0 json encoding
        encodings = _remove_onnx_qdq_nodes(model)
        encodings = {enc["name"]: enc for enc in encodings}

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

        _remove_delegatable_excess_encodings(sim, encodings)

        excess_encodings = encodings.keys() - (quantizable_tensor_names | bias_names)

        if excess_encodings:
            if strict:
                raise NotImplementedError(
                    "Unexpected QuantizeLinear/DequantizeLinear nodes were found "
                    "for the following tensors: "
                    f"{excess_encodings}"
                )
            else:
                logger.warning(  # pylint: disable=logging-fstring-interpolation
                    "Unexpected QuantizeLinear/DequantizeLinear nodes were found. "
                    "The encodings for the following tensors will be ignored: "
                    f"{excess_encodings}"
                )
                encodings = {
                    name: enc
                    for name, enc in encodings.items()
                    if name not in excess_encodings
                }

        # Make sure each encoding is associated with only one quantizer
        sim.set_quantizers(
            {
                name: qtzr._copy() if name in encodings else qtzr
                for name, qtzr in sim.qc_quantize_op_dict.items()
            }
        )

        load_encodings_to_sim(
            sim,
            {"version": "2.0.0", "encodings": list(encodings.values())},
            strict=False,
            allow_overwrite=False,
            disable_missing_quantizers=False,
        )

        return sim

    def get_supported_kernels(self) -> Dict:
        """
        Return _supported_kernels parsed from the config file
        :return: Dictionary containing supported_kernels
        """
        return self._supported_kernels

    def _graphs_to_quantize(self) -> List[onnx.GraphProto]:
        """
        Returns the top-level graph, followed by the Scan bodies if subgraph quantization is enabled.

        NOTE: Computed on every call since rebuilding graph.node (e.g. _remove_quantization_nodes)
        copies the nodes, invalidating references to the bodies they hold.
        """
        if not self._quantize_subgraphs:
            return [self.model.graph()]
        return list(_iterate_graphs_recursive(self.model.graph()))

    def _get_param_names(self):
        """
        Get the names of params

        NOTE: Unlike activations, which are only gathered from _graphs_to_quantize(), params come from
        connected graph ops, which include body ops, so body params are excluded explicitly.
        """
        # Without subgraph quantization, params declared inside a body are not quantized.
        body_params = set()
        if not self._quantize_subgraphs:
            for graph in list(_iterate_graphs_recursive(self.model.graph()))[1:]:
                body_params.update(init.name for init in graph.initializer)
                body_params.update(name for node in graph.node for name in node.output)

        valid_ops = self._get_ops_with_parameter()
        for op in valid_ops:
            for param_info in op.parameters.values():
                param, _ = param_info
                if (
                    param.name
                    and param.name not in self.param_names
                    and param.name not in body_params
                ):
                    self.param_names.append(param.name)

    def _get_ops_with_parameter(self) -> List[Op]:
        """
        Gets ops with parameters to add quantization nodes for

        :return: Connected graph ops
        """
        valid_ops = list(self.connected_graph.get_all_ops().values())
        return valid_ops

    def _get_activations_to_quantize(self, dummy_input: Dict[str, np.ndarray] | None):
        """
        Get the names of activations to quantize

        :param dummy_input: Sample input to be run through the model
        """
        try:
            self.activation_dtypes = self._infer_activation_dtypes()
        except onnx.shape_inference.InferenceError:
            if dummy_input is None:
                dummy_input = make_dummy_input(self.model.model)
            self.activation_dtypes = self._observe_activation_dtypes(dummy_input)

        graphs = self._graphs_to_quantize()
        self.input_name_to_nodes = defaultdict(list)
        self.output_name_to_node = {}
        for graph in graphs:
            for node in graph.node:
                for name in node.input:
                    if name:
                        self.input_name_to_nodes[name].append(node)
                for name in node.output:
                    if name:
                        self.output_name_to_node[name] = node

        # Capture model inputs and body inputs
        for node in (node for graph in graphs for node in graph.input):
            name = node.name
            if (
                name not in self.activation_names
                and name not in self.param_names
                and self._is_tensor_quantizable(name)
            ):
                self.activation_names.append(name)

        # Capture intermediate activations and model outputs
        for node in (node for graph in graphs for node in graph.node):
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
        for graph in graphs:
            for node in graph.output:
                if node.name in self.activation_names:
                    node.name += "_updated"

    def _is_quantizable_dtype(self, name: str) -> bool:
        if name in self.activation_dtypes:
            np_dtype = self.activation_dtypes[name]
            if np_dtype not in data_types_to_quantize:
                return False
        else:
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

            if all(
                consumer.op_type == "MaskedSoftmax"
                and is_fused_supergroup(consumer)
                and name == consumer.input[2]
                for consumer in consumer_nodes
            ):
                # MaskedSoftmax's third input (mask_val) is a very large negative value
                # that we don't want to simulate quantization for.
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
        ir_model: onnx_ir.Model = onnx_ir.from_proto(self.model.model)
        ShapeInferencePass(strict_mode=True, data_prop=False).call(ir_model)
        value_map = onnx_ir.convenience.create_value_mapping(ir_model.graph)

        # ShapeInferencePass catches InferenceError internally, re-raise if failed
        if any(value.dtype is None for value in value_map.values()):
            raise onnx.shape_inference.InferenceError()

        activation_dtypes = {
            act_name: value.dtype.numpy() for act_name, value in value_map.items()
        }
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
        sess = OrtInferenceSession(
            self.model.model,
            ["CPUExecutionProvider"],
            session_options=self._ort_session_options,
            path=self._path,
            save_as_external_data=True,
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
        body_of = self._get_body_of_tensors()
        for name in self.param_names:
            self._insert_quantizer(
                name, is_param=True, graph=body_of.get(name, self.model.graph())
            )

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
            if op.transposed_params:
                return -2, -1
            return -1, -2

        return None, None

    def _insert_activation_quantization_nodes(self):
        """
        Insert quantization node for each activation tensor
        """
        body_of = self._get_body_of_tensors()
        for name in self.activation_names:
            self._insert_quantizer(
                name, is_param=False, graph=body_of.get(name, self.model.graph())
            )

    def _get_body_of_tensors(self) -> Dict[str, onnx.GraphProto]:
        """
        Map each tensor defined in a body (input, initializer or node output) to that body, so
        quantizers can be placed without searching all graphs per quantizer.
        """
        body_of = {}
        for graph in self._graphs_to_quantize()[1:]:
            body_of.update({inp.name: graph for inp in graph.input})
            body_of.update({init.name: graph for init in graph.initializer})
            body_of.update({name: graph for node in graph.node for name in node.output})

        return body_of

    def _insert_quantizer(
        self, input_name: str, is_param: bool, graph: Optional[onnx.GraphProto] = None
    ):
        """
        Inserts a quantizer for tensor `input_name` in the graph and adds it to `self.qc_quantize_op_dict`

        self.session must be rebuilt after calling this for changes to take effect.

        :param input_name: Name of the tensor to quantize
        :param is_param: True if the tensor is a parameter, False if it is an activation
        :param graph: Graph which defines `input_name`. Searched for if not given.
        """
        if input_name in self.qc_quantize_op_dict:
            raise RuntimeError(f"Quantizer already exists for tensor {input_name}")

        # TODO: Revisit all tensor/node naming
        node_name = "QcQuantizeOp_" + input_name
        if is_param:
            output_name = input_name + "_qdq"
            op_mode = OpMode.oneShotQuantizeDequantize
            precision = self._param_type
            tensor_quantizer_params = self._create_tensor_quantizer_params(input_name)
        else:
            output_name = input_name + "_updated"
            op_mode = OpMode.updateStats
            precision = self._activation_type
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
        # Add the quantizer to the graph which defines the tensor
        if graph is None:
            graph = next(
                (
                    graph
                    for graph in self._graphs_to_quantize()[1:]
                    if any(input_name in node.output for node in graph.node)
                    or any(input_name == init.name for init in graph.initializer)
                    or any(input_name == inp.name for inp in graph.input)
                ),
                self.model.graph(),
            )
        graph.node.append(custom_node)
        self.qc_quantize_op_dict[input_name] = QcQuantizeOp(
            quant_info=quant_info,
            quant_scheme=self._quant_scheme,
            op_mode=op_mode,
            tensor_quantizer_params=tensor_quantizer_params,
        )
        self.qc_quantize_op_dict[input_name].set_precision(precision)

    @staticmethod
    @deprecated("Use `aimet_onnx.utils.OrtInferenceSession` instead")
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

            if op.type in _NORM_OP_TYPES:
                output_bw = None
                output_symmetry = False
                if (
                    output_quantizers
                    and output_quantizers[0]
                    and output_quantizers[0].enabled
                ):
                    output_bw = output_quantizers[0].bitwidth
                    output_symmetry = output_quantizers[0].use_symmetric_encodings

                for param_quantizer in param_quantizers.values():
                    if not param_quantizer.enabled:
                        continue
                    if param_quantizer.data_type != QuantizationDataType.int:
                        continue
                    param_quantizer.set_bitwidth(max(param_quantizer.bitwidth, 8))
                    if op.type in (
                        "GroupNormalization",
                        "InstanceNormalization",
                        "RMSNormalization",
                    ):
                        param_quantizer.use_symmetric_encodings = output_symmetry
                        if output_bw:
                            param_quantizer.set_bitwidth(output_bw)
                    elif op.type in ("LayerNormalization", "BatchNormalization"):
                        # Only 8-bit weight kernels are supported before V73
                        if self._hw_version in {"V66", "V68", "V69"}:
                            param_quantizer.set_bitwidth(8)
                        param_quantizer.use_symmetric_encodings = (
                            param_quantizer.bitwidth >= 16
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
                # Exception rules do not apply for float matmuls
                if (
                    target_quantizer_for_first_input is None
                    or target_quantizer_for_second_input is None
                ):
                    continue
                if QuantizationDataType.float in (
                    target_quantizer_for_first_input.data_type,
                    target_quantizer_for_second_input.data_type,
                ):
                    continue
                if self._hw_version in {"V66", "V68", "V69"}:
                    target_quantizer_for_second_input.use_symmetric_encodings = True
                    target_quantizer_for_second_input.set_precision(int8)
                elif target_quantizer_for_second_input.precision() == int16:
                    target_quantizer_for_second_input.use_symmetric_encodings = True
                    target_quantizer_for_first_input.set_precision(int16)

            elif op.type == "ConvTranspose":
                groups = utils.get_node_attribute(op.get_module(), "group")
                weight_qtzr = param_quantizers.get("weight", None)
                # TODO: Fix handling of PCQ for grouped ConvTranspose
                if groups not in (None, 1) and weight_qtzr:
                    weight_qtzr.enable_per_channel_quantization(False)

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
        if not self._path:
            raise ValueError("Path not specified to save the model.")

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
            if not encoding:
                continue
            encoding_dict[name] = encoding

        if version.parse(enc_version) < version.parse("1.0.0"):
            return encoding_dict

        for name, encoding in encoding_dict.items():
            encoding["name"] = name
        return list(encoding_dict.values())

    def _export_encodings(
        self,
        encoding_file_path,
        encoding_version: str,
        force_activation_as: str | None = "unsigned",
    ):
        """
        Export encodings to json file

        :param encoding_file_path: path to save the encoding file
        """
        if encoding_version not in VALID_ENCODING_VERSIONS:
            raise NotImplementedError(
                f"Encoding version {encoding_version} not in set of valid encoding "
                f"versions {VALID_ENCODING_VERSIONS}."
            )

        encodings_dict = {
            "producer": {
                "package": "aimet-onnx",
                "version": aimet_onnx.__version__,
            },
            "version": encoding_version,
        }

        if encoding_version >= "2.0.0":
            encodings = self._get_encodings(
                self.qc_quantize_op_dict.keys(), encoding_version
            )

            if force_activation_as is not None:
                param_names = set(self.param_names)
                encodings = [
                    enc
                    if enc["name"] in param_names
                    else _to_unsigned_encoding(enc)
                    if force_activation_as == "unsigned"
                    else _to_signed_encoding(enc)
                    for enc in encodings
                ]

            if self._export_data_movement_op_output_quantizers:
                with self._remove_quantization_nodes():
                    derived_encodings = _derive_data_movement_op_encodings(
                        self.model.model,
                        {enc["name"]: enc for enc in encodings},
                    )

                encodings.extend(
                    {**enc, "name": name} for name, enc in derived_encodings.items()
                )

            encodings_dict.update(
                {
                    "encodings": encodings,
                }
            )
        else:
            param_encodings = self._get_encodings(self.param_names, encoding_version)
            activation_encodings = self._get_encodings(
                self.activation_names, encoding_version
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
            self.remove_quantizers(self.model)
            yield

        finally:
            self.model.model.graph.ClearField("node")
            self.model.model.graph.node.extend(sim_nodes)
            for output, name in zip(self.model.graph().output, sim_outputs):
                output.name = name

    @classmethod
    def remove_quantizers(cls, model: Union[ONNXModel, ModelProto]):
        """
        Removes all QcQuantizeOp layers from model
        """
        if isinstance(model, ONNXModel):
            model = model.model

        all_quantizers = set(
            node.name for node in model.graph.node if node.op_type == "QcQuantizeOp"
        )
        return cls._remove_quantizers(model, all_quantizers)

    @classmethod
    def _remove_quantizers(
        cls, model: ModelProto, to_be_removed: Iterable[str]
    ) -> ModelProto:
        to_be_removed = set(to_be_removed)

        for node in model.graph.node:
            if node.name in to_be_removed and node.op_type != "QcQuantizeOp":
                raise RuntimeError(
                    f"Node {node.name} is not a QcQuantizeOp, cannot be removed."
                )

        to_remain = [
            node for node in model.graph.node if node.name not in to_be_removed
        ]
        tensor_name_map = {
            node.output[0]: node.input[0]
            for node in model.graph.node
            if node.name in to_be_removed
        }

        model.graph.ClearField("node")
        model.graph.node.extend(to_remain)

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

    def _adjust_weight_scales_against_overflow(self):
        """
        Given
          y = round((xW + b) * sx * sw / sy - zy)

        HTP implements this equation as either eq 1 or eq 2, depending on hw version and other parameters
          y = round(( xW + b                ) * sx * sw / sy - zy)
            = round(( xW + b                ) *     s'       - zy)  ... eq 1
            = round(( xW + b -       zy/s'  ) *     s'           )
            ≈ round(( xW + b - round(zy/s') ) *     s'           )
            = round(( xW + b'               ) *     s'           )  ... eq 2

        where:

        | name |        description        |        dtype         |          equation          |
        |------|---------------------------|----------------------|----------------------------|
        |  x   | input                     | uint8 or uint16      | round(x_float / sx + zx)   |
        |  W   | weight                    | int4, int8, or int16 | round(W_float / sw)        |
        |  b   | bias                      | int32                | round(b_float / (sx * sw)) |
        |  y   | output                    | dtype(x)             |        given above         |
        |  sx  | input scale               | float                |             -              |
        |  zx  | input zero_point          | dtype(x)             |             -              |
        |  sw  | weight scale              | float                |             -              |
        |  sy  | output scale              | float                |             -              |
        |  zy  | output zero_point         | dtype(y)             |             -              |
        |  s'  | requantization scale      | float                |        sx * sw / sy        |
        |  b'  | combined accumulator bias | int32                |      b - round(zy / s')    |


        This function adjusts the weight scale to prevent 3 possible scenarios of overflow/underflow

        1. Bias Overflow
           - occurs if:   |b| > 2**31
           - bad because: Causes severe clipping error when exported as int32

        2. Requantization Scale Underflow
           - occurs if:   s' <= 2**-24
           - bad because: If exponent e <= -24, HexNN misinterprets s'=2**e as 2**(e+32)
                          due to internal type casting bug

        3. Accumulator Bias Overflow
           - occurs if:   |b'| > 2**31
           - bad because: HexNN internally stores b' as int32

        4. Export-dtype Bias-scale Underflow
           - occurs if:   sx * sw < finfo(export_dtype).tiny (fp16 export only)
           - bad because: The bias's y_scale in the exported QDQ graph carries
                          the surrounding activation dtype; on fp16 it collapses
                          to zero, breaking ``bias_scale = sx * sw`` fusion and
                          causing divide-by-zero at export time.

        Stage 4 gates on both (a) the bias's export dtype and (b) whether any
        channel actually underflows, so fp32 export and well-scaled fp16 export
        remain bit-identical to today.
        """
        # pylint: disable=redefined-builtin, protected-access

        matmul_ops = {
            op: self._get_weight_and_bias(op)
            for op in self.connected_graph.get_all_ops().values()
            if op.type
            in (
                "Conv",
                "Gemm",
                "MatMul",
                "ConvTranspose",
            )
        }

        for op, (weight, bias) in matmul_ops.items():
            # TODO(hitameht): weight being None indicates that something went wrong during
            # onnx graph parsing. Need to investigate further.
            if weight is None:
                continue

            input, *_ = op.inputs
            input_qtzr = self._get_enabled_quantizer(input.name)

            if not (
                input_qtzr
                and input_qtzr.enabled
                and input_qtzr.data_type == QuantizationDataType.int
                and input_qtzr.is_initialized()
            ):
                continue

            (output,) = op.outputs
            output_qtzr = self._get_enabled_quantizer(output.name)

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

            if bias:
                bias_proto = self.model.get_initializer(bias.name)

                if not bias_proto:
                    bias_proto = next(
                        (
                            attr.t
                            for node in self.model.graph().node
                            if bias.name in node.output
                            for attr in node.attribute
                            if attr.type == onnx.AttributeProto.TENSOR
                        ),
                        None,
                    )
            else:
                bias_proto = None

            weight_scale = weight_qtzr._get_scale()
            input_scale = input_qtzr._get_scale()

            if weight_scale is None or input_scale is None:
                continue

            bias_float = (
                to_array(bias_proto) if bias_proto else np.zeros_like(weight_scale)
            )

            encodings = weight_qtzr.get_encodings()
            if encodings is None:
                continue

            # Prevent bias overflow (1)
            # Use slightly discounted num_steps to account for floating point precision error
            adjusted_weight_scale = _adjust_weight_scale_against_bias_overflow(
                bias_float, input_scale, weight_scale, num_steps=2**31 - 2**15
            )

            if (
                output_qtzr
                and output_qtzr.enabled
                and output_qtzr.data_type == QuantizationDataType.int
                and output_qtzr.is_initialized()
            ):
                output_scale = output_qtzr._get_scale()
                output_offset = output_qtzr._get_offset()

                # Prevent requantization scale underflow (2)
                adjusted_weight_scale = _adjust_weight_scale_against_scale_underflow(
                    input_scale, adjusted_weight_scale, output_scale
                )

                # Prevent accumulator bias overflow (3)
                bias_scale = input_scale * adjusted_weight_scale
                accumulator_bias = np.round(bias_float / bias_scale) + np.round(
                    output_offset * output_scale / bias_scale
                )
                # Use slightly discounted num_steps to account for floating point precision error
                adjusted_weight_scale = _adjust_weight_scale_against_bias_overflow(
                    accumulator_bias * bias_scale,
                    input_scale,
                    adjusted_weight_scale,
                    num_steps=2**31 - 2**15,
                )

            # Prevent export-dtype bias-scale underflow (4)
            # No-op unless the bias tensor exports at a dtype whose `tiny` is
            # larger than _INT32_MINIMUM_SCALE (currently only fp16) AND some
            # channel of input_scale * weight_scale falls below that floor.
            # If input_scale itself underflows the floor, no weight bump can
            # save it — leave the scales untouched here and let
            # ``_concretize_int32_bias_quantizers`` raise if the caller later
            # asks for int32-bias export.
            if bias is not None:
                bias_floor = self._get_bias_scale_floor(bias.name)
                if (
                    bias_floor > _INT32_MINIMUM_SCALE
                    and not np.any(input_scale < bias_floor)
                    and np.any(input_scale * adjusted_weight_scale < bias_floor)
                ):
                    adjusted_weight_scale = (
                        _adjust_weight_scale_against_export_dtype_underflow(
                            input_scale, adjusted_weight_scale, bias_floor
                        )
                    )

            offset = np.array([enc.offset for enc in encodings], dtype=np.float32)
            adjusted_min, adjusted_max = compute_min_max_given_delta_offset(
                adjusted_weight_scale,
                offset,
                weight_qtzr.bitwidth,
                weight_qtzr.use_symmetric_encodings,
                weight_qtzr.use_strict_symmetric,
            )

            if isinstance(adjusted_weight_scale, float):
                adjusted_weight_scale = np.array([adjusted_weight_scale])

            adjusted_weight_scale = adjusted_weight_scale.flatten()

            assert len(adjusted_weight_scale) == len(encodings), (
                "Weight scale adjustment only supported for per-tensor and per-channel scales."
            )
            for new_scale, new_min, new_max, enc in zip(
                adjusted_weight_scale, adjusted_min, adjusted_max, encodings
            ):
                enc.min, enc.max, enc.delta = new_min, new_max, new_scale
            if weight_qtzr.is_encoding_frozen():
                if not np.array_equal(weight_scale.flatten(), adjusted_weight_scale):
                    logger.warning(
                        "Bias/scale overflow-underflow expected for %s but could not adjust "
                        "scale for frozen weight quantizer.",
                        op.name,
                    )
                continue
            weight_qtzr.load_encodings(encodings)
            logger.info(
                "Adjusted weight scale for %s to prevent bias/scale overflow-underflow.",
                op.name,
            )

    def _get_bias_scale_floor(self, bias_name: str) -> float:
        """
        Return the minimum representable bias scale for `bias_name`.

        The int32 bias scale is exported as `QuantizeLinear.y_scale` /
        `DequantizeLinear.x_scale`, whose ONNX-required dtype matches the
        surrounding activation dtype.
        If that dtype is fp16, values like `_INT32_MINIMUM_SCALE` (~2.33e-12) underflow to zero on cast,
        producing divide-by-zero during export.
        Floor at the greater of the int32 precision floor and the smallest normal of the export dtype so
        the scale round-trips through the exported graph without loss.

        TODO: Drop the dtype floor once we can always export scales as fp32.
        ONNX opset 23+ decouples Q/DQ scale dtype from tensor dtype
        (``y_scale`` can be fp32 even for fp16 tensors), but as of ORT
        1.23.2 the CPUExecutionProvider has no Q/DQ opset-23 kernels, so
        emitting that pattern breaks session load. When ORT ships opset-23
        Q/DQ kernels and the HTP converter accepts them, switch export to
        force fp32 scales unconditionally and remove this floor.
        """
        export_dtype = self.activation_dtypes.get(bias_name, np.float32)
        if np.issubdtype(export_dtype, np.floating):
            dtype_floor = float(np.finfo(export_dtype).tiny)
        else:
            dtype_floor = 0.0
        return max(_INT32_MINIMUM_SCALE, dtype_floor)

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

        bias_float = to_array(bias_proto).astype(np.float64, copy=False)
        bias_scale = np.maximum(
            abs(bias_float) / 2**31, self._get_bias_scale_floor(bias.name)
        )

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

        if weight is None:
            weight_qtzr = self._get_enabled_quantizer(op.inputs[WEIGHT_INDEX].name)
        else:
            weight_qtzr = self.qc_quantize_op_dict.get(weight.name)
        input_qtzr = self._get_enabled_quantizer(input.name)

        if not (
            input_qtzr
            and input_qtzr.enabled
            and input_qtzr.is_initialized()
            and weight_qtzr
            and weight_qtzr.enabled
            and weight_qtzr.is_initialized()
        ):
            return self._get_statistical_bias_scale(op)

        if len(bias.consumers) > 1:
            raise RuntimeError(
                f"Cannot determine single analytical bias scale for bias tensor {bias} with "
                f"multiple uses ({bias.consumers}). Call "
                "``aimet_onnx.utils.duplicate_shared_initializers(onnx_model.graph)`` before "
                "instantiating QuantizationSimModel to resolve the conflicts"
            )

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

        if weight_qtzr._encoding_type() == EncodingType.LPBQ:
            # NOTE: In LPBQ, bias encodings should be derived from per-channel weight scale
            scale_encoding = weight_qtzr._scale_encoding_dict()
            weight_scale = scale_encoding["x_scale"] if scale_encoding else None
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
            "LSTM": self._get_statistical_bias_scale,
            "GRU": self._get_statistical_bias_scale,
            "RNN": self._get_statistical_bias_scale,
        }

        ops_with_bias = {
            op: self._get_weight_and_bias(op)
            for op in self.connected_graph.get_all_ops().values()
            if op.type in switcher
        }

        original_bias_encodings: dict[str, tuple[bool, EncodingBase]] = {}
        for _, (_, bias) in ops_with_bias.items():
            if bias and bias.name in self.qc_quantize_op_dict:
                bias_qtzr = self.qc_quantize_op_dict[bias.name]
                original_bias_encodings[bias.name] = (
                    bias_qtzr.enabled,
                    EncodingBase.from_quantizer(bias_qtzr),
                )

        def cleanup():
            for name, (enabled, encoding) in original_bias_encodings.items():
                bias_qtzr = self.qc_quantize_op_dict[name]
                bias_qtzr.enabled = enabled
                if encoding:
                    encoding.load_to(bias_qtzr)
                else:
                    bias_qtzr.reset_encoding_stats()

        try:
            for op, (weight, bias) in ops_with_bias.items():
                if bias is None:
                    continue

                if bias.name not in self.qc_quantize_op_dict:
                    continue

                bias_qtzr = self.qc_quantize_op_dict[bias.name]

                if weight is not None:
                    weight_qtzr = self.qc_quantize_op_dict[weight.name]
                elif op.type in ("Conv", "ConvTranspose", "Gemm", "MatMul"):
                    weight_qtzr = self._get_enabled_quantizer(
                        op.inputs[WEIGHT_INDEX].name
                    )
                else:
                    weight_qtzr = None

                encoding_type = (
                    weight_qtzr._encoding_type().name
                    if weight_qtzr
                    else EncodingType.PER_TENSOR.name
                )

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

                input_qtzr = self._get_enabled_quantizer(op.inputs[0].name)
                if not (
                    input_qtzr
                    and input_qtzr.enabled
                    and input_qtzr.data_type == QuantizationDataType.int
                    and input_qtzr.is_initialized()
                ):
                    # Input quantizer wasn't created, enabled, or initialized.
                    # Since input_scale isn't available, exclude bias from quantization.
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

                if weight_qtzr is None:
                    # Edge case: Op has no weight quantizer. Fall back to statistical bias scale
                    get_bias_scale = self._get_statistical_bias_scale
                    is_analytic = False
                else:
                    get_bias_scale = switcher.get(
                        op.type, self._get_statistical_bias_scale
                    )
                    is_analytic = op.type in (
                        "Conv",
                        "Gemm",
                        "MatMul",
                        "ConvTranspose",
                    )

                # If the caller opted into int32-bias export and the bias
                # y_scale will be cast to fp16, an input_scale below fp16.tiny
                # means no weight_scale bump can produce a survivable
                # bias_scale. Raise instead of silently emitting divide-by-zero.
                if is_analytic:
                    bias_floor = self._get_bias_scale_floor(bias.name)
                    if bias_floor > _INT32_MINIMUM_SCALE:
                        input_scale = input_qtzr._get_scale()
                        if np.any(input_scale < bias_floor):
                            raise RuntimeError(
                                f"input_scale ({float(input_scale.min()):.3e}) for "
                                f"{op.name} is below the fp16 export-dtype floor "
                                f"({bias_floor:.3e}); cannot derive an analytic "
                                "bias_scale that round-trips through the export "
                                "dtype. Please recalibrate the input encoding or "
                                "increase activation precision."
                            )

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
            return Handle(cleanup)
        except:  # pylint disable=bare-except
            cleanup()
            raise

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

    @docstring(
        f"""
    Compute encodings and export to files

    Args:
        path: dir to save encoding files
        filename_prefix: filename to save encoding files
        export_model (bool, optional):
            If True, then ONNX model is exported. When False, only encodings are exported.
        export_int32_bias (bool, optional):
            If true, generate and export int32 bias encoding on the fly.
            Default: `True` if encoding version is 2.0.0 or higher, otherwise `False`.
        encoding_version (str, optional):
            Version of the encoding format to use. (default: {quantsim.encoding_version})
            Supported versions are: {sorted(list(quantsim.VALID_ENCODING_VERSIONS))}
        force_activation_as (str, optional):
            Force representing quantized activations as signed or unsigned integers.
            This argument is only applicable for encoding version 2.0.0.
            For versions 0.6.1 and 1.0.0, this argument is ignored. (default: `"unsigned"`)

    Example:

        >>> sim.export(path=".", filename_prefix="model", encoding_version="2.0.0")
    """
    )
    def export(
        self,
        path: str,
        filename_prefix: str,
        export_model: bool = True,
        *,
        export_int32_bias: Optional[bool] = None,
        encoding_version: Optional[str] = None,
        force_activation_as: Literal["unsigned"]
        | Literal["signed"]
        | None = "unsigned",
    ):
        self._raise_if_subgraphs_quantized("export()")
        encoding_version = encoding_version or quantsim.encoding_version

        if encoding_version not in quantsim.VALID_ENCODING_VERSIONS:
            raise ValueError(
                f"Unsupported encoding_version '{encoding_version}'. "
                f"Supported versions are: {quantsim.VALID_ENCODING_VERSIONS}"
            )

        if encoding_version == "0.6.1":
            msg = _red(
                "Encoding version 0.6.1 was deprecated in favor of 1.0.0 since aimet-onnx==2.1. "
                "If your code depends on parsing the exported encodings file, ensure that it is "
                "updated to be able to parse 1.0.0 format"
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        if export_int32_bias is None:
            export_int32_bias = version.parse(encoding_version) >= version.parse(
                "2.0.0"
            )

        with (
            self._concretize_int32_bias_quantizers()
            if export_int32_bias
            else contextlib.nullcontext()
        ):
            self._export_encodings(
                os.path.join(path, filename_prefix) + ".encodings",
                encoding_version,
                force_activation_as=force_activation_as,
            )

        if export_model:
            ir_model = onnx_ir.from_proto(self.model.model)
            ir_utils.remove_aimet_quantizers(ir_model)
            if any(is_fused_supergroup(node) for node in ir_model.graph.all_nodes()):
                inline_all_supergroups(ir_model)

            onnx_ir.save(
                ir_model,
                os.path.join(path, filename_prefix) + ".onnx",
                external_data=filename_prefix + ".data"
                if self._use_external_data
                else None,
            )

    def set_and_freeze_param_encodings(self, encoding_path: str):
        """
        Set and freeze parameter encodings from encodings JSON file

        :param encoding_path: path from where to load parameter encodings file
        """

        # Load encodings file
        with open(encoding_path) as json_file:
            encodings = json.load(json_file)

        encoding_version = None

        if isinstance(encodings, dict):
            encoding_version = encodings.get("version", None)

        if encoding_version is None:
            param_encodings = encodings

            if isinstance(param_encodings, dict):
                encoding_version = "0.6.1"
            elif not param_encodings:
                # trivial case: param_encodings is an empty list. default to 1.0.0
                encoding_version = "1.0.0"
            else:
                encoding_version = (
                    "2.0.0" if "y_scale" in param_encodings[0] else "1.0.0"
                )
        else:
            if encoding_version not in VALID_ENCODING_VERSIONS:
                raise NotImplementedError(
                    f"Encoding version should be one of {VALID_ENCODING_VERSIONS}; "
                    f"got {encoding_version}"
                )

            if encoding_version in ("0.6.1", "1.0.0"):
                param_encodings = encodings["param_encodings"]
            elif encoding_version == "2.0.0":
                # For version 2.0.0, filter only parameter encodings
                param_names_set = set(self.param_names)
                param_encodings = [
                    enc
                    for enc in encodings["encodings"]
                    if enc["name"] in param_names_set
                ]
            else:
                raise NotImplementedError(
                    f"Unsupported encoding version {encoding_version} in encodings file. "
                )

        if encoding_version in ("0.6.1", "1.0.0"):
            encodings = {
                "version": encoding_version,
                "param_encodings": param_encodings,
                "activation_encodings": {} if encoding_version == "0.6.1" else [],
            }
        else:
            encodings = {
                "version": encoding_version,
                "encodings": param_encodings,
            }

        load_encodings_to_sim(
            self,
            encodings,
            strict=False,
            allow_overwrite=False,
            disable_missing_quantizers=False,
        )

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
        self._sort_quantizer_nodes()
        self.session = None
        self.session = OrtInferenceSession(
            self.model.model,
            self.providers,
            session_options=self._ort_session_options,
            path=self._path,
            save_as_external_data=self._use_external_data,
        )

    def _raise_if_subgraphs_quantized(self, api: str):
        if any(
            node.op_type == "QcQuantizeOp"
            for graph in self._graphs_to_quantize()[1:]
            for node in graph.node
        ):
            raise NotImplementedError(
                f"{api} is not supported yet for models with quantizers inside Scan bodies"
            )

    def _sort_quantizer_nodes(self):
        """
        Moves every QcQuantizeOp right after the producer of its input, so that a topologically
        sorted source model stays sorted.

        NOTE: Sort graph.node in place.
        """
        for graph in self._graphs_to_quantize():
            quantizers = defaultdict(list)
            other_nodes = []
            for node in graph.node:
                if node.op_type == "QcQuantizeOp":
                    quantizers[node.input[0]].append(node)
                else:
                    other_nodes.append(node)

            # Quantizers of graph inputs and initializers come first
            produced = {name for node in other_nodes for name in node.output}
            order = [
                qtzr
                for name, qtzrs in quantizers.items()
                if name not in produced
                for qtzr in qtzrs
            ]
            for node in other_nodes:
                order.append(node)
                for name in node.output:
                    order.extend(quantizers.get(name, []))

            position = {id(node): i for i, node in enumerate(order)}
            graph.node.sort(key=lambda node: position[id(node)])  # pylint: disable=cell-var-from-loop

    def set_quantizers(self, quantizer_dict: Dict[str, QcQuantizeOp]):
        """
        Updates `self.qc_quantize_op_dict` with the entries in `quantizer_dict`

        :param quantizer_dict: Dictionary mapping tensor names to QcQuantizeOp objects
        """
        self._set_quantizers(quantizer_dict, rebuild_session=True)

    def _set_quantizers(
        self, quantizer_dict: Dict[str, QcQuantizeOp], rebuild_session: bool
    ):
        """
        Updates `self.qc_quantize_op_dict` with the entries in `quantizer_dict`

        :param quantizer_dict: Dictionary mapping tensor names to QcQuantizeOp objects
        """

        # Walk the graph and create a node input to op map, only for QcQuantizeOp nodes
        node_input_map = {}
        for graph in self._graphs_to_quantize():
            for node in graph.node:
                if node.op_type != "QcQuantizeOp":
                    continue
                for input_name in node.input:
                    node_input_map[input_name] = node

        for tensor, quantizer in quantizer_dict.items():
            self._set_quantizer(tensor, node_input_map, quantizer)

        if rebuild_session:
            self._rebuild_session()

    def _set_quantizer(
        self, tensor_name: str, node_input_map: Dict, quantizer: QcQuantizeOp
    ):
        """
        Places `quantizer` at `tensor_name` and updates the onnx graph.

        :param tensor_name: Name of the tensor at which to place the source quantizer
        :param quantizer: Quantizer to place at tensor_name
        """
        if not isinstance(quantizer, QcQuantizeOp):
            raise TypeError(
                f"Quantizer object {quantizer} is not of type {QcQuantizeOp.__qualname__}"
            )
        if (
            tensor_name not in self.qc_quantize_op_dict
            or tensor_name not in node_input_map
        ):
            raise ValueError(f"Tensor {tensor_name} is not an input to a quantize node")

        dst_onnx_node = node_input_map[tensor_name]

        self._set_quant_info(dst_onnx_node, quantizer)
        self.qc_quantize_op_dict[tensor_name] = quantizer

    def _set_quant_info(self, dst_onnx_node: onnx.NodeProto, src_qtzr: QcQuantizeOp):
        """
        Set quant_info attribute (pointer to the libquant_info object)

        :param dst_qtzr_tensor_name: destination quantizer node name in graph.
        :param src_qtzr: source quantizer.
        """

        for atr in dst_onnx_node.attribute:
            if atr.name == "quant_info":
                atr.i = libpymo.PtrToInt64(src_qtzr.quant_info)
                # Session is now invalid and must be rebuilt
                self.session = None
                return

    def _tie_quantizers_for_op_types(self, op_types_to_tie: List[str]):
        """
        Tie the input and output quantizers for given op types.

        :param op_types_to_tie: List of onnx ops for which to tie quantizers
        """
        op_types_to_tie = set(op_types_to_tie)
        # Walk the graph and create a node input to op map, only for QcQuantizeOp nodes
        node_input_map = {}
        for graph in self._graphs_to_quantize():
            for node in graph.node:
                if node.op_type != "QcQuantizeOp":
                    continue
                for input_name in node.input:
                    node_input_map[input_name] = node

        op_types_to_propagate_backward = {
            op_type
            for op_type in op_types_to_tie
            if _is_grid_equivariant_op(op_type, include_unary=False)
        }
        op_types_to_propagate_forward = op_types_to_tie - op_types_to_propagate_backward

        self._propagate_output_encodings(op_types_to_propagate_backward, node_input_map)
        self._propagate_input_encodings(op_types_to_propagate_forward, node_input_map)

    def _propagate_output_encodings(
        self, op_types_to_tie: Set[str], node_input_map: Dict
    ):
        """
        Let input quantizers inherit output encodings

        :param op_types_to_tie: List of onnx ops for which to tie quantizers
        """
        # pylint: disable = protected-access
        if not op_types_to_tie:
            return
        qtzr_to_name = {qtzr: name for name, qtzr in self.qc_quantize_op_dict.items()}
        visited = set()

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
            src_min_max_ranges = set()
            stack = [op]

            while stack:
                op = stack.pop()

                if op in visited:
                    continue

                visited.add(op)

                for inp in op.inputs:
                    if not self._is_quantizable_dtype(inp.name):
                        continue
                    src_qtzr = self._get_enabled_quantizer(inp.name)
                    if src_qtzr:
                        src_name = qtzr_to_name[src_qtzr]
                        src_min_max_ranges.add(
                            src_qtzr._encoding_min_max_fixed_vals if src_qtzr else None
                        )
                    else:
                        src_name = inp.name

                    consumers = set(
                        self.connected_graph.get_product(src_name).consumers
                    )
                    path_to_quantizer = self._get_path_to_effective_quantizer(inp.name)
                    if path_to_quantizer:
                        cg_path = [
                            self.connected_graph._ops[node.name]
                            for node in path_to_quantizer
                            if node.op_type != "QcQuantizeOp"
                        ]
                        all_consumers = consumers | set(
                            consumer for op in cg_path for consumer in op.output_ops
                        )
                        consumers = set(
                            consumer
                            for consumer in all_consumers
                            if consumer not in cg_path
                        )

                    if len(consumers) > 1:
                        continue

                    src_qtzrs[src_name] = src_qtzr

                    if inp.producer and (
                        inp.producer.type in op_types_to_tie
                        or _is_grid_preserving_op(
                            inp.producer.type, domain=inp.producer.domain
                        )
                    ):
                        stack.append(inp.producer)

            if len(src_min_max_ranges) == 1:
                # If all inputs are quantized and have the same fixed quantization range,
                # output quantizer will inherit the fixed range.
                # In practice, this will be most relevant when
                # Concat takes all its inputs from Softmax/Sigmoid.
                #            [0, 1]         [0, 1]
                #   Sigmoid ---Q1--> Concat --Q1-->
                #   Softmax ---Q1------^
                #            [0, 1]
                output_qtzr.set_fixed_encoding_range(src_min_max_ranges.pop())
            else:
                # If inputs have conflicting encoding constraints,
                # only tie the input quantizers without encoding constraints,
                # leaving the constrained ones untouched.
                #            [0, 1]         [X, Y]
                #   Sigmoid ---Q1--> Concat --Q2-->
                #      Conv ---Q2------^
                #            [X, Y]
                src_qtzrs = {
                    src_name: src_qtzr
                    for src_name, src_qtzr in src_qtzrs.items()
                    if not src_qtzr or not src_qtzr._encoding_min_max_fixed_vals  # pylint: disable=protected-access
                }

            for src_name, src_qtzr in src_qtzrs.items():
                if src_qtzr:
                    self._set_quantizer(src_name, node_input_map, output_qtzr)

    def _propagate_input_encodings(
        self, op_types_to_tie: Set[str], node_input_map: Dict
    ):
        """
        Let output quantizers inherit input encodings

        :param op_types_to_tie: List of onnx ops for which to tie quantizers
        """
        # pylint: disable=protected-access
        n_consumers: dict[str, int] = defaultdict(int)
        for graph in self._graphs_to_quantize():
            for node in graph.node:
                for inp in node.input:
                    n_consumers[inp] += 1

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

                # If output is not quantized (e.g., part of supergroup) do not propagate
                if not (output_qtzr and output_qtzr.enabled):
                    continue

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

                if (
                    output_qtzr._encoding_min_max_fixed_vals
                    and output_qtzr._encoding_min_max_fixed_vals
                    != input_qtzr._encoding_min_max_fixed_vals
                ):
                    path = self._get_path_to_effective_quantizer(
                        op.get_module().input[0]
                    )
                    if path and any(n_consumers[node.output[0]] != 1 for node in path):
                        # Input is consumed by multiple consumers when output range is constrained.
                        # For example:
                        #                           [0, ?]
                        #   Conv ---> Q1 -+-> Relu -> Q2
                        #                 +-> Add --> Q3
                        #
                        # In this case, we skip tying Q1 and Q2
                        # as it can lead to catastrophic accuracy drop.
                        continue

                input_qtzr._merge_constraints(output_qtzr)

                self._set_quantizer(out.name, node_input_map, input_qtzr)

    def to_onnx_qdq(
        self,
        *,
        export_int32_bias: bool = False,
        prequantize_constants: bool = False,
        force_activation_as: Literal["unsigned"]
        | Literal["signed"]
        | None = "unsigned",
    ) -> onnx.ModelProto:
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

        Args:
            export_int32_bias (bool, optional):
                If true, generate and export int32 bias encoding on the fly (default: `True`)
            prequantize_constants (bool, optional):
                If True, weights will be represented as quantized weight followed by DequantizeLinear nodes.
                If False, weights will be represented as float tensors followed by QuantizeLinear and DequantizeLinear nodes.
            force_activation_as (str, optional):
                Force representing quantized activations as signed or unsigned integers (default: `"unsigned"`)

        .. note::
            FP8 (``float8e4m3fn``/``float8e5m2``) quantizers are exported as
            onnx::QuantizeLinear/DequantizeLinear with a float8 ``y_zero_point`` of 0,
            which requires opset >= 19.  The opset is raised automatically when needed.

            Such a model must be run with an onnxruntime graph optimization level of
            at most ``ORT_ENABLE_BASIC``.  From ``ORT_ENABLE_EXTENDED`` upward,
            onnxruntime fuses QDQ pairs into integer kernels such as ``QLinearConv``,
            which do not accept float8 inputs and cause session creation to fail with
            "Type 'tensor(float8e4m3fn)' ... is invalid".

        .. image:: ../../images/conv_qdq.onnx.svg
            :align: center
        """
        self._raise_if_subgraphs_quantized("to_onnx_qdq()")
        with (
            self._concretize_int32_bias_quantizers()
            if export_int32_bias
            else contextlib.nullcontext()
        ):
            return self._to_onnx_qdq(
                prequantize_constants=prequantize_constants,
                force_activation_as=force_activation_as,
            )

    _export_data_movement_op_output_quantizers = True

    def _to_onnx_qdq(
        self, prequantize_constants: bool, force_activation_as: str | None
    ) -> onnx.ModelProto:
        lstm_int32_cell_states = [
            name
            for name, qtzr in self._lstm_cell_state_quantizers()
            if qtzr.enabled and qtzr.bitwidth == 32
        ]

        if lstm_int32_cell_states:
            raise RuntimeError(
                f"Detected int32 LSTM cell states: {lstm_int32_cell_states}. "
                f"{type(self).to_onnx_qdq.__qualname__} cannot export int32 encodings to ONNX QDQ format "
                "because onnx:: QuantizeLinaar doesn't support int32 data type. "
                f"Please try {type(self).export.__qualname__} instead."
            )

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

        if onnx_opset_version < 19 and any(
            qtzr.precision() in _QDQ_FLOAT_TYPES
            for qtzr in self.qc_quantize_op_dict.values()
        ):
            desired_onnx_opset_version = max(desired_onnx_opset_version, 19)
            logger.info(
                "onnx::QuantizeLinear and DequantizeLinear with FP8 are only supported in opset >= 19;"
                " got opset=%d",
                onnx_opset_version,
            )

        model_copy = onnx.ModelProto()
        model_copy.CopyFrom(self.model.model)

        # _get_qdq_parameters applies each param quantizer via quantize_dequantize, which
        # needs a valid encoding. Compute encodings for any param quantizers that aren't
        # initialized yet so we don't fold with garbage scales/offsets (or raise on export
        # before compute_encodings was called).
        self._compute_param_encodings(overwrite=False)

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
            "float_types": [],
        }

        param_names = {
            product.name
            for op in self.connected_graph.get_all_ops().values()
            for product, _ in op.parameters.values()
            if product.name in self.qc_quantize_op_dict
            and self.qc_quantize_op_dict[product.name].bitwidth <= 32
        }

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
                    encodings = (
                        _to_unsigned_encoding(encodings)
                        if force_activation_as == "unsigned"
                        else _to_signed_encoding(encodings)
                        if force_activation_as == "signed"
                        else encodings
                    )

                # Affine quantizer
                # Replace QcQuantizeOp with onnx::QuantizeLinear and DequantizeLinear
                qdq_node_info["input_names"].append(aimet_node.input[0])
                qdq_node_info["output_names"].append(aimet_node.output[0])
                qdq_node_info["node_name_prefixes"].append(aimet_node.name)
                qdq_node_info["encodings"].append(encodings)
                qdq_node_info["float_types"].append(
                    self.activation_dtypes[aimet_node.input[0]]
                )

        self.remove_quantizers(model_copy)

        if self._export_data_movement_op_output_quantizers:
            derived_encodings = _derive_data_movement_op_encodings(
                model_copy,
                dict(zip(qdq_node_info["input_names"], qdq_node_info["encodings"])),
            )
            # When exporting to ONNX QDQ, int32 encodings cannot be propagated
            # because ONNX only supports int32 DequantizeLinear but not QuantizeLinear.
            derived_encodings = {
                name: enc
                for name, enc in derived_encodings.items()
                if enc["output_dtype"] != "int32"
            }

            for name, encoding in derived_encodings.items():
                qdq_node_info["input_names"].append(name)
                qdq_node_info["output_names"].append(name + "_qdq")
                qdq_node_info["node_name_prefixes"].append(name)
                qdq_node_info["encodings"].append(encoding)
                qdq_node_info["float_types"].append(
                    self.activation_dtypes.get(name, np.float32)
                )

        # Note: Must inline supergroups before version conversion, since version_converter does not handle functions
        ir_model = onnx_ir.from_proto(model_copy)
        if any(is_fused_supergroup(node) for node in ir_model.graph.all_nodes()):
            inline_all_supergroups(ir_model)
            model_copy = onnx_ir.to_proto(ir_model)

        if onnx_opset_version < desired_onnx_opset_version:
            model_copy = _convert_version(model_copy, desired_onnx_opset_version)

        _add_onnx_qdq_nodes(
            model_copy,
            **qdq_node_info,
            onnx_opset=desired_onnx_opset_version,
            prequantize_constants=prequantize_constants,
        )

        ONNXModel(model_copy).topological_sort()

        # Add metadata property to indicate the model is exported by AIMET and its version
        prop = model_copy.metadata_props.add()
        prop.key = "producer"
        prop.value = f"aimet-onnx {aimet_onnx.__version__}"

        return model_copy

    def _get_qdq_parameters(self):
        """
        Compute the quantize-dequantize'd value of every foldable parameter.

        Each param quantizer is applied directly to its parameter's static value via the
        quantizer's own ``quantize_dequantize`` (the same libpymo kernel the in-graph
        ``QcQuantizeOp`` runs), one parameter at a time. This avoids materializing every
        parameter into a single throwaway ONNX graph, which would overflow protobuf's 2GB
        message limit for large models.

        Only enabled, sub-int32 quantizers that still have a ``QcQuantizeOp`` node in the
        graph are folded:

        * int32 (and wider) quantizers are skipped, since QcQuantizeOp simulates
          quantization in float32, which is insufficient for int32.
        * params whose QcQuantizeOp node was already removed (e.g. a prior fold) are
          skipped, which keeps re-folding a no-op.

        :return: Mapping of parameter name to its quantize-dequantize'd value.
        """
        # Names of parameters (weights/biases) tracked by the connected graph whose
        # quantizers are sub-int32.
        param_names = {
            product.name
            for op in self.connected_graph.get_all_ops().values()
            for product, _ in op.parameters.values()
            if product.name in self.qc_quantize_op_dict
            and self.qc_quantize_op_dict[product.name].bitwidth <= 32
        }

        # Map every static value (initializer / Constant node) to its TensorProto once, so
        # the per-parameter lookups below are O(1). Scanning the whole graph per parameter
        # would be quadratic and dominate export time on large models.
        constants = _get_all_constants(self.model.model)

        qdq_parameters = {}
        for node in self.model.model.graph.node:
            if node.op_type != "QcQuantizeOp" or node.input[0] not in param_names:
                continue

            param_name = node.input[0]
            quantizer = self.qc_quantize_op_dict[param_name]

            # Skip disabled quantizers, and int32+ params (simulated in fp32, unreliable).
            if not quantizer.enabled or quantizer.bitwidth >= 32:
                continue

            param_value = to_array(constants[param_name])

            if (
                quantizer.data_type == QuantizationDataType.float
                and quantizer.precision() not in _QDQ_FLOAT_TYPES
            ):
                # Plain-cast floats have no encoding, so mirror QcQuantizeOp and round-trip through float16
                qdq_parameters[param_name] = param_value.astype(np.float16).astype(
                    param_value.dtype
                )
            else:
                qdq_parameters[param_name] = quantizer.quantize_dequantize(param_value)

        return qdq_parameters

    @staticmethod
    def _overwrite_parameters(
        model: onnx.ModelProto, parameters: Dict[str, np.ndarray]
    ):
        initializers = [
            (init, parameters.pop(init.name))
            for init in model.graph.initializer
            if init.name in parameters
        ]
        constants = [
            (node, parameters.pop(node.output[0]))
            for node in model.graph.node
            if node.op_type == "Constant" and node.output[0] in parameters
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

    def fold_param_quantizers(self):
        """
        Fold parameter quantizers into their associated parameters to accelerate inference.

        This bakes the quantize-dequantize operation of each (enabled, sub-int32) param
        quantizer directly into the parameter's initializer/constant, then removes the
        corresponding ``QcQuantizeOp`` nodes from the graph. As a result, the simulated
        quantized weights are computed once instead of on every inference.

        This is a terminal, inference-only optimization: the original floating point weights
        are overwritten in place and cannot be recovered. The folded quantizers' encodings
        are still emitted on :meth:`export`, but the exported ONNX model carries the
        pre-quantized weights and no longer contains QDQ nodes for those parameters.

        int32 (and any disabled or >16-bit) parameter quantizers are left untouched, since
        QcQuantizeOp simulates quantization in float32 which is insufficient for int32.

        Example:

            >>> sim = QuantizationSimModel(...)
            >>> sim.compute_encodings(...)
            >>> sum(node.op_type == "QcQuantizeOp" for node in sim.model.model.graph.node)
            42
            >>> sim.fold_param_quantizers()
            >>> sum(node.op_type == "QcQuantizeOp" for node in sim.model.model.graph.node)
            21
        """
        # Compute encodings for any param quantizers that aren't initialized yet so we don't
        # fold with garbage scales/offsets.
        self._compute_param_encodings(overwrite=False)

        # _get_qdq_parameters already skips int32/>32-bit and disabled param quantizers, and
        # only collects params that still have a QcQuantizeOp node, so re-folding is a no-op.
        qdq_parameters = self._get_qdq_parameters()

        folded_param_names = {
            name for name in qdq_parameters if name in self.qc_quantize_op_dict
        }

        if not folded_param_names:
            return

        # Bake the quantize-dequantize'd values into the live model's initializers/constants.
        self._overwrite_parameters(self.model.model, dict(qdq_parameters))

        # Physically remove the folded params' QcQuantizeOp nodes from the graph. This rewires
        # consumers from the "<param>_qdq" tensor back to the raw parameter tensor, so the
        # session no longer runs quantize-dequantize for these params during inference.
        nodes_to_remove = {
            node.name
            for node in self.model.model.graph.node
            if node.op_type == "QcQuantizeOp" and node.input[0] in folded_param_names
        }
        self._remove_quantizers(self.model.model, nodes_to_remove)

        # The quantizer objects are intentionally kept in qc_quantize_op_dict and param_names
        # (still enabled) so that their encodings are still emitted on export and so that
        # param-indexed code paths (e.g. int32 bias concretization) keep working. They simply
        # no longer have a node in the graph. Track them for introspection.
        for name in folded_param_names:
            self._folded_param_quantizers[name] = self.qc_quantize_op_dict[name]

        self._rebuild_session()

    def _get_enabled_quantizer(self, tensor_name: str) -> Optional[QcQuantizeOp]:
        """
        Returns closest enabled quantizer to tensor traversing upwards only through invariant ops

        :param tensor_name: Name of tensor for which to find quantizer
        """
        quantizer_name = self._get_enabled_quantizer_name(tensor_name)
        if quantizer_name:
            return self.qc_quantize_op_dict[quantizer_name]
        return None

    def _get_enabled_quantizer_name(self, tensor_name) -> Optional[str]:
        """
        Returns closest enabled quantizer to tensor traversing upwards only through invariant ops

        :param tensor_name: Name of tensor for which to find quantizer
        """
        if tensor_name not in self.connected_graph.get_all_products():
            if tensor_name.endswith(("_updated", "_qdq")):
                raise KeyError(
                    f"Could not find quantizer for tensor {tensor_name}. Input tensor_name must be the name of a tensor in the original (unquantized) graph"
                )
            else:
                raise KeyError(
                    f"Could not find quantizer for tensor {tensor_name}. Tensor name does not exist in the graph"
                )
        quantizer = self.qc_quantize_op_dict.get(tensor_name, None)
        if quantizer and quantizer.enabled:
            return tensor_name

        path = self._get_path_to_effective_quantizer(tensor_name)

        if path:
            *_, qc_quantize_op_node = path
            quantizer_name = qc_quantize_op_node.input[0]
            if self.qc_quantize_op_dict[quantizer_name].enabled:
                return quantizer_name

        return None

    def _get_path_to_effective_quantizer(
        self, tensor_name: str
    ) -> Optional[List[onnx.NodeProto]]:
        """
        Returns graph path from a tensor to the "effective" quantizer associated with it (if any).

        Node Q is the effective quantizer of tensor X if and only if:

            1. Q is an enabled QcQuantizeOp, and;
            2. Q(X) == X, and;
            3. Q is the closest ancestor of tensor X that satisfies 1 and 2


        For example, Q is the effective quantizer of X in all of the following examples:

            1. ... -> QcQuantizeOp ---> ...
                          (Q)      (X)

            2. ... -> QcQuantizeOp -> Transpose ---> ...
                          (Q)                   (X)

        On the other hand, Q is NOT the effective quantizer of X in any of the following examples:

            1. ... -> QcQuantizeOp -> QcQuantizeOp ---> ...
                          (Q)                      (X)

               (Reason: Q is not the closest ancestor of X that satisfies 1 and 2)

            2. ... -> QcQuantizeOp -> Conv ---> ...
                          (Q)              (X)

               (Reason: Q(X) != X)

        Args:
            tensor_name (str): Name of tensor for which to find the path to the effective quantizer

        Returns:
            List of tensor name and NodeProto in alternating order,
            indicating the path from the tensor to its effective quantizer.
            Returns None if the effective quantizer doesn't exist.

            For example, given the following graph:

             ... -> QcQuantizeOp -----> Transpose -----> ...
                        (Q)       (Y)      (T)     (X)

            `_get_path_to_effective_quantizer("X")` returns [T, Q].
        """
        producer = self._producers.get(tensor_name, None)
        path = [producer]

        def is_disabled_quantizer(node: onnx.NodeProto):
            return (
                node.op_type == "QcQuantizeOp"
                and not self.qc_quantize_op_dict[node.input[0]].enabled
            )

        def is_enabled_quantizer(node: onnx.NodeProto):
            return (
                node.op_type == "QcQuantizeOp"
                and self.qc_quantize_op_dict[node.input[0]].enabled
            )

        def is_float_to_float_cast(node: onnx.NodeProto):
            if node.op_type != "Cast":
                return False

            input_name = node.input[0]
            if self.activation_dtypes.get(input_name) not in data_types_to_quantize:
                # Outputs of QcQuantizeOps are always float dtype
                producer = self._producers.get(input_name)
                if not producer or producer.op_type != "QcQuantizeOp":
                    return False

            to_attr = utils.get_node_attribute(node, "to")
            if to_attr is None:
                return False

            output_type = onnx.helper.tensor_dtype_to_np_dtype(to_attr)
            return output_type in data_types_to_quantize

        while (
            producer
            and producer.input
            and (
                _is_grid_preserving_op(producer.op_type, domain=producer.domain)
                or is_disabled_quantizer(producer)
                or is_float_to_float_cast(producer)
            )
        ):
            tensor_name = producer.input[0]
            producer = self._producers.get(tensor_name, None)
            path += [producer]

        if producer and is_enabled_quantizer(producer):
            return [node for node in path if not is_disabled_quantizer(node)]

        return None

    def _tie_rnn_hidden_state_quantizers(self):
        """
        Tie the hidden state and cell state (if applicable) quantizers of RNN/GRU/LSTM

        * Y, Y_h, and initial_h share the same quantizer as they all represent "hidden state"
        * Y_c, initial_c share the same quantizer as they all represent "cell state" (only appliable for LSTM)

                X --> Q_x-+
                W --> Q_w-+
                R --> Q_r-+                   +--> Q_h --> Y
                B --------+--> RNN/GRU/LSTM --+--> Q_h --> Y_h
        initial_h --> Q_h-+                   +--> Q_c --> Y_c
        initial_c --> Q_c-+
        """
        to_be_replaced: Dict[QcQuantizeOp, QcQuantizeOp] = {}

        for node in self.model.model.graph.node:
            if node.op_type not in ("LSTM", "GRU", "RNN"):
                continue

            X = node.input[0]
            Y = node.output[0]
            Y_h = node.output[1] if len(node.output) >= 2 else None
            initial_h = node.input[5] if len(node.input) >= 6 else None

            path = self._get_path_to_effective_quantizer(X)
            if path:
                *_, qc_quantize_op_node = path
                X = qc_quantize_op_node.input[0]
                to_be_replaced.update(
                    {self.qc_quantize_op_dict[X]: self.qc_quantize_op_dict[Y]}
                )

            if Y_h:
                to_be_replaced.update(
                    {self.qc_quantize_op_dict[Y_h]: self.qc_quantize_op_dict[Y]}
                )

            if initial_h:
                path = self._get_path_to_effective_quantizer(initial_h)
                if path:
                    *_, qc_quantize_op_node = path
                    initial_h = qc_quantize_op_node.input[0]
                    to_be_replaced.update(
                        {
                            self.qc_quantize_op_dict[
                                initial_h
                            ]: self.qc_quantize_op_dict[Y]
                        }
                    )

            if node.op_type != "LSTM":
                continue

            Y_c = node.output[2] if len(node.output) >= 3 else None
            initial_c = node.input[6] if len(node.input) >= 7 else None

            if Y_c and initial_c:
                path = self._get_path_to_effective_quantizer(initial_c)
                if path:
                    *_, qc_quantize_op_node = path
                    initial_c = qc_quantize_op_node.input[0]
                    to_be_replaced.update(
                        {
                            self.qc_quantize_op_dict[
                                initial_c
                            ]: self.qc_quantize_op_dict[Y_c]
                        }
                    )

        # Replace all reference to the old quantizers with the new quantizers
        new_qc_quantize_op_dict = self.qc_quantize_op_dict.copy()

        for old_qtzr, new_qtzr in to_be_replaced.items():
            for name, qtzr in new_qc_quantize_op_dict.items():
                if qtzr == old_qtzr:
                    new_qc_quantize_op_dict[name] = new_qtzr

        self._set_quantizers(new_qc_quantize_op_dict, rebuild_session=False)

    def _lstm_cell_state_quantizers(self) -> Iterable[Tuple[str, QcQuantizeOp]]:
        for node in self.model.model.graph.node:
            if node.op_type != "LSTM":
                continue

            # Cell state of the last time stamp
            Y_c = node.output[2] if len(node.output) >= 3 else None

            # Initial cell state
            initial_c = node.input[6] if len(node.input) >= 7 else None

            if Y_c:
                Y_c_qtzr = self.qc_quantize_op_dict.get(Y_c)
                if Y_c_qtzr:
                    yield Y_c, Y_c_qtzr

            if initial_c:
                path = self._get_path_to_effective_quantizer(initial_c)
                if path:
                    *_, qc_quantize_op_node = path
                    initial_c = qc_quantize_op_node.input[0]
                else:
                    # No quantizer is enabled for initial_c. In this case,
                    # interpret the closest disabled quantizer (if any)
                    # as the initial_c quantizer.
                    producer = self._producers.get(initial_c)

                    while producer and _is_grid_preserving_op(
                        producer.op_type, domain=producer.domain
                    ):
                        producer = self._producers.get(producer.input[0])

                    if producer and producer.op_type == "QcQuantizeOp":
                        initial_c = producer.input[0]

                initial_c_qtzr = self.qc_quantize_op_dict.get(initial_c)
                if initial_c_qtzr:
                    yield initial_c, initial_c_qtzr

    def _disable_lstm_cell_state_quantizers(self):
        """
        Disable cell state quantizers of LSTM
        """
        for _, qtzr in self._lstm_cell_state_quantizers():
            qtzr.enabled = False

    def _concretize_int32_lstm_cell_state_quantizers(self, scale: float = 2**-20):
        """
        Create int32 cell state quantizers of LSTM
        By default, use scale = 2**-20 to match LPAI's requirement
        """
        for _, qtzr in self._lstm_cell_state_quantizers():
            if qtzr.data_type == QuantizationDataType.float:
                # Float16 quantizers are not exported to onnx QDQ graph
                continue

            if qtzr and qtzr.enabled and qtzr.is_initialized():
                # Edge case: LSTM cell state encoding already exists.
                # Always honor the existing bias encoding
                continue

            encoding = libpymo.TfEncoding()
            encoding.bw = 32
            encoding.delta = scale
            encoding.offset = -(2**31)
            encoding.min = scale * -(2**31)
            encoding.max = scale * (2**31 - 1)

            qtzr.enabled = True
            qtzr.use_symmetric_encodings = True
            qtzr.bitwidth = 32
            qtzr.enable_per_channel_quantization(False)
            qtzr.load_encodings([encoding])

    def set_tensor_precision(
        self, names: str | list[str], precision: qtype | str, *, strict: bool = True
    ):
        """
        Set the quantization precision for each tensor in names to the specified precision

        If quantization is not performed directly on the specified tensor (e.g., for outputs
        of data movement ops), propagates upwards to the closest enabled quantizer.

        Example:

            sim = QuantizationSimModel(...)
            sim.set_tensor_precision("/Conv_output_0", aimet_onnx.int16)

        Args:
            names: List of tensor names to set the precision for
            precision: Precision to quantize to. If string, must be a valid alias of a qtype
            strict: If True, throws an error if the tensor does not exist or is not quantized
        """
        if isinstance(names, str):
            names = [names]

        products = self.connected_graph.get_all_products()
        missing_names = set(names) - set(products.keys())

        if missing_names and strict:
            raise ValueError(f"No tensor found in graph with names: {missing_names}")

        quantizers = {
            name: self._get_enabled_quantizer(name)
            for name in names
            if name in products
        }
        missing_quantizers = set(
            name for name, qtzr in quantizers.items() if qtzr is None
        )
        if missing_quantizers and strict:
            raise ValueError(f"No quantizer exists for tensors: {missing_quantizers}")

        for quantizer in quantizers.values():
            if quantizer is not None:
                quantizer.set_precision(precision)


def _to_signed_encoding(encoding: dict) -> dict:
    return _to(encoding, signed=True)


def _to_unsigned_encoding(encoding: dict) -> dict:
    return _to(encoding, signed=False)


def _to(encoding: dict, signed: bool) -> dict:
    if ("output_dtype" not in encoding) or ("y_scale" not in encoding):
        raise RuntimeError(
            f"Expected 2.0.0 encoding format. Got unexpected keys: {list(encoding.keys())}"
        )

    encoding = encoding.copy()
    output_dtype = encoding["output_dtype"]

    if not output_dtype.startswith(("int", "uint")):
        return encoding  # floating point encoding
    if output_dtype.startswith("int") and signed:
        return encoding
    if output_dtype.startswith("uint") and not signed:
        return encoding

    if "y_zero_point" in encoding:
        y_zero_point = np.array(encoding["y_zero_point"], dtype=np.int64)
    else:
        y_scale = np.array(encoding["y_scale"])
        y_zero_point = np.zeros(y_scale.shape, dtype=np.int64)

    if signed:
        output_dtype = output_dtype[1:]
        bw = int(output_dtype[3:])
        y_zero_point -= 2 ** (bw - 1)
    else:
        output_dtype = f"u{output_dtype}"
        bw = int(output_dtype[4:])
        y_zero_point += 2 ** (bw - 1)

    # Update dtype from int to uint and shift zero_point accordingly
    encoding["output_dtype"] = output_dtype
    encoding["y_zero_point"] = y_zero_point.tolist()
    return encoding


def _remove_delegatable_excess_encodings(
    sim: QuantizationSimModel, encodings: Dict[str, dict]
):
    # pylint: disable=protected-access
    quantizable_tensor_names = set(
        name for name, qtzr in sim.qc_quantize_op_dict.items() if qtzr and qtzr.enabled
    )
    bias_names = set(
        bias.name
        for op in sim.connected_graph.get_all_ops().values()
        for _, bias in [sim._get_weight_and_bias(op)]
        if bias is not None
    )
    excess_encodings = encodings.keys() - (quantizable_tensor_names | bias_names)
    delegatable = set()

    for output_name in excess_encodings:
        output_product = sim.connected_graph.get_product(output_name)
        producer = output_product.producer if output_product is not None else None
        output_encoding = encodings[output_name]

        while producer and (
            _is_grid_preserving_op(producer.type, domain=producer.domain)
            or producer.type == "Cast"
        ):
            # Delegate excess encoding to producer's input
            #                                                      (excess encoding)
            #   input_name                                            output_name
            #       ↓                                                      ↓
            # ... ----> producer -----> [ 0 or more grid-preserving ops ] -->
            #      (grid-preserving)
            input_name = producer.inputs[0].name

            if (
                sim.qc_quantize_op_dict.get(input_name)
                and sim.qc_quantize_op_dict[input_name].enabled
                and (
                    (
                        input_name not in encodings
                        and len(sim.connected_graph.get_product(input_name).consumers)
                        == 1
                        and all(
                            encodings.get(other_output.name)
                            and EncodingBase.from_qnn_encoding_dict(
                                encodings[other_output.name]
                            )
                            == EncodingBase.from_qnn_encoding_dict(output_encoding)
                            for other_output in producer.outputs
                        )
                    )
                    or (
                        input_name in encodings
                        and EncodingBase.from_qnn_encoding_dict(encodings[input_name])
                        == EncodingBase.from_qnn_encoding_dict(output_encoding)
                    )
                )
            ):
                encodings[input_name] = {**output_encoding, "name": input_name}
                delegatable.add(output_name)

            producer = sim.connected_graph.get_product(input_name).producer

    for name in delegatable:
        encodings.pop(name)


def _flatten_0_6_1(encodings_dict: dict) -> Tuple[Dict[str, Any], Set[str]]:
    """0.6.1 keys its sections by tensor name. Normalized to 1.0.0, AIMET's own path."""
    param_encodings = {
        name: EncodingBase.from_qnn_encoding_dict(e).to_qnn_encoding_dict("1.0.0")
        for name, e in encodings_dict["param_encodings"].items()
    }
    activation_encodings = {
        name: EncodingBase.from_qnn_encoding_dict(e).to_qnn_encoding_dict("1.0.0")
        for name, e in encodings_dict["activation_encodings"].items()
    }
    return param_encodings | activation_encodings, set(param_encodings)


def _flatten_1_0_0(encodings_dict: dict) -> Tuple[Dict[str, Any], Set[str]]:
    """1.0.0 sections are lists whose entries carry a ``"name"`` field."""
    param_encodings = {e["name"]: e for e in encodings_dict["param_encodings"]}
    activation_encodings = {
        e["name"]: e for e in encodings_dict["activation_encodings"]
    }
    return param_encodings | activation_encodings, set(param_encodings)


def _flatten_2_x(encodings_dict: dict) -> Tuple[Dict[str, Any], Set[str]]:
    """
    2.x is a single flat ``"encodings"`` list, left as-is because it is the only layout that
    stores a per-channel axis. It carries no param/activation split, so parameters are
    classified from the graph instead.
    """
    return {e["name"]: e for e in encodings_dict["encodings"]}, set()


# Maps an encoding version to the loader for its file layout. Versions that share a layout
# share a loader, so supporting a newly added AIMET encoding version is usually one entry
# here plus a test. test_encodings_to_onnx_qdq_covers_all_encoding_versions fails when a
# version in VALID_ENCODING_VERSIONS is missing from this table.
_ENCODING_LOADERS = {
    "0.6.1": _flatten_0_6_1,
    "1.0.0": _flatten_1_0_0,
    "2.0.0": _flatten_2_x,
    "2.1.0": _flatten_2_x,
}

# The sections each layout stores its encodings in, and the type each must have.
_ENCODING_SECTIONS = {
    "0.6.1": {"activation_encodings": dict, "param_encodings": dict},
    "1.0.0": {"activation_encodings": list, "param_encodings": list},
    "2.0.0": {"encodings": list},
    "2.1.0": {"encodings": list},
}

# Fields every entry needs: what EncodingBase.get_subclass reads to pick a parser. Scale fields
# are checked separately, per encoding type, because float16/bfloat16 are plain casts with no
# scale at all and LPBQ spells its scale differently -- requiring one here would reject them.
_REQUIRED_ENCODING_FIELDS = {
    "0.6.1": ("bitwidth", "dtype"),
    "1.0.0": ("bw", "dtype", "enc_type"),
    "2.0.0": ("output_dtype",),
    "2.1.0": ("output_dtype",),
}

# Fields only an integer encoding needs, read unconditionally by the affine parsers.
_REQUIRED_AFFINE_FIELDS = {
    "0.6.1": ("is_symmetric", "offset"),
    "1.0.0": ("is_sym", "offset"),
    "2.0.0": (),
    "2.1.0": (),
}

# Derived from the enum so a new member cannot fall out of sync.
_VALID_ENC_TYPES = {member.name for member in EncodingType}
_BLOCK_ENC_TYPES = {EncodingType.PER_BLOCK.name, EncodingType.LPBQ.name}

# Any one of these counts as "this entry carries a scale".
_SCALE_FIELDS = {
    "0.6.1": ("scale",),
    "1.0.0": ("scale",),
    "2.0.0": ("y_scale", "per_channel_float_scale", "per_block_int_scale"),
    "2.1.0": ("y_scale", "per_channel_float_scale", "per_block_int_scale"),
}

# Where each version states its dtype, and the substring marking an integer one. Mirrors the
# dtype checks in EncodingBase.get_subclass.
_INT_DTYPE_FIELD = {
    "0.6.1": ("dtype", "int"),
    "1.0.0": ("dtype", "INT"),
    "2.0.0": ("output_dtype", "int"),
    "2.1.0": ("output_dtype", "int"),
}

# Fields that must NOT appear, because EncodingBase._infer_encoding_version re-derives each
# entry's version from its own shape ("bw" present means 1.0.0, absent means 2.x) and ignores the
# file's declared version. An entry carrying another layout's marker is otherwise parsed as that
# version, silently, until some unrelated field turns up missing.
_FOREIGN_ENCODING_FIELDS = {
    "0.6.1": (),
    "1.0.0": (),
    "2.0.0": ("bw",),
    "2.1.0": ("bw",),
}


def _integer_dtype_range(dtype: str) -> Optional[Tuple[int, int]]:
    """
    The representable range of an integer dtype name such as ``"uint8"`` or ``"int4"``.

    :return: (min, max), or None if the name is not an integer dtype this can size
    """
    match = re.fullmatch(r"(u?)int(\d+)", dtype)
    if not match:
        return None

    unsigned, bits = match.group(1) == "u", int(match.group(2))
    if unsigned:
        return 0, 2**bits - 1
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def _validate_zero_point(element: dict, name: str, dtype: str, encoding_version: str):
    """
    Check that zero points fit the quantized dtype.

    A zero point outside the dtype's range wraps when written as an ONNX initializer, shifting
    the whole grid: the graph still validates and runs, and the output is simply wrong.

    :param element: One encoding entry
    :param name: The tensor the entry belongs to, for the error message
    :param dtype: The entry's integer dtype, e.g. ``"uint8"`` (2.x) or ``"INT"`` (1.0.0)
    :param encoding_version: The file's declared ``"version"``
    :raises ValueError: If a zero point does not fit the dtype
    """
    if encoding_version in ("0.6.1", "1.0.0"):
        # Legacy states an additive offset and a bitwidth rather than a named dtype.
        bitwidth = element.get("bitwidth" if encoding_version == "0.6.1" else "bw")
        if not isinstance(bitwidth, int) or isinstance(bitwidth, bool):
            raise ValueError(
                f"Encoding for '{name}' has a non-integer bitwidth ({bitwidth!r})."
            )
        if not 1 <= bitwidth <= 32:
            raise ValueError(
                f"Encoding for '{name}' has an out-of-range bitwidth ({bitwidth}); expected "
                f"1 to 32."
            )
        low, high = -(2**bitwidth), 2**bitwidth - 1
        values, field = element.get("offset"), "offset"
    else:
        bounds = _integer_dtype_range(dtype)
        if bounds is None:
            # The exporter reports unrecognized dtype names against the target opset.
            return
        low, high = bounds
        values, field = element.get("y_zero_point"), "y_zero_point"

    if values is None:
        return

    for scalar in np.asarray(values, dtype=object).reshape(-1):
        # AIMET writes legacy offsets as floats (e.g. -128.0), so accept any integral number.
        if (
            not isinstance(scalar, (int, float))
            or isinstance(scalar, bool)
            or not float(scalar).is_integer()
        ):
            raise ValueError(
                f"Encoding for '{name}' has a non-integer '{field}' ({scalar!r})."
            )
        if not low <= scalar <= high:
            raise ValueError(
                f"Encoding for '{name}' has '{field}' {scalar}, outside the range "
                f"[{low}, {high}] representable by {dtype}. A zero point that does not fit "
                f"the quantized dtype wraps silently and shifts the entire grid."
            )


def _validate_encoding_entry(entry: Any, name: str, encoding_version: str):
    """
    Check one encoding entry against the layout its file declares.

    :param entry: A single encoding, as stored in the file
    :param name: The tensor the entry belongs to, for the error message
    :param encoding_version: The file's declared ``"version"``
    :raises ValueError: If the entry is the wrong type, or is missing or misdeclaring fields
    """
    # 0.6.1 stores a list of per-channel dicts; every other layout stores one dict.
    if encoding_version == "0.6.1":
        if not isinstance(entry, list) or not entry:
            raise ValueError(
                f"Encoding for '{name}' in a version 0.6.1 file must be a non-empty list of "
                f"per-channel encodings, got "
                f"{type(entry).__name__ if not isinstance(entry, list) else 'an empty list'}."
            )
        per_channel = entry
    else:
        if not isinstance(entry, dict):
            raise ValueError(
                f"Encoding for '{name}' in a version {encoding_version} file must be a dict, "
                f"got {type(entry).__name__}."
            )
        per_channel = [entry]

    for element in per_channel:
        if not isinstance(element, dict):
            raise ValueError(
                f"Encoding for '{name}' in a version {encoding_version} file must contain "
                f"dicts, got {type(element).__name__}."
            )

        missing = [
            field
            for field in _REQUIRED_ENCODING_FIELDS[encoding_version]
            if field not in element
        ]
        if missing:
            raise ValueError(
                f"Encoding for '{name}' is missing field(s) {missing}, required by encoding "
                f"version {encoding_version}. Does the file's version match its contents?"
            )

        foreign = [
            field
            for field in _FOREIGN_ENCODING_FIELDS[encoding_version]
            if field in element
        ]
        if foreign:
            raise ValueError(
                f"Encoding for '{name}' declares field(s) {foreign}, which belong to an older "
                f"encoding version, in a version {encoding_version} file. Mixed-version "
                f"encodings are parsed as the wrong version; re-export the file."
            )

        enc_type = element.get("enc_type")
        if enc_type is not None:
            if enc_type not in _VALID_ENC_TYPES:
                raise ValueError(
                    f"Encoding for '{name}' has unsupported enc_type '{enc_type}'. Expected "
                    f"one of {sorted(_VALID_ENC_TYPES)}."
                )
            if enc_type in _BLOCK_ENC_TYPES and "block_size" not in element:
                raise ValueError(
                    f"Encoding for '{name}' has enc_type '{enc_type}' but no 'block_size', "
                    f"which every blockwise layout requires."
                )

        # Legacy is exempt: it derives the axis from enc_type plus a caller-supplied default, so
        # its blockwise entries legitimately carry no axis.
        if (
            encoding_version not in ("0.6.1", "1.0.0")
            and "block_size" in element
            and "axis" not in element
        ):
            raise ValueError(
                f"Encoding for '{name}' has 'block_size' but no 'axis'. Blockwise encodings "
                f"must state the axis the blocks run along."
            )

        axis = element.get("axis")
        if axis is not None and not isinstance(axis, int) or isinstance(axis, bool):
            raise ValueError(
                f"Encoding for '{name}' has a non-integer 'axis' ({axis!r})."
            )

        # Only integer encodings define a grid. float16/bfloat16 are simulated as a plain cast,
        # so carrying no scale is correct for them.
        dtype_field, int_marker = _INT_DTYPE_FIELD[encoding_version]
        dtype = element.get(dtype_field)
        if isinstance(dtype, str) and int_marker in dtype:
            affine_missing = [
                field
                for field in _REQUIRED_AFFINE_FIELDS[encoding_version]
                if field not in element
            ]
            if affine_missing:
                raise ValueError(
                    f"Integer encoding for '{name}' is missing field(s) {affine_missing}, "
                    f"required by encoding version {encoding_version}."
                )

            scales = [f for f in _SCALE_FIELDS[encoding_version] if f in element]
            if not scales:
                raise ValueError(
                    f"Integer encoding for '{name}' has no scale. Expected one of "
                    f"{list(_SCALE_FIELDS[encoding_version])} in a version "
                    f"{encoding_version} file."
                )

            for field in scales:
                value = element[field]
                flat = value if isinstance(value, (list, tuple)) else [value]
                if not len(flat):
                    raise ValueError(f"Encoding for '{name}' has an empty '{field}'.")
                # A non-positive or non-finite scale gives a degenerate or NaN grid, which ONNX
                # runs without complaint.
                for scalar in np.asarray(value, dtype=object).reshape(-1):
                    if not isinstance(scalar, (int, float)) or isinstance(scalar, bool):
                        raise ValueError(
                            f"Encoding for '{name}' has a non-numeric value in '{field}': "
                            f"{scalar!r}."
                        )
                    if field != "per_block_int_scale" and (
                        not np.isfinite(scalar) or scalar <= 0
                    ):
                        raise ValueError(
                            f"Encoding for '{name}' has a non-positive or non-finite "
                            f"'{field}' ({scalar}). Scales must be finite and > 0."
                        )

            _validate_zero_point(element, name, dtype, encoding_version)


def _validate_encodings(encodings_dict: dict, encoding_version: str):
    """
    Check that an encodings file matches the layout its ``"version"`` declares.

    The loaders subscript the file directly, so without this a malformed one surfaces as a
    ``KeyError`` or ``TypeError`` from inside a comprehension.

    :param encodings_dict: Loaded encodings file
    :param encoding_version: The file's ``"version"``, already known to be registered
    :raises ValueError: If a section or entry does not match the declared version's layout
    """
    for section, expected_type in _ENCODING_SECTIONS[encoding_version].items():
        if section not in encodings_dict:
            raise ValueError(
                f"Encodings file of version {encoding_version} is missing required section "
                f"'{section}'. Expected section(s): "
                f"{sorted(_ENCODING_SECTIONS[encoding_version])}."
            )

        entries = encodings_dict[section]
        if not isinstance(entries, expected_type):
            raise ValueError(
                f"Section '{section}' of a version {encoding_version} encodings file must be "
                f"a {expected_type.__name__}, got {type(entries).__name__}."
            )

        if isinstance(entries, dict):
            # 0.6.1 keys by tensor name, so duplicates cannot occur.
            for name, entry in entries.items():
                _validate_encoding_entry(entry, name, encoding_version)
        else:
            seen = set()
            for position, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Entry {position} of '{section}' in a version {encoding_version} "
                        f"encodings file must be a dict, got {type(entry).__name__}."
                    )
                if "name" not in entry:
                    raise ValueError(
                        f"Entry {position} of '{section}' in a version {encoding_version} "
                        f"encodings file has no 'name' field, so it cannot be matched to a "
                        f"tensor."
                    )
                # The loaders key by name, so a repeat discards the earlier entry.
                if entry["name"] in seen:
                    raise ValueError(
                        f"Tensor '{entry['name']}' has more than one encoding in '{section}'. "
                        f"Only one would be applied and the rest silently discarded."
                    )
                seen.add(entry["name"])
                _validate_encoding_entry(entry, entry["name"], encoding_version)

    # The loaders merge the legacy sections with `param | activation`, so a name in both keeps
    # one encoding and drops the other.
    if encoding_version in ("0.6.1", "1.0.0"):
        activations, params = (
            encodings_dict["activation_encodings"],
            encodings_dict["param_encodings"],
        )
        if isinstance(activations, dict):
            shared = set(activations) & set(params)
        else:
            shared = {e["name"] for e in activations} & {e["name"] for e in params}
        if shared:
            raise ValueError(
                f"Tensor(s) {sorted(shared)} appear in both 'activation_encodings' and "
                f"'param_encodings'. Each tensor must be quantized as one or the other."
            )


def _flatten_encodings(
    encodings: dict | str,
) -> Tuple[Dict[str, Any], Set[str]]:
    """
    Load an encodings file (or dict) and flatten it to ``{tensor_name: encoding}``.

    :return: (all encodings keyed by tensor name, names that came from ``param_encodings``)
    """
    if isinstance(encodings, dict):
        encodings_dict = encodings
    elif isinstance(encodings, (str, os.PathLike)):
        with open(encodings) as json_file:
            encodings_dict = json.load(json_file)
    else:
        raise ValueError(
            f"Encodings must be a dict or a path to an encodings file, got "
            f"{type(encodings).__name__}."
        )

    if not isinstance(encodings_dict, dict):
        raise ValueError(
            f"Encodings must be a dict at the top level, got "
            f"{type(encodings_dict).__name__}."
        )

    encoding_version = encodings_dict.get("version", None)
    try:
        flatten = _ENCODING_LOADERS[encoding_version]
    except KeyError:
        # Deliberately not falling back to a layout guess: a version added to AIMET after
        # this table was written must be registered explicitly, not silently misread.
        raise ValueError(
            f"Unsupported encoding version {encoding_version!r}; expected one of "
            f"{sorted(_ENCODING_LOADERS)}."
        ) from None

    _validate_encodings(encodings_dict, encoding_version)

    return flatten(encodings_dict)


def _num_scales(encoding: Any) -> int:
    """Number of scales in a native (any-version) encoding, without parsing it."""
    if isinstance(encoding, list):  # 0.6.1
        return len(encoding)

    scale = encoding.get("scale", encoding.get("y_scale", None))
    if scale is None:
        scale = encoding.get("per_channel_float_scale", None)
    if scale is None:
        return 1
    return int(np.asarray(scale).size)


def _tensor_float_type(model: onnx.ModelProto, name: str) -> np.dtype:
    """
    The float dtype of a tensor, used as the dtype of the scale emitted alongside it.

    Falls back to float32 for a tensor whose type the graph does not declare.
    """
    for tensor in model.graph.initializer:
        if tensor.name == name:
            return onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)

    for value_info in (
        *model.graph.input,
        *model.graph.value_info,
        *model.graph.output,
    ):
        if value_info.name == name and value_info.type.tensor_type.elem_type:
            return onnx.helper.tensor_dtype_to_np_dtype(
                value_info.type.tensor_type.elem_type
            )

    return np.float32


def _unique_axis_by_shape(
    dims: Optional[Tuple[int, ...]], n_scales: int
) -> Optional[int]:
    """
    The channel axis implied by the tensor's shape.

    If exactly one dimension holds one scale per channel, that is the channel axis. Returns
    None for per-tensor encodings and for shapes where more than one dimension qualifies.
    """
    if n_scales <= 1 or not dims:
        return None

    matches = [i for i, dim in enumerate(dims) if dim == n_scales]
    return matches[0] if len(matches) == 1 else None


def _channel_axes_from_connected_graph(model: onnx.ModelProto) -> Dict[str, int]:
    """
    Channel axis of every parameter, from :meth:`~QuantizationSimModel._get_quantization_axes`.

    Resolves the axis by op type rather than by shape, so it can disambiguate tensors whose
    shape alone is not conclusive.
    """
    axes = {}

    try:
        connected_graph = ConnectedGraph(model)
    except Exception as e:
        raise RuntimeError(
            "Could not determine the per-channel axis of every parameter from tensor "
            "shapes alone, and building a ConnectedGraph to resolve the rest failed. "
            "Export encodings in version 2.0.0, which stores the axis explicitly."
        ) from e

    for op in connected_graph.get_all_ops().values():
        channel_axis, _ = QuantizationSimModel._get_quantization_axes(op)  # pylint: disable=protected-access

        if channel_axis is None:
            continue

        for product, _unused in (op.parameters or {}).values():
            axes[product.name] = channel_axis

    return axes


def _required_opset(encodings: List[dict], current_opset: int) -> int:
    """Minimum ONNX opset able to represent these 2.0.0 encodings."""
    desired = max(current_opset, 10)

    for encoding in encodings:
        if encoding.get("axis") is not None:
            desired = max(desired, 13)
        if encoding.get("block_size"):
            desired = max(desired, 21)

        output_dtype = encoding.get("output_dtype", "")
        if output_dtype.startswith("float8"):
            desired = max(desired, 19)
        elif output_dtype.startswith(("int", "uint")) and output_dtype not in (
            "int8",
            "uint8",
            "int32",
            "uint32",
        ):
            # INT4/INT16
            desired = max(desired, 21)

    return desired


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def encodings_to_onnx_qdq(
    model: onnx.ModelProto | ONNXModel | str,
    encodings: dict | str,
    *,
    prequantize_constants: bool = False,
    force_activation_as: Optional[Literal["unsigned", "signed"]] = "unsigned",
) -> onnx.ModelProto:
    """
    Convert an ONNX model and its AIMET encodings into an ONNX QDQ model.

    This is a direct translation: the encodings determine exactly which tensors are
    quantized and how. No :class:`QuantizationSimModel` is created, so no quantsim
    configuration (supergroups, op tying, encoding constraints) can add or remove
    quantizers relative to what the encodings specify.

    [b]float16 is not a quantized data type and onnx QDQ has no representation for it, so
    plain-cast float encodings are omitted with a warning, whether they name a parameter or an
    activation. Scaled float formats such as FP8 do carry a scale, and are emitted as Q/DQ.

    An encoding naming a tensor that is not in the model is an error.

    Args:
        model: ONNX model as a :class:`ModelProto` or :class:`ONNXModel`, or a path to one.
            The model is not modified in place.
        encodings: AIMET encodings as a dict, or a path to an encodings file. Versions
            0.6.1, 1.0.0, 2.0.0, and 2.1.0 are accepted.
        prequantize_constants: If True, store quantized parameters as integer initializers
            rather than emitting ``QuantizeLinear`` for them. (default: False)
        force_activation_as: Cast all activation encodings to signed or unsigned, a
            workaround for QNN converter limitations. Pass None to leave them as-is.
            (default: "unsigned")

    Returns:
        The equivalent ONNX QDQ model.

    Examples:
        >>> qdq_model = encodings_to_onnx_qdq("model.onnx", "model.encodings")
        >>> onnx.save_model(qdq_model, "model_qdq.onnx")
    """
    if isinstance(model, str):
        model = onnx.load(model)
    else:
        if isinstance(model, ONNXModel):
            model = model.model
        model_copy = onnx.ModelProto()
        model_copy.CopyFrom(model)
        model = model_copy

    # TODO: Support exporting (b)float16 models. The scale tensors are emitted as float32
    # below, which a (b)float16 graph cannot consume.
    if contains_tensor_type(
        model, (onnx.TensorProto.BFLOAT16, onnx.TensorProto.FLOAT16)
    ):
        raise RuntimeError(
            "Exporting to onnx QDQ is only supported for float32 models."
        )

    all_encodings, param_names = _flatten_encodings(encodings)

    constants = _get_all_constants(model)
    producers = {out for node in model.graph.node for out in node.output}
    graph_tensors = set(constants) | producers | {i.name for i in model.graph.input}

    # Recover the per-channel axis, which only version 2.0.0 stores. ConnectedGraph is built
    # only for tensors whose shape is inconclusive, since it is not free on large models.
    shape_axes = {}
    unresolved = []
    for name, encoding in all_encodings.items():
        if name not in constants:
            continue
        n_scales = _num_scales(encoding)
        axis = _unique_axis_by_shape(tuple(constants[name].dims), n_scales)
        if axis is not None:
            shape_axes[name] = axis
        elif n_scales > 1:
            unresolved.append(name)

    channel_axes = shape_axes
    if unresolved:
        logger.info(
            "Deriving the per-channel axis of %d parameter(s) from the model graph: %s",
            len(unresolved),
            sorted(unresolved)[:8],
        )
        channel_axes = {**_channel_axes_from_connected_graph(model), **shape_axes}

    qdq_node_info: Dict[str, List] = {
        "input_names": [],
        "output_names": [],
        "node_name_prefixes": [],
        "encodings": [],
    }
    plain_cast_floats: List[str] = []
    unmatched: List[str] = []

    for name, encoding in all_encodings.items():
        if name not in graph_tensors:
            unmatched.append(name)
            continue

        # 2.0.0 has no param/activation split, so a static tensor counts as a parameter
        # regardless of which section (if any) the encoding came from.
        is_constant = name in constants
        is_param = is_constant or name in param_names
        dims = tuple(constants[name].dims) if is_constant else None

        parsed = EncodingBase.from_qnn_encoding_dict(
            encoding, input_shape=dims, default_channel_axis=channel_axes.get(name)
        )

        qdq_encoding = parsed.to_qnn_encoding_dict("2.0.0")

        if not qdq_encoding:
            # Plain-cast float formats (float16, bfloat16) carry no scale and so have no
            # 2.0.0 encoding. Scaled floats such as FP8 do, and take the QDQ path below.
            plain_cast_floats.append(name)
            continue

        # A wrong axis yields a valid-but-wrong graph, so check it against the shape. Only
        # per-channel encodings hold one scale per channel along ``axis``; blockwise scales
        # are 2-D and report the block axis, so they are exempt.
        emitted_axis = qdq_encoding.get("axis")
        emitted_scale = np.asarray(qdq_encoding.get("y_scale", ()))
        if (
            emitted_axis is not None
            and dims
            and not qdq_encoding.get("block_size")
            and emitted_scale.ndim == 1
            and dims[emitted_axis % len(dims)] != emitted_scale.size
        ):
            raise RuntimeError(
                f"Per-channel axis {emitted_axis} of '{name}' does not match its shape "
                f"{dims}: expected dimension {emitted_scale.size} to hold one scale per "
                "channel."
            )

        if not is_param and force_activation_as is not None:
            qdq_encoding = (
                _to_unsigned_encoding(qdq_encoding)
                if force_activation_as == "unsigned"
                else _to_signed_encoding(qdq_encoding)
            )

        qdq_node_info["input_names"].append(name)
        qdq_node_info["output_names"].append(name + "_qdq")
        qdq_node_info["node_name_prefixes"].append(name)
        qdq_node_info["encodings"].append(qdq_encoding)

    if unmatched:
        raise RuntimeError(
            f"The following encoding names were present in the encodings to load but not "
            f"found in the model: {sorted(unmatched)}"
        )

    derived_encodings = _derive_data_movement_op_encodings(
        model, dict(zip(qdq_node_info["input_names"], qdq_node_info["encodings"]))
    )
    for name, encoding in derived_encodings.items():
        qdq_node_info["input_names"].append(name)
        qdq_node_info["output_names"].append(name + "_qdq")
        qdq_node_info["node_name_prefixes"].append(name)
        qdq_node_info["encodings"].append(encoding)

    current_opset = next(
        opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")
    )
    desired_opset = _required_opset(qdq_node_info["encodings"], current_opset)
    if current_opset < desired_opset:
        logger.info(
            "Converting model from opset %d to %d to represent these encodings",
            current_opset,
            desired_opset,
        )
        model = _convert_version(model, desired_opset)

    # _required_opset floors at current_opset, so this is the model's opset whether or not the
    # conversion above ran.
    model_opset = max(current_opset, desired_opset)

    # float_types is the dtype the scale tensors are stored as. onnx::QuantizeLinear required
    # the scale and the input to share a dtype until opset 22; from opset 23 the scale may be
    # float32 whatever the input's dtype is. Decided here rather than alongside the encodings
    # because _required_opset may still raise the opset above.
    qdq_node_info["float_types"] = [
        np.float32 if model_opset >= 23 else _tensor_float_type(model, name)
        for name in qdq_node_info["input_names"]
    ]

    _add_onnx_qdq_nodes(
        model,
        **qdq_node_info,
        onnx_opset=model_opset,
        prequantize_constants=prequantize_constants,
    )

    if plain_cast_floats:
        warnings.warn(
            _red(
                "[b]float16 is no more considered a quantized data type, so onnx QDQ has no "
                f"representation for it. Omitting {len(plain_cast_floats)} encoding(s): "
                f"{sorted(plain_cast_floats)[:8]}"
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    ONNXModel(model).topological_sort()

    # onnx requires metadata_props keys to be unique, so overwrite rather than append when
    # converting a model that already carries a producer.
    producer = next((p for p in model.metadata_props if p.key == "producer"), None)
    if producer is None:
        producer = model.metadata_props.add()
        producer.key = "producer"
    producer.value = f"aimet-onnx {aimet_onnx.__version__}"

    return model


# pylint: disable=too-many-locals, too-many-branches
def load_encodings_to_sim(
    quant_sim_model: QuantizationSimModel,
    onnx_encoding_path: str | dict,
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
    # pylint: disable=protected-access
    mismatched_encodings = []

    # Load encodings file
    if isinstance(onnx_encoding_path, dict):
        encodings = onnx_encoding_path
    else:
        with open(onnx_encoding_path) as json_file:
            encodings = json.load(json_file)

    encoding_version = encodings.get("version", None)
    if encoding_version not in VALID_ENCODING_VERSIONS:
        raise NotImplementedError(
            f"Encoding version should be one of {VALID_ENCODING_VERSIONS}; "
            f"got {encoding_version}"
        )

    if encoding_version == "0.6.1":
        param_encodings = encodings["param_encodings"].copy()
        activation_encodings = encodings["activation_encodings"].copy()
        all_encodings = param_encodings | activation_encodings
    elif encoding_version == "1.0.0":
        param_encodings = {
            encoding["name"]: encoding for encoding in encodings["param_encodings"]
        }
        activation_encodings = {
            encoding["name"]: encoding for encoding in encodings["activation_encodings"]
        }
        all_encodings = param_encodings | activation_encodings
    else:
        all_encodings = {e["name"]: e for e in encodings["encodings"]}

    if encoding_version in ("0.6.1", "1.0.0"):
        all_quantizers = quant_sim_model.qc_quantize_op_dict.copy()
    else:
        # 2.0.0 doesn't have float16 encoding, ignore float quantizers
        # unless they appear in encodings
        all_quantizers = {
            name: qtzr
            for name, qtzr in quant_sim_model.qc_quantize_op_dict.items()
            if name in all_encodings
            or not (
                qtzr.data_type == QuantizationDataType.float and qtzr.bitwidth >= 16
            )
        }

    _validate_encodings_to_load(quant_sim_model, all_encodings.keys())

    if encoding_version == "2.0.0":
        _remove_delegatable_excess_encodings(quant_sim_model, all_encodings)

    # First pass through quantizers to check for mismatched encodings
    missing_quantizers = all_encodings.keys() - all_quantizers.keys()

    for name in missing_quantizers:
        mismatched_encodings.append(
            _EncodingMismatchInfo(name, enabled_mismatch=(False, True))
        )

    for quantizer_name, quantizer in all_quantizers.items():
        e = all_encodings.get(quantizer_name, None)

        if not e:
            mismatched_info = get_encoding_mismatch_info(
                quantizer_name, quantizer, None
            )
            if mismatched_info.has_mismatch():
                mismatched_encodings.append(mismatched_info)
            continue

        encoding = EncodingBase.from_qnn_encoding_dict(e)

        if isinstance(encoding, FloatEncoding) and encoding.is_scaled:
            # The generic mismatch check operates on 1.0.0 dicts, which describe integer
            # affine grids only and cannot represent a scaled float8 grid.
            if not quantizer.enabled:
                # Encoding provided for a disabled quantizer. As in the generic path,
                # other mismatches are not reported once the enabled state disagrees.
                mismatched_encodings.append(
                    _EncodingMismatchInfo(
                        quantizer_name, enabled_mismatch=(quantizer.enabled, True)
                    )
                )
                continue

            loaded_precision = qtype.from_string(encoding.dtype)
            mismatch_info = _EncodingMismatchInfo(quantizer_name)
            if quantizer.precision() != loaded_precision:
                mismatch_info.dtype_mismatch = (
                    str(quantizer.precision()),
                    str(loaded_precision),
                )

            if quantizer._encoding_type() != encoding._encoding_type():
                mismatch_info.enc_type_mismatch = (
                    quantizer._encoding_type(),
                    encoding._encoding_type().name,
                )

            if mismatch_info.has_mismatch():
                mismatched_encodings.append(mismatch_info)
            continue

        e = encoding.to_qnn_encoding_dict("1.0.0")

        mismatched_info = get_encoding_mismatch_info(quantizer_name, quantizer, e)
        if mismatched_info.has_mismatch():
            mismatched_encodings.append(mismatched_info)

    log_and_catch_mismatched_encodings(mismatched_encodings, strict)

    if missing_quantizers:
        if not strict and encoding_version in ("0.6.1", "1.0.0"):
            _add_missing_quantizers(
                quant_sim_model,
                param_encodings.keys(),
                activation_encodings.keys(),
            )
            all_quantizers |= {
                name: qtzr
                for name, qtzr in quant_sim_model.qc_quantize_op_dict.items()
                if name in missing_quantizers
            }
        else:
            # 2.0.0 encoding has no distinction between param and act encodings,
            # so we cannot add missing quantizers
            raise RuntimeError(
                f"Encodings were provided for missing quantizers: {missing_quantizers}. "
            )

    bias_names = set(
        bias.name
        for op in quant_sim_model.connected_graph.get_all_ops().values()
        for _, bias in [quant_sim_model._get_weight_and_bias(op)]
        if bias is not None
    )

    # Second pass through quantizers to set quantizer settings
    for quantizer_name, quantizer in all_quantizers.items():
        e = all_encodings.get(quantizer_name, None)

        if not e:
            if disable_missing_quantizers:
                quantizer.enabled = False
            continue

        quantizer._load_encodings_dict(e, allow_overwrite=allow_overwrite)

        if quantizer.bitwidth >= 32 and quantizer_name in bias_names:
            # Disable bias quantizers with bitwidth >= 32 assuming lossless quantization.
            # The loaded int32 bias encoding won't be simulated but will be exported
            # if sim.export or sim.to_onnx_qdq is called with export_int32_bias=True
            quantizer.enabled = False

    return mismatched_encodings


def _validate_encodings_to_load(
    quant_sim_model: QuantizationSimModel,
    encoding_names: Iterable[str],
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
    for name in encoding_names:
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
    sim: QuantizationSimModel,
    param_names: Iterable[str],
    activation_names: Iterable[str],
):
    """
    Add quantizers for any tensors which are present in encodings_to_load but are not present in
    sim.qc_quantize_op_dict
    """
    # pylint:disable = protected-access
    added_quantizers = set()

    # Insert any missing activation quantizers as disabled act quantizers
    for tensor_name in activation_names:
        if tensor_name not in sim.qc_quantize_op_dict:
            # No graph given: searched per call, fine for the few quantizers missing in the sim
            sim._insert_quantizer(tensor_name, is_param=False)
            sim.qc_quantize_op_dict[tensor_name].enabled = False
            sim.activation_names.append(tensor_name)
            added_quantizers.add(tensor_name)

    # Insert any missing param quantizers as disabled param quantizers
    for tensor_name in param_names:
        if tensor_name not in sim.qc_quantize_op_dict:
            # No graph given: searched per call, fine for the few quantizers missing in the sim
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
    if args and (inputs is not None or forward_pass_callback):
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

    if inputs is not None and (
        forward_pass_callback or forward_pass_callback_args is not _NOT_SPECIFIED
    ):
        raise TypeError(msg)
    if inputs is None and forward_pass_callback is None:
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
    nodes_to_exclude: Optional[List[str]] = None,
    *,
    excluded_nodes: Optional[List[str]] = None,
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
    :param nodes_to_exclude: List of onnx node names to exclude from blockwise weight quantization. It can be empty if no nodes are excluded


    Examples:

        >>> # Assume 'sim' is a QuantizationSimModel object
        >>> # Allows setting of all Linear and Conv weight quantizers to block_size 64 in the input_channels dimension:
        >>> set_blockwise_quantization_for_weights(sim=sim,
        ...                                        op_types=("Gemm", "MatMul", "Conv"),
        ...                                        bitwidth=4,
        ...                                        symmetric=True,
        ...                                        block_size=64,
    ...                                            nodes_to_exclude = ['conv1'])
    """

    if excluded_nodes is not None:
        logger.warning(
            "The argument 'excluded_nodes' is deprecated and will be removed in future releases. Use 'nodes_to_exclude' instead."
        )
        if nodes_to_exclude is not None:
            raise ValueError(
                "Both 'nodes_to_exclude' and 'excluded_nodes' parameters cannot be set at the same time. Use only 'nodes_to_exclude'."
            )
        nodes_to_exclude = excluded_nodes

    if isinstance(op_types, str):
        op_types = (op_types,)

    if not nodes_to_exclude:
        nodes_to_exclude = []

    qspec = QSpec.blockwise(
        qtype.int(bitwidth), block_size=block_size, symmetric=symmetric
    )

    for op in sim.connected_graph.ordered_ops:
        if op.type not in op_types:
            continue

        if op.name in nodes_to_exclude:
            continue

        _, _, param_quantizers = sim.get_op_quantizers(op)

        weight_quantizer: QcQuantizeOp = param_quantizers.get("weight")
        bias_quantizer: QcQuantizeOp = param_quantizers.get("bias")

        if not weight_quantizer:
            continue

        try:
            weight_quantizer.set_qspec(qspec)
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


@deprecated("Use 'set_lpbq_for_params' instead.")
def set_grouped_blockwise_quantization_for_weights(
    sim: QuantizationSimModel,
    op_types: Union[str, Tuple],
    bitwidth: int,
    decompressed_bw: int,
    block_size: int,
    strict: bool = False,
    nodes_to_exclude: Optional[List[str]] = None,
    *,
    excluded_nodes: Optional[List[str]] = None,
):
    """
    Set weight parameter quantizers of modules to grouped blockwise quantization.

    :param sim: Quantsim to set weight quantizers for
    :param op_types: Operator types for which to enable grouped blockwise weight quantizaiton
    :param bitwidth: Bitwidth for affine quantization
    :param decompressed_bw: Decompressed bw for grouped block quantization
    :param block_size: Block size for affine quantization. The block size will be applied to the weight's input features
        dimension, while per-channel will be used for the weight's output features dimension
    :param nodes_to_exclude: List of onnx node names to exclude from blockwise weight quantization. It can be empty if no nodes are excluded

    Examples:

        >>> # Assume 'sim' is a QuantizationSimModel object
        >>> # Sets of all Gemm, MatMul, and Conv weight quantizers to block_size 64 in the input_channels dimension:
        >>> set_grouped_blockwise_quantization_for_weights(sim=sim,
        ...                                                op_types=("Gemm", "MatMul", "Conv"),
        ...                                                bitwidth=4,
        ...                                                decompressed_bw=8,
        ...                                                block_size=64,
        ...                                                nodes_to_exclude = ['conv1'])
    """
    if excluded_nodes is not None:
        logger.warning(
            "The argument 'excluded_nodes' is deprecated and will be removed in future releases. Use 'nodes_to_exclude' instead."
        )
        if nodes_to_exclude is not None:
            raise ValueError(
                "Both 'nodes_to_exclude' and 'excluded_nodes' parameters cannot be set at the same time. Use only 'nodes_to_exclude'."
            )
        nodes_to_exclude = excluded_nodes

    if isinstance(op_types, str):
        op_types = (op_types,)

    if not nodes_to_exclude:
        nodes_to_exclude = []

    def get_lpbq_params(op: Op):
        if op.type in op_types and op.name not in nodes_to_exclude:
            return QSpec.lpbq(
                qtype.int(bitwidth), block_size, decompressed_bw - bitwidth
            )
        return None

    return _set_grouped_blockwise_quantization_for_weights(sim, get_lpbq_params, strict)


@overload
def set_lpbq_for_params(
    sim: QuantizationSimModel,
    bitwidth: int,
    block_size: int,
    *,
    nodes_to_include: Set[str] = None,
): ...


@overload
def set_lpbq_for_params(
    sim: QuantizationSimModel,
    bitwidth: int,
    block_size: int,
    *,
    op_types: Union[str, Set[str]] = None,
    nodes_to_exclude: None = None,
    strict: bool = False,
): ...


def set_lpbq_for_params(
    sim: QuantizationSimModel,
    bitwidth: int,
    block_size: int,
    *,
    op_types: Optional[Union[str, Set[str]]] = None,
    nodes_to_exclude: Optional[Set[str]] = None,
    nodes_to_include: Optional[Set[str]] = None,
    strict: bool = None,
):
    """
    Set weight quantizers of specified nodes to use low-power blockwise quantization.

    This function is overloaded with the following signatures:

    .. function:: set_lpbq_for_params(sim, bitwidth, block_size, *, nodes_to_include = None)
        :noindex:

        :param QuantizationSimModel sim: Quantsim to set weight quantizers for
        :param int bitwidth: Compressed bitwidth for lpbq quantization
        :param int block_size: Block size for affine quantization. The block size will be applied to the
            weight's input features dimension, while per-channel will be used for the weight's output features dimension
        :param Set[str] nodes_to_include: Set of onnx node names to include for blockwise weight quantization.

    .. function:: set_lpbq_for_params(sim, bitwidth, block_size, *, op_types=None, nodes_to_exclude=None, strict=False)
        :noindex:

        :param QuantizationSimModel sim: Quantsim to set weight quantizers for
        :param int bitwidth: Compressed bitwidth for lpbq quantization
        :param int block_size: Block size for affine quantization. The block size will be applied to the
            weight's input features dimension, while per-channel will be used for the weight's output features dimension
        :param Union[str, Set[str]] op_types: Operator types for which to enable grouped blockwise weight quantizaiton
        :param Set[str] nodes_to_exclude: Set of onnx node names to exclude from blockwise weight quantization.
        :param bool strict: If False, only enable blockwise quant for layers with dimensions evenly divisible by block_size.
            If True, throw an error for layers with incompatible shapes.

    Examples:

        >>> sim = QuantizationSimModel(...)
        >>> set_lpbq_for_params(sim, bitwidth=4, block_size=64, op_types={"Gemm", "MatMul", "Conv"})
        >>> # or
        >>> set_lpbq_for_params(sim, bitwidth=4, block_size=64, nodes_to_include={"/lm_head/MatMul", ...})

    """
    if isinstance(op_types, str):
        op_types = {op_types}

    if nodes_to_exclude and nodes_to_include:
        raise ValueError(
            "Both 'nodes_to_exclude' and 'nodes_to_include' arguments cannot be set at the same time."
        )

    if op_types and nodes_to_include:
        raise ValueError(
            "Both 'op_types' and 'nodes_to_include' arguments cannot be set at the same time."
        )

    if op_types is None and nodes_to_include is None:
        raise ValueError(
            "Either 'op_types' or 'nodes_to_include' argument must be provided."
        )

    if not nodes_to_include:
        nodes_to_exclude = nodes_to_exclude or set()
        nodes_to_include = {
            op.name
            for op in sim.connected_graph.ordered_ops
            if op.type in op_types and op.name not in nodes_to_exclude
        }
        strict = strict or False
    else:
        if strict is not None:
            raise TypeError(
                "Cannot specify 'strict' when 'nodes_to_include' is provided."
            )
        strict = True

    def get_lpbq_params(op: Op):
        if op.name in nodes_to_include:
            return QSpec.lpbq(qtype.int(bitwidth), block_size, scale_bits=bitwidth)
        return None

    return _set_grouped_blockwise_quantization_for_weights(sim, get_lpbq_params, strict)


def _set_grouped_blockwise_quantization_for_weights(
    sim: QuantizationSimModel,
    get_lpbq_params: Callable[[Op], QSpec],
    strict: bool = False,
):
    for op in sim.connected_graph.ordered_ops:
        qspec = get_lpbq_params(op)

        if qspec is None:
            continue

        _, _, param_quantizers = sim.get_op_quantizers(op)

        weight_quantizer: QcQuantizeOp = param_quantizers.get("weight")
        bias_quantizer: QcQuantizeOp = param_quantizers.get("bias")

        if not weight_quantizer:
            continue

        try:
            weight_quantizer.set_qspec(qspec)
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


@overload
def set_param_type(
    sim: QuantizationSimModel,
    param_type: QSpec | qtype | str,
    *,
    op_types: Optional[Tuple[str] | str] = None,
    nodes_to_exclude: Optional[Set[str]] = None,
    strict: bool = True,
    shift_zero_point: bool = False,
): ...


@overload
def set_param_type(
    sim: QuantizationSimModel,
    param_type: QSpec | qtype | str,
    *,
    nodes_to_include: Optional[Set[str]] = None,
    strict: bool = True,
    shift_zero_point: bool = False,
): ...


def set_param_type(
    sim: QuantizationSimModel,
    param_type: QSpec | qtype | str,
    **kwargs,
):
    """
    Set parameter quantization data type for specified layers.

    This function is overloaded with the following signatures:

    .. function:: set_param_type(sim, param_type, *, nodes_to_include=None, strict=True, shift_zero_point=False)
        :noindex:

        :param QuantizationSimModel sim: Quantsim to set param type for
        :param QSpec | qtype | str param_type: Quantization data type to set for the parameters
        :param Set[str] nodes_to_include: Set of onnx node names for which to set parameter quantization data type. If None, all nodes are included
        :param bool strict: If False, skip quantizers incompatible with ``param_type``.
        :param bool shift_zero_point: (Deprecated) Whether to shift the quantizer's zero point (only for int2 param type).

    .. function:: set_param_type(sim, param_type, *, nodes_to_include=None, strict=True, shift_zero_point=False)
        :noindex:

        :param QuantizationSimModel sim: Quantsim to set param type for
        :param QSpec | qtype | str param_type: Quantization data type to set for the parameters
        :param Set[str] op_types: Set of onnx op types for which to set parameter quantization data type. If None, all types are included
        :param Set[str] nodes_to_exclude: Set of onnx node names to exclude for setting parameter quantization data type
        :param bool strict: If False, skip quantizers incompatible with ``param_type``.
        :param bool shift_zero_point: (Deprecated) Whether to shift the quantizer's zero point (only for int2 param type).

    Examples:

        >>> sim = QuantizationSimModel(...)

        >>> # Set all parameter quantizers to int8 data type
        >>> set_param_type(sim, aimet_onnx.int8)

        >>> # Set parameter quantizers of Conv, MatMul, and Gemm layers to int4 data type
        >>> set_param_type(sim, aimet_onnx.int4, op_types={"Conv", "MatMul", "Gemm"})

        >>> # Set parameter quantizers of "/lm_head/MatMul" to int2 with shifted zero point
        >>> spec = aimet_onnx.QSpec.blockwise(int2, block_size=64, symmetric=True, shift_zero_point=True)
        >>> set_param_type(sim, spec, nodes_to_include={"/lm_head/MatMul"})

        >>> # Set parameter quantizers of "/lm_head/MatMul" to LPBQ
        >>> spec = aimet_onnx.QSpec.lpbq(int4, block_size=64)
        >>> set_param_type(sim, spec, nodes_to_include={"/lm_head/MatMul"})

    """
    nodes_to_exclude = kwargs.pop("nodes_to_exclude", None)
    nodes_to_include = kwargs.pop("nodes_to_include", None)
    op_types = kwargs.pop("op_types", None)
    strict = kwargs.pop("strict", True)
    shift_zero_point = kwargs.pop("shift_zero_point", False)

    if kwargs:
        raise TypeError(
            f"set_param_type() got unexpected keyword arguments: {list(kwargs.keys())}"
        )

    if isinstance(op_types, str):
        op_types = (op_types,)

    if not isinstance(param_type, (str, qtype, QSpec)):
        raise TypeError(
            f"param_type must be QSpec, qtype, or string, got {type(param_type)}"
        )

    if isinstance(param_type, str):
        param_type = qtype.from_string(param_type)

    if shift_zero_point:
        if isinstance(param_type, QSpec):
            raise ValueError(
                "shift_zero_point should only be specified during construction of QSpec"
            )
        if param_type != int2:
            raise ValueError("shift_zero_point is only supported for int2 param type.")
        logger.warning(
            "shift_zero_point is a deprecated argument. Specify via QSpec.blockwise(..., shift_zero_point=True) instead."
        )

    if isinstance(param_type, qtype):
        symmetric = True if shift_zero_point else None
        param_type = QSpec(
            param_type,
            granularity=None,
            symmetric=symmetric,
            shift_zero_point=shift_zero_point,
        )

    if nodes_to_exclude and nodes_to_include:
        raise ValueError(
            "Both 'nodes_to_exclude' and 'nodes_to_include' arguments cannot be set at the same time."
        )

    if op_types and nodes_to_include:
        raise ValueError(
            "Both 'op_types' and 'nodes_to_include' arguments cannot be set at the same time."
        )

    if not nodes_to_include:
        nodes_to_exclude = nodes_to_exclude or set()
        nodes_to_include = {
            op.name
            for op in sim.connected_graph.ordered_ops
            if op.name not in nodes_to_exclude
            and (op_types is None or op.type in op_types)
        }

    for op in sim.connected_graph.ordered_ops:
        if op.name not in nodes_to_include:
            continue

        _, _, param_quantizers = sim.get_op_quantizers(op)
        param_quantizers.pop("bias", None)  # Skip bias quantizers

        for quantizer in param_quantizers.values():
            if quantizer and quantizer.enabled:
                try:
                    quantizer.set_qspec(param_type)
                except ValueError:
                    if strict:
                        raise


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
