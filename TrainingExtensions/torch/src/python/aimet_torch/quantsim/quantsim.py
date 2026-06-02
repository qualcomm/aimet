# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=too-many-lines
"""Top level API for performing quantization simulation of a pytorch model"""

from collections import defaultdict
import copy
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    overload,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    Literal,
    List,
)
import math
import warnings
import itertools
import io
import json
import contextlib
import os
from packaging import version
from aimet_torch.nn.true_quant import QuantizedGRU, QuantizedLSTM, QuantizedRNN
import torch
import onnx

from aimet_torch.common import quantsim
from aimet_torch.common.defs import QuantScheme, QuantizationDataType
from aimet_torch.common.onnx._utils import (
    _is_htp_interpolation_op,
    _get_all_constants,
    _derive_const_rescale_op_output_encodings,
)
from aimet_torch.common.quantsim_config.quantsim_config import _config_file_aliases
from aimet_torch.common.utils import deprecated, _red, docstring
from aimet_torch.nn.modules import custom
from aimet_torch._base.quantsim import (
    _QuantizationSimModelBase,
    logger,
    unquantizable_modules,
    QuantParams,
    ExportableQuantModule,
    save_checkpoint,
    load_checkpoint,
    check_accumulator_overflow,
)
from aimet_torch.onnx_utils import _onnx_model_size_larger_than_max_protobuf
from aimet_torch import nn as aimet_nn
from aimet_torch.nn import (
    BaseQuantizationMixin,
    QuantizationMixin,
    UnknownModuleError,
    QuantizedReLU,
)
from aimet_torch.nn.fake_quant import _legacy_impl
from aimet_torch._builder import _V2LazyQuantizeWrapper
from aimet_torch.quantization.base import QuantizerBase, EncodingBase
from aimet_torch.quantization.tensor import QuantizedTensorBase
from aimet_torch.quantization.affine import AffineQuantizerBase, AffineEncoding
from aimet_torch.quantization.encoding_analyzer import PercentileEncodingAnalyzer
from aimet_torch.utils import patch_attr
from aimet_torch import utils
from aimet_torch.deepspeed_utils import _register_zero3_forward_hooks
from aimet_torch.experimental.transforms.transform_ops import is_mergeable_transform
import aimet_torch


__all__ = [
    "QuantizationSimModel",
    "QuantizationSimModelOnnxExporter",
    "QuantParams",
    "ExportableQuantModule",
    "save_checkpoint",
    "load_checkpoint",
    "check_accumulator_overflow",
    "load_encodings_to_sim",
    "compute_encodings_for_sims",
]

unquantizable_modules = (QuantizerBase, *unquantizable_modules)
quantized_modules = (BaseQuantizationMixin, _V2LazyQuantizeWrapper)
containers = (
    torch.nn.Container,
    torch.nn.Sequential,
    torch.nn.ModuleList,
    torch.nn.ModuleDict,
    torch.nn.ParameterList,
    torch.nn.ParameterDict,
)


class _NOT_SPECIFIED:
    pass


def _convert_to_qmodel(model: torch.nn.Module):
    """
    Helper function to convert all modules to quantized aimet.nn modules.
    """

    def _convert_to_qmodule(module: torch.nn.Module):
        if is_mergeable_transform(module):
            return module

        if not isinstance(
            module, (*quantized_modules, *unquantizable_modules, *containers)
        ) and not is_mergeable_transform(module):
            qmodule = None
            try:
                qmodule = QuantizationMixin.from_module(module)
                module = qmodule
            except UnknownModuleError as e:
                try:
                    qmodule = _legacy_impl.FakeQuantizationMixin.from_module(module)
                    module = qmodule
                except UnknownModuleError:
                    pass

                if (
                    type(module).forward != torch.nn.Module.forward
                ):  # We don't complain if the user has no intent of
                    # running forward with this module
                    if not qmodule and not tuple(module.children()):
                        exceptions[e.module_cls] = e

        for name, child in module.named_children():
            setattr(module, name, _convert_to_qmodule(child))

        return module

    exceptions: Dict[Type[torch.nn.Module], UnknownModuleError] = {}
    model = _convert_to_qmodule(model)

    if not exceptions:
        return model

    if len(exceptions) == 1:
        e = next(iter(exceptions.values()))
        raise e

    # Multiple unknown modules found. Batch all error messages in one exception
    e = next(iter(exceptions.values()))
    msg = "\n".join(
        [
            "Quantized module definitions of the following modules are not registered: [",
            *(f"    {e.module_cls}," for e in exceptions.values()),
            "]",
        ]
    )

    raise RuntimeError(
        "\n\n".join(
            [
                msg,
                "Please register the quantized module definition of the modules listed above "
                f"using `@{e.mixin_cls.__name__}.implements()` decorator.",
                "For example:",
                *(e.generate_code_example() for e in exceptions.values()),
                "If you believe these modules need not be quantized, "
                "please exclude them from quantization by calling `QuantizationMixin.ignore` "
                f"(for example, `QuantizationMixin.ignore({e.module_cls.__name__})`) "
                "before creating QuantizationSimModel.",
                f"For more details, please refer to the official API reference:\n{e.api_reference_url}",
            ]
        )
    )


class QuantizationSimModel(_QuantizationSimModelBase):  # pylint: disable=missing-class-docstring
    __doc__ = f"""
    Class that simulates the quantized model execution on a target hardware backend.

    QuantizationSimModel simulates quantization of a given model by converting
    all PyTorch modules into :ref:`quantized modules<api-torch-quantized-modules>`
    with input/output/parameter :ref:`quantizers<api-torch-quantizers>` as necessary.

    Example:

        >>> model = torchvision.models.resnet18()
        >>> dummy_input = torch.randn(1, 3, 224, 224)
        >>> sim = QuantizationSimModel(model, dummy_input)
        >>> print(model)
        ResNet(
          (conv1): Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
          )
          ...
        )
        >>> print(sim.model)
        ResNet(
          (conv1): QuantizedConv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(), qmin=-128, qmax=127, symmetric=True)
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): None
            )
          )
          ...
        )

    .. warning::
       The default value of `quant_scheme` has changed
       from `QuantScheme.post_training_tf_enhanced` to `QuantScheme.training_range_learning_with_tf_init`
       since 2.0.0, and will be deprecated in the longer term.

    Args:
        model (torch.nn.Module): Model to simulate the quantized execution of
        dummy_input (Tensor | Sequence[Tensor]): Dummy input to be used to capture
            the computational graph of the model. All input tensors are expected to be
            already placed on the appropriate devices to run forward pass of the model.
        quant_scheme (QuantScheme, optional): Quantization scheme that indicates
            how to observe and calibrate the quantization encodings (Default: `QuantScheme.post_training_tf_enhanced`)
        default_output_bw (int, optional): Default bitwidth (4-31) to use for quantizing all layer inputs and outputs
            unless otherwise specified in the config file. (Default: 8)
        default_param_bw (int, optional): Default bitwidth (4-31) to use for quantizing all layer parameters
            unless otherwise specified in the config file. (Default: 8)
        in_place (bool, optional): If True, then the given model is modified in-place into a quantized model. (Default: `False`)
        config_file (str, optional): File path or alias of the configuration file.
            Alias can be one of {{ {", ".join(_config_file_aliases.keys())} }} (Default: `"default"`)
        default_data_type (QuantizationDataType, optional): Default data type to use for quantizing all
            inputs, outputs and parameters unless otherwise specified in the config file.
            Possible options are QuantizationDataType.int and QuantizationDataType.float.
            Note that the mode default_data_type=QuantizationDataType.float is only supported with
            default_output_bw=16 or 32 and default_param_bw=16 or 32. (Default: `QuantizationDataType.int`)
    """

    _quantized_modules = quantized_modules

    def __init__(
        self,  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
        model: torch.nn.Module,
        dummy_input: Union[torch.Tensor, Sequence[torch.Tensor]],
        quant_scheme: Union[str, QuantScheme] = None,  # NOTE: Planned to be deprecated
        default_output_bw: int = 8,
        default_param_bw: int = 8,
        in_place: bool = False,
        config_file: Optional[str] = None,
        default_data_type: QuantizationDataType = QuantizationDataType.int,
    ):
        if not quant_scheme:
            old_default = QuantScheme.post_training_tf_enhanced
            new_default = QuantScheme.min_max
            msg = _red(
                f"The default value of 'quant_scheme' has changed from '{old_default}' "
                f"to '{new_default}' since aimet-torch==2.0.0. "
                "If you wish to maintain the legacy default behavior, "
                f"please explicitly pass 'quant_scheme={old_default}'"
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            quant_scheme = new_default

        qmodules = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, BaseQuantizationMixin)
        }
        quantizers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, QuantizerBase)
        }

        if isinstance(model, BaseQuantizationMixin):
            problem = (
                f"the model itself is already a quantized module of type {type(model)}."
            )
        elif isinstance(model, QuantizerBase):
            problem = (
                f"the model itself is already a quantizer object of type {type(model)}."
            )
        elif qmodules:
            problem = f"the model already contains quantized modules: {', '.join(qmodules.keys())}."
        elif quantizers:
            problem = f"the model already contains quantizers: {', '.join(quantizers.keys())}."
        else:
            problem = None

        if problem:
            raise RuntimeError(
                "QuantizationSimModel can only take base models WITHOUT quantized modules or quantizers, "
                "but " + problem
            )

        if not in_place:
            model = copy.deepcopy(model)
            in_place = True

        model = _convert_to_qmodel(model)

        with _register_zero3_forward_hooks(model, use_dummy_params=True):
            # NOTE: Register for the model is pre-partitioned by deepspeed zero3 or zero3-offload.
            #       Pre-partitioned models aren't runnable as-is, but are needed to to be initialized
            #       with `deepspeed.initialize` before running forward pass.
            #       However, `deepspeed.initialize` can only come after quantsim is created, since
            #       quantsim will add additional learnable parameters to the model which also need
            #       to be initialized by deepspeed.
            #       Since quantsim constructor relies on torch.jit tracing which involves running
            #       forward pass of the model, here we register a temporary hook to make
            #       uninitialized but pre-partitioned models runnable.
            super().__init__(
                model,
                dummy_input,
                quant_scheme,
                default_output_bw=default_output_bw,
                default_param_bw=default_param_bw,
                in_place=in_place,
                config_file=config_file,
                default_data_type=default_data_type,
            )

        # Quantization parameters are placed on cpu by default.
        # Move them to cuda device as necessary

        default_device = torch.device("cpu")

        for param_or_buffer in itertools.chain(
            self.model.parameters(), self.model.buffers()
        ):
            if param_or_buffer.device.type != "cpu":
                # Use the first non-cpu device as default device.
                # Default device is necessary for the input/output quantizers of
                # modules without any parameters such as ReLU
                default_device = param_or_buffer.device
                break

        for module in self.model.modules():
            if not isinstance(module, BaseQuantizationMixin):
                continue

            try:
                # Find the device of the first parameter of the orignal module
                param_or_buffer = next(
                    iter(
                        itertools.chain(
                            module.parameters(recurse=False),
                            module.buffers(recurse=False),
                        )
                    )
                )
                device = param_or_buffer.device
            except StopIteration:
                # If the original module has no parameter, use default device
                device = default_device

            # Set quantization parameters to the device of the original module
            module.to(device=device)

        # Class instantiation for supporting sim.onnx.export()
        self._onnx_exporter = QuantizationSimModelOnnxExporter(self)

        self._disable_quantizers_for_reused_modules()

        if self._hw_version is not None:
            # Let input/output of HTP resize ops to share same encoding
            self._propagate_encodings()

    @property
    def onnx(self) -> "QuantizationSimModelOnnxExporter":
        """
        Returns :class:`QuantizationSimModelOnnxExporter <aimet_torch.quantsim.QuantizationSimModelOnnxExporter>`,
        which exports QuantizationSimModel into ONNX model and JSON encoding.
        The returned object can be used to call
        :meth:`QuantizationSimModelOnnxExporter.export <aimet_torch.quantsim.QuantizationSimModelOnnxExporter.export>`,
        which takes the same set of arguments as `torch.onnx.export()`.

        For more details, see :meth:`QuantizationSimModelOnnxExporter.export <aimet_torch.quantsim.QuantizationSimModelOnnxExporter.export>`.

        Example:

            >>> sim.onnx.export(args=(x,), f="model.onnx",
            ...                 input_names=["input"], output_names=["output"],
            ...                 dynamo=False, export_int32_bias=True)
        """
        return self._onnx_exporter

    # pylint: disable=arguments-differ
    @overload
    def compute_encodings(
        self, forward_pass_callback: Callable[[torch.nn.Module], Any]
    ): ...

    T = TypeVar("T")

    # pylint: disable=arguments-differ
    @overload
    def compute_encodings(
        self,
        forward_pass_callback: Callable[[torch.nn.Module, T], Any],
        forward_pass_callback_args: T,
    ): ...

    del T

    # pylint: disable=arguments-differ
    def compute_encodings(
        self, forward_pass_callback, forward_pass_callback_args=_NOT_SPECIFIED
    ):
        r"""
        Computes encodings for all quantizers in the model.

        This API will invoke `forward_pass_callback`, a function written by the user that runs
        forward pass(es) of the quantized model with a small, representative subset of the training dataset.
        By doing so, the quantizers in the quantized model will observe the inputs and initialize
        their quantization encodings according to the observed input statistics.

        This function is overloaded with the following signatures:

        .. function:: compute_encodings(forward_pass_callback)
           :noindex:

           :param forward_pass_callback_: A function that takes a quantized model and runs forward passes
               with a small, representative subset of training dataset
           :type forward_pass_callback_: Callable[[torch.nn.Module], Any]

        .. function:: compute_encodings(forward_pass_callback, forward_pass_callback_args)
           :noindex:

           :param forward_pass_callback_: A function that takes a quantized model and runs forward passes
               with a small, representative subset of training dataset
           :type forward_pass_callback_: Callable[[torch.nn.Module, T], Any]
           :param T forward_pass_callback_args: The second argument to `forward_pass_callback`.

        Example:

            >>> sim = QuantizationSimModel(...)
            >>> _ = sim.model(input) # Can't run forward until quantizer encodings are initialized
            RuntimeError: Failed to run QuantizeDequantize since quantization parameters are not initialized.
            Please initialize the quantization parameters using `compute_encodings()`.
            >>> def run_forward_pass(quantized_model: torch.nn.Module):
            ...     for input in train_dataloader:
            ...         with torch.no_grad():
            ...             _ = quantized_model(input)
            ...
            >>> sim.compute_encodings(run_forward_pass)
            >>> _ = sim.model(input) # Now runs successfully!
        """

        if forward_pass_callback_args is _NOT_SPECIFIED:
            args = (self.model,)
        else:
            args = (self.model, forward_pass_callback_args)

        # Run forward iterations so we can collect statistics to compute the appropriate encodings
        with utils.in_eval_mode(self.model), torch.no_grad():
            with aimet_nn.compute_encodings(self.model):
                _ = forward_pass_callback(*args)

    @deprecated(
        """
Use sim.onnx.export() or aimet_torch.onnx.export() instead. For more information, see"
  - https://qualcomm.github.io/aimet-pages/releases/latest/apiref/torch/quantsim.html#aimet_torch.QuantizationSimModel.onnx"
  - https://qualcomm.github.io/aimet-pages/releases/latest/apiref/torch/onnx.html#aimet_torch.onnx.export""".strip()
    )
    def export(
        self,
        path: str,
        filename_prefix: str,
        dummy_input: Union[torch.Tensor, Tuple],
        *args,
        **kwargs,
    ):
        if quantsim.encoding_version not in ("0.6.1", "1.0.0"):
            msg = "QuantizationSimModel.export only supports encoding version 0.6.1 and 1.0.0."
            if quantsim.encoding_version == "2.0.0":
                msg += (
                    " To export 2.0.0 encoding, please use sim.onnx.export() instead."
                    " For more information on the new export API, see"
                    " https://qualcomm.github.io/aimet-pages/releases/latest/apiref/torch/quantsim.html#aimet_torch.QuantizationSimModel.onnx"
                )
            raise RuntimeError(msg)

        rnn_layers = [
            name
            for name, qmodule in self.named_qmodules()
            if isinstance(qmodule, (QuantizedRNN, QuantizedGRU, QuantizedLSTM))
        ]

        if len(rnn_layers) > 3:
            # Only show up to 3 layer names for readability
            rnn_layers = [*rnn_layers[:3], "..."]

        if rnn_layers:
            raise RuntimeError(
                "QuantizationSimModel.export does not support exporting RNN/GRU/LSTM layers. "
                f"Found the following RNN/GRU/LSTM layers in the model: [{', '.join(rnn_layers)}]. "
                "Please use aimet_torch.onnx.export() or sim.onnx.export() instead, "
                "which supports exporting RNN/GRU/LSTM layers. "
                "For more information, see "
                "https://qualcomm.github.io/aimet-pages/releases/latest/apiref/torch/"
                "quantsim.html#aimet_torch.QuantizationSimModel.onnx"
            )

        if isinstance(dummy_input, torch.Tensor):
            dummy_input = (dummy_input,)

        @torch.no_grad()
        def concretize_block_size(qtzr, inp):
            """
            Fill in block sizes for dimensions with block size -1
            """
            (inp,) = inp
            dims = len(qtzr.block_size)
            input_shape = inp.shape[-dims:]
            scale_shape = qtzr.get_scale().shape[-dims:]
            block_size = qtzr.block_size

            concrete_block_size = tuple(
                inp_size // scale_size if blk_size == -1 else blk_size
                for inp_size, scale_size, blk_size in zip(
                    input_shape, scale_shape, block_size
                )
            )
            ctx = patch_attr(qtzr, "block_size", concrete_block_size)
            stack.enter_context(ctx)

        handles = []

        try:
            with contextlib.ExitStack() as stack:
                for qtzr in self.model.modules():
                    if not isinstance(qtzr, AffineQuantizerBase):
                        continue

                    if qtzr.block_size and any(size == -1 for size in qtzr.block_size):
                        h = qtzr.register_forward_pre_hook(concretize_block_size)
                        handles.append(h)

                if handles:
                    with utils.in_eval_mode(self.model), torch.no_grad():
                        _ = self.model(*dummy_input)

                # TODO
                # stack.enter_context(self._concretize_int32_bias_quantizers(dummy_input))
                return super().export(
                    path, filename_prefix, dummy_input, *args, **kwargs
                )

        finally:
            for h in handles:
                h.remove()

    @classmethod
    def _export_encodings_to_1_0_0(
        cls,
        path: str,
        filename_prefix: str,
        tensor_to_activation_encodings: Dict[str, List],
        tensor_to_param_encodings: Dict[str, List],
        tensor_to_quantizer_map: Dict,
        excluded_layer_names: List[str],
        onnx_model: Optional["onnx.ModelProto"],
        quantizer_args: Dict,
    ):
        """
        Export encodings using format version 1.0.0.

        :param path: Path to save encodings
        :param filename_prefix: Filename to save encodings with
        :param tensor_to_activation_encodings: Dictionary of activation encodings which maps onnx attribute to encodings
        :param tensor_to_param_encodings: Dictionary of param encodings
        :param tensor_to_quantizer_map: Dictionary mapping tensor names to quantizers
        :param excluded_layer_names: List of names of layers that have been excluded from quantization
        :param quantizer_args: Arguments to top leve quantsim
        """
        if onnx_model is not None:
            tensor_to_encoding_map = {
                name: quantizer.get_encodings()
                for name, quantizer in tensor_to_quantizer_map.items()
                if quantizer
            }
            tensor_to_encoding_dict_map = {
                name: encoding.to_qnn_encoding_dict("2.0.0")
                for name, encoding in tensor_to_encoding_map.items()
                if isinstance(encoding, AffineEncoding)
            }
            derived_encodings = _derive_const_rescale_op_output_encodings(
                onnx_model, tensor_to_encoding_dict_map
            )
            # pylint: disable=protected-access
            derived_encodings = {
                name: AffineEncoding._from_qnn_encoding_dict(
                    encoding_dict
                ).to_qnn_encoding_dict("1.0.0")
                for name, encoding_dict in derived_encodings.items()
                if name not in tensor_to_activation_encodings
                and name not in tensor_to_param_encodings
            }
            tensor_to_activation_encodings.update(derived_encodings)
        super()._export_encodings_to_1_0_0(
            path,
            filename_prefix,
            tensor_to_activation_encodings,
            tensor_to_param_encodings,
            tensor_to_quantizer_map,
            excluded_layer_names,
            onnx_model,
            quantizer_args,
        )

    def set_percentile_value(self, percentile_value: float):
        """
        Set the percentile value to be used while computing encodings
        """
        self._percentile_value = percentile_value
        for module in self.model.modules():
            if isinstance(module, QuantizerBase):
                if isinstance(module.encoding_analyzer, PercentileEncodingAnalyzer):
                    module.encoding_analyzer.set_percentile(percentile_value)

    def __str__(self):
        stream = io.StringIO(newline="\n")
        stream.write("-------------------------\n")
        stream.write("Quantized Model Report\n")
        stream.write("-------------------------\n")
        stream.write(f"{self.model}\n")
        return stream.getvalue()

    def exclude_param_from_quantization(self, param_name_to_exclude: str):
        """
        Excludes all parameters matching 'param_name' from quantization
        :param param_name_to_exclude: Name of the parameter to exclude
        :return: None
        """
        super().exclude_param_from_quantization(param_name_to_exclude)
        for module in self.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                if param_name_to_exclude in module.param_quantizers:
                    module.param_quantizers[param_name_to_exclude] = None

    # pylint: disable=missing-function-docstring
    @staticmethod
    def compute_layer_encodings_for_sim(sim: "QuantizationSimModel"):
        raise NotImplementedError(
            "QuantizationSimModel.compute_layer_encodings_for_sim has been removed."
        )

    # pylint: disable=missing-function-docstring, unused-argument
    @staticmethod
    def prepare_sim_for_compute_encodings(sim: "QuantizationSimModel"):
        logger.warning(
            "QuantizationSimModel.prepare_sim_for_compute_encodings has been deprecated and is no longer necessary. "
            "Any calls can be safely removed."
        )

    # pylint: disable=missing-function-docstring
    @classmethod
    def set_mode_for_recurrent_module(cls, layer, name: str):
        raise NotImplementedError(
            "QuantizationSimModel.set_mode_for_recurrent_module has been removed."
        )

    @staticmethod
    def save_model_with_embedded_quantization_nodes(
        sim_model,
        path: str,
        filename_prefix: str,
        dummy_input,
        onnx_export_args=None,
        export_to_torchscript=False,
        is_conditional=False,
    ):
        raise NotImplementedError(
            "QuantizationSimModel.save_model_with_embedded_quantization_nodes has been removed."
        )

    @staticmethod
    def _replace_quantization_wrapper_with_native_torch_quantization_nodes(
        quant_sim_model, device: torch.device
    ):
        raise NotImplementedError()

    @classmethod
    @torch.no_grad()
    def _apply_qdq_to_model_parameters(cls, model: torch.nn.Module):
        """
        Applies quant-dequant to the parameters of a PyTorch model
        to avoid rounding error during weight quantization.

        :param model: The PyTorch model whose parameters will be quant-dequantized.
        """
        stack = contextlib.ExitStack()
        try:
            for module in model.modules():
                if not isinstance(module, BaseQuantizationMixin):
                    continue

                encodings = {
                    param_name: getattr(param, "encoding", None)
                    for param_name, param in module.named_parameters(recurse=False)
                }

                # pylint: disable=protected-access
                stack.enter_context(module._patch_quantized_parameters())
                if isinstance(module, QuantizationMixin):
                    stack.enter_context(module._patch_dequantized_parameters())
                stack.enter_context(cls._update_parameters_by_attr(module))

                # Restore the original encodings which might have been
                # overwritten by _patch_quantized_parameters
                for param_name, encoding in encodings.items():
                    qparam = getattr(module, param_name)
                    if isinstance(qparam, QuantizedTensorBase) and encoding:
                        qparam.encoding = encoding
                    else:
                        setattr(
                            module,
                            param_name,
                            torch.nn.Parameter(qparam.as_subclass(torch.Tensor)),
                        )
            return stack
        except Exception:  # pylint: disable=broad-exception-caught
            # In case of any exception, make sure to restore
            # the original parameters before raising the exception
            stack.close()
            raise

    def qmodules(self):
        """
        Generator that yields all quantized modules

        Example:

            >>> for qmodule in sim.qmodules():
            ...     print(qmodule)
        """
        yield from super().qmodules()

    def named_qmodules(self):
        """
        Generator that yields all quantized modules along with their names

        Example:

            >>> for name, qmodule in sim.named_qmodules():
            ...     print(f"{name}: {type(qmodule)}")
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (BaseQuantizationMixin, _V2LazyQuantizeWrapper)):
                yield name, module

    def quantizers(self):
        """
        Generator that yields all quantizers

        Example:

            >>> for quantizer in sim.quantizers():
            ...     print(quantizer)
        """
        yield from (module for _, module in self.named_quantizers())

    def named_quantizers(self):
        """
        Generator that yields all quantizers along with their names

        Example:

            >>> for name, quantizer in sim.named_quantizers():
            ...     print(f"{name}: {type(quantizer)}")
        """
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizerBase):
                yield name, module

    def quantizer_parameters(self):
        """
        Generator that yields all quantization parameters

        Example:

            >>> for param in sim.quantizer_parameters():
            ...     print(param.size())
        """
        yield from (param for _, param in self.named_quantizer_parameters())

    def named_quantizer_parameters(self):
        """
        Generator that yields all quantization parameters along with their names

        Example:

            >>> for name, param in sim.named_quantizer_parameters():
            ...     print(f"{name}: {param.size()}")
        """
        memo = set()
        for module_name, module in self.named_quantizers():
            for param_name, param in module.named_parameters(recurse=False):
                if param in memo:
                    continue
                memo.add(param)
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                yield full_name, param

    def quantizer_state_dict(self):
        """
        Returns light-weight state dict that only consists of quantizer states.

        Example:

            >>> state_dict = sim.quantizer_state_dict()
            >>> sim2 = QuantizationSimModel(...)
            >>> sim2.model.load_state_dict(state_dict, strict=False)
            >>> assert torch.equal(sim.model(x), sim2.model(x))
        """
        state_dict = {}

        for name, qtzr in self.named_quantizers():
            state_dict.update(qtzr.state_dict(prefix=f"{name}." if name else ""))

        return state_dict

    @deprecated(f"Use {named_qmodules.__qualname__} instead.")
    def quant_wrappers(self):  # pylint: disable=missing-docstring
        return self.named_qmodules()

    def _add_quantization_wrappers(self, module, num_inout_tensors, default_data_type):
        visited = set()

        def wrap_children(parent: torch.nn.Module):
            for name, child in parent.named_children():
                if not isinstance(child, BaseQuantizationMixin):
                    continue

                if child in visited:
                    continue

                visited.add(child)

                child_wrapper = self._create_quantizer_module(
                    child, num_inout_tensors, default_data_type
                )
                setattr(parent, name, child_wrapper)
                visited.add(child_wrapper)

        module.apply(wrap_children)

    def _realize_quant_wrappers_in_model(self, model: torch.nn.Module):
        for name, child in model.named_children():
            if isinstance(child, _V2LazyQuantizeWrapper):
                child = child.realize()
                setattr(model, name, child)
            self._realize_quant_wrappers_in_model(child)

    def _create_quantizer_module(
        self,
        module_to_quantize: torch.nn.Module,
        num_inout_tensors,
        data_type: QuantizationDataType,
    ) -> torch.nn.Module:
        """Instantiates wrapper based on quant scheme"""
        # We lookup the number of input and output tensors already determined
        # Special case, we are adding a wrapper for a module not in the forward pass: Use default of 1, 1
        num_in_tensors, num_out_tensors = num_inout_tensors.get(
            module_to_quantize, (1, 1)
        )
        quantized_module = _V2LazyQuantizeWrapper(
            module_to_quantize,
            self._default_param_bw,
            self._default_output_bw,
            "nearest",
            self._quant_scheme,
            num_inputs=num_in_tensors,
            num_outputs=num_out_tensors,
            data_type=data_type,
        )
        return quantized_module

    @classmethod
    def _remove_quantization_wrappers(cls, starting_module, list_of_modules_to_exclude):
        """
        Recursively remove quantization wrappers from all appropriate modules starting with a given module
        :param starting_module: Module to recursive search downstream from
        :param list_of_modules_to_exclude: List of torch modules to remove quantization wrappers from (if present)
        :return: None
        """
        for name, module in starting_module.named_children():
            if module in list_of_modules_to_exclude:
                if isinstance(module, BaseQuantizationMixin):
                    orig_module = module.get_original_module()
                    setattr(starting_module, name, orig_module)
                    module = orig_module
            # Recursively call children modules if present
            if not utils.is_leaf_module(module):
                cls._remove_quantization_wrappers(module, list_of_modules_to_exclude)

    def fold_param_quantizers(self):
        """
        Fold parameter quantizers into their associated parameters to accelerate inference.

        Example:

          >>> sim = QuantizationSimModel(...)
          >>> type(sim.model[0].weight)
          <class 'torch.nn.parameter.Parameter'>
          >>> sim.model[0]
          QuantizedLinear(
            in_features=10, out_features=10, bias=True
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
          )
          >>> sim.fold_param_quantizers()
          >>> type(sim.model[0].weight)
          <class 'aimet_torch.quantization.tensor.DequantizedTensor'>
          >>> sim.model[0]
          QuantizedLinear(
            in_features=10, out_features=10, bias=True
            (param_quantizers): ModuleDict(
              (weight): None
              (bias): None
            )
          )
        """
        for qmodule in self.qmodules():
            qmodule.fold_param_quantizers()

    def _propagate_encodings(self):
        # pylint: disable=import-outside-toplevel
        if not self.connected_graph.is_safe():
            return

        from aimet_torch.onnx_utils import map_torch_types_to_onnx
        from aimet_torch.experimental.quantsim_utils import (
            propagate_output_encodings,
        )

        htp_interpolation_ops = set()

        for qmodule in self.qmodules():
            orig_module_type = type(qmodule.get_original_module())
            onnx_op_types = map_torch_types_to_onnx.get(orig_module_type)

            if not onnx_op_types:
                continue

            # Output encoding back-propagation only works when output quantizer exists
            if (
                len(qmodule.output_quantizers) != 1
                or qmodule.output_quantizers[0] is None
            ):
                continue

            if all(_is_htp_interpolation_op(op_type) for op_type in onnx_op_types):
                htp_interpolation_ops.add(qmodule)

        propagate_output_encodings(
            self,
            lambda module: module in htp_interpolation_ops
            or (isinstance(module, QuantizedReLU) and module.output_quantizers[0]),
        )

    def _disable_quantizers_for_constant_rescale_ops(self):
        """
        Disables quantizers for Div/Mul ops with constant scalar scaling factor
        """
        for op in self.connected_graph.ordered_ops:
            wrapper = self._get_qmodule(op)
            if not isinstance(wrapper, _V2LazyQuantizeWrapper):
                continue
            original_module = wrapper.get_original_module()
            if isinstance(original_module, (custom.Divide, custom.Multiply)):
                if len(op.inputs) != 2:
                    continue
                inp_idx, scale_idx = (0, 1)
                if (
                    isinstance(original_module, custom.Multiply)
                    and op.inputs[scale_idx].constant_value is None
                ):
                    inp_idx, scale_idx = scale_idx, inp_idx
                scalar_value = op.inputs[scale_idx].constant_value
                if scalar_value is None:
                    continue
                if op.inputs[scale_idx].numel != 1:
                    continue
                if isinstance(scalar_value, torch.nn.Parameter):
                    continue
                if isinstance(scalar_value, torch.Tensor):
                    scalar_value = scalar_value.item()
                if scalar_value is None:
                    continue
                if scalar_value <= 0:
                    continue
                if math.isnan(scalar_value) or math.isinf(scalar_value):
                    continue
                inp_quantizer = wrapper.input_quantizers[inp_idx]
                input_op = op.inputs[inp_idx].producer
                inp_quantizer = self._get_target_quantizer(inp_quantizer, input_op)
                if inp_quantizer and inp_quantizer.enabled:
                    wrapper.input_quantizers[scale_idx].enabled = False
                    wrapper.output_quantizers[0].enabled = False

    def _disable_quantizers_for_reused_modules(self):
        """
        Disables quantizers for stateless modules that are used multiple times in the model.

        Quantizing reused modules can lead to accuracy degradation as it forces tied encodings.
        """
        module_count = defaultdict(int)
        for op in self.connected_graph.ordered_ops:
            qmodule = self._get_qmodule(op)
            if qmodule:
                module_count[qmodule] += 1

        for name, qmodule in self.named_qmodules():
            if module_count.get(qmodule, 0) <= 1:
                continue
            if qmodule.param_quantizers:
                continue
            disabled = False
            for quantizer_list in (qmodule.input_quantizers, qmodule.output_quantizers):
                for idx, quantizer in enumerate(quantizer_list):
                    if quantizer is None:
                        continue
                    # If encodings are fully fixed, leave enabled
                    if quantizer.is_initialized() and all(
                        not quantizer.is_overwrite_allowed(param_name)
                        for param_name, _ in quantizer.named_parameters(recurse=False)
                    ):
                        continue
                    quantizer_list[idx] = None
                    disabled = True

            if disabled:
                logger.warning(
                    "Module %s is used %d times in the model. Disabling quantizers for this module "
                    "to avoid accuracy drop due to re-used quantization parameters.",
                    name,
                    module_count[qmodule],
                )


class QuantizationSimModelOnnxExporter:
    """
    Helper class for exporting quantized models to ONNX format.
    """

    def __init__(self, sim: QuantizationSimModel):
        self.sim = sim

    @docstring(
        f"""
    Export QuantizationSimModel into ONNX model and JSON encoding.
    This function takes the same set of arguments as `torch.onnx.export()`_.

    Args:
        args: Same as `torch.onnx.export()`
        f: Same as `torch.onnx.export()`
        encoding_version (str, optional):
            Version of the encoding format to use. (default: {quantsim.encoding_version})
            Supported versions are: {sorted(list(quantsim.VALID_ENCODING_VERSIONS))}
        export_int32_bias (bool, optional):
            If true, generate and export int32 bias encoding on the fly.
            Default: `True` if encoding version is 2.0.0 or higher, otherwise `False`.
        force_activation_as (str, optional):
            Force representing quantized activations as signed or unsigned integers.
            This argument is only applicable for encoding version 2.0.0.
            For versions 0.6.1 and 1.0.0, this argument is ignored. (uefault: `"unsigned"`)
        **kwargs: Same as `torch.onnx.export()`

    .. _torch.onnx.export(): https://docs.pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export

    .. note::
        See also :func:`aimet_torch.onnx.export`, an equivalent API that
        exports a single ONNX graph with quantization encodings
        embedded in QuantizeLinear and DequantizeLinear nodes

    Example:

        >>> sim.onnx.export(args=(x,), f="model.onnx",
        ...                 input_names=["input"], output_names=["output"],
        ...                 dynamo=False, export_int32_bias=True,
        ...                 encoding_version="2.0.0")
    """
    )
    @torch.no_grad()
    def export(
        self,
        args: Union[Tuple[Any, ...], torch.Tensor],
        f: Union[str, io.BytesIO],
        *,
        encoding_version: Optional[str] = None,
        export_int32_bias: Optional[bool] = None,
        force_activation_as: Literal["unsigned"]
        | Literal["signed"]
        | None = "unsigned",
        **kwargs,
    ):
        encoding_version = encoding_version or quantsim.encoding_version

        if encoding_version not in quantsim.VALID_ENCODING_VERSIONS:
            raise ValueError(
                f"Unsupported encoding_version '{encoding_version}'. "
                f"Supported versions are: {quantsim.VALID_ENCODING_VERSIONS}"
            )

        if export_int32_bias is None:
            export_int32_bias = version.parse(encoding_version) >= version.parse(
                "2.0.0"
            )

        from aimet_torch.onnx import (
            _temporarily_convert_activation_to,
            _check_unsupported_args,
            _concretize_int32_bias_quantizers,
            _remove_fp16_quantizers,
            _remove_fp16_quantized_parameters,
            _to_onnx,
            _temporarily_unfold_param_quantizers,
        )

        _check_unsupported_args(self.sim.model, force_activation_as, kwargs)

        with contextlib.ExitStack() as stack:
            # Unfold all param quantizers to incorporate QuantizeLinear/DequantizeLinear
            # of those parameters in tracing time
            stack.enter_context(_temporarily_unfold_param_quantizers(self.sim.model))

            if export_int32_bias:
                # Temoprarily instantiate int32 bias quantizers
                stack.enter_context(
                    _concretize_int32_bias_quantizers(
                        self.sim.model, args, kwargs.get("kwargs")
                    )
                )

            # Export quantize-dequantized weight
            # pylint: disable=protected-access
            if force_activation_as in ("unsigned", "signed"):
                signed = force_activation_as == "signed"
                stack.enter_context(
                    _temporarily_convert_activation_to(self.sim.model, signed=signed)
                )

            stack.enter_context(self.sim._apply_qdq_to_model_parameters(self.sim.model))

            # Remove [b]float16 quantizers
            stack.enter_context(_remove_fp16_quantizers(self.sim.model))
            stack.enter_context(_remove_fp16_quantized_parameters(self.sim.model))

            onnx_model, tensor_to_encoding_map = _to_onnx(
                self.sim.model, args, f, **kwargs
            )

        onnx.save(
            onnx_model,
            f,
            save_as_external_data=_onnx_model_size_larger_than_max_protobuf(onnx_model),
        )
        constants = _get_all_constants(onnx_model)
        encodings_dict = self._to_json(
            tensor_to_encoding_map, encoding_version, constants
        )

        # export weight encodings to output json file
        onnx_file_path = str(f)
        encoding_file_path = os.path.splitext(onnx_file_path)[0] + ".encodings"
        with open(encoding_file_path, "w", encoding="utf-8") as encoding_file:
            json.dump(encodings_dict, encoding_file, indent=2)

    def _to_json(
        self,
        tensor_to_encoding_map: Mapping[str, Tuple[EncodingBase, bool]],
        encoding_version: str,
        constants: Mapping[str, onnx.TensorProto],
    ):
        qnn_encodings = {
            name: (encoding.to_qnn_encoding_dict(encoding_version), is_param)
            for name, (encoding, is_param) in tensor_to_encoding_map.items()
        }

        encodings_dict: Mapping[str, Any]
        encodings_dict = {
            "producer": {
                "package": "aimet-torch",
                "version": aimet_torch.__version__,
            },
            "version": encoding_version,
        }

        if encoding_version >= "2.0.0":
            encodings_dict.update(
                {
                    "encodings": [
                        {"name": name, **qnn_encoding}
                        for name, (qnn_encoding, _) in qnn_encodings.items()
                        if qnn_encoding
                    ]
                }
            )

            for enc in encodings_dict["encodings"]:
                name = enc["name"]
                axis = enc.get("axis", None)

                # Convert to positive index. Not strictly necessary; just for convenience
                if axis is not None and axis < 0 and name in constants:
                    enc["axis"] = axis + len(constants[name].dims)
        else:
            if encoding_version >= "1.0.0":
                param_encodings = [
                    {"name": name, **qnn_encoding}
                    for name, (qnn_encoding, is_param) in qnn_encodings.items()
                    if qnn_encoding and is_param
                ]
                activation_encodings = [
                    {"name": name, **qnn_encoding}
                    for name, (qnn_encoding, is_param) in qnn_encodings.items()
                    if qnn_encoding and not is_param
                ]
            else:
                param_encodings = {
                    name: qnn_encoding
                    for name, (qnn_encoding, is_param) in qnn_encodings.items()
                    if qnn_encoding and is_param
                }
                activation_encodings = {
                    name: qnn_encoding
                    for name, (qnn_encoding, is_param) in qnn_encodings.items()
                    if qnn_encoding and not is_param
                }

            encodings_dict.update(
                {
                    "activation_encodings": activation_encodings,
                    "param_encodings": param_encodings,
                    "excluded_layers": self.sim._excluded_layer_names,  # pylint: disable=protected-access
                }
            )

            if self.sim.quant_args:
                encodings_dict.update({"quantizer_args": self.sim.quant_args})

        return encodings_dict


@deprecated("""
Use QuantizationSimModel.load_encodings with the following keyword arguments instead:
```
sim.load_encodings(encoding_path
                   strict=True,
                   partial=False,
                   requires_grad=None,
                   allow_overwrite=None)
```
""")
def load_encodings_to_sim(
    quant_sim_model: _QuantizationSimModelBase, pytorch_encoding_path: str
):
    """
    Loads the saved encodings to quant sim model. The encoding filename to load should end in _torch.encodings,
    generated as part of quantsim export.

    :param quant_sim_model: Quantized model to load encodings for. Note: The model configuration should be the same as
        when encodings were exported.
    :param pytorch_encoding_path: Path of the encodings file to load.
    """
    quant_sim_model.load_encodings(
        pytorch_encoding_path,
        strict=True,
        partial=False,
        requires_grad=None,
        allow_overwrite=None,
    )


@deprecated(r"""
Use aimet_torch.nn.compute_encodings contextmanager on each sim.model instead. For example:
```
with torch.no_grad(), \
        aimet_torch.nn.compute_encodings(sim_0.model), \
        aimet_torch.nn.compute_encodings(sim_1.model), \
        aimet_torch.nn.compute_encodings(sim_2.model):
    # Run forward pass with calibration dataset
```
""")
def compute_encodings_for_sims(
    sim_list: Sequence[QuantizationSimModel],
    forward_pass_callback: Callable,
    forward_pass_callback_args: Any,
):
    """
    Compute encodings for a list of QuantSims.

    :param sim_list: List of QuantSims to compute encodings for.
    :param forward_pass_callback: A callback function that simply runs forward passes on the models. This callback
        function should use representative data for the forward pass, so the calculated encodings work for all
        data samples. This callback internally chooses the number of data samples it wants to use for calculating
        encodings.
        The callback expects exactly two inputs:
            - List of models which are involved in the forward pass. The models are taken directly from calling
            sim.model for each sim in sim_list, passed in the same order in which the sims appear in sim_list.
            - Forward pass callback args
    :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
        the user to determine the type of this parameter. E.g. could be simply an integer representing the number
        of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
        If set to None, forward_pass_callback will be invoked with no parameters.
    """
    ctx_managers = [torch.no_grad()]
    for sim in sim_list:
        ctx_managers.append(utils.in_eval_mode(sim.model))
        ctx_managers.append(aimet_nn.compute_encodings(sim.model))

    with contextlib.ExitStack() as stack:
        for mgr in ctx_managers:
            stack.enter_context(mgr)
        _ = forward_pass_callback(
            [sim.model for sim in sim_list], forward_pass_callback_args
        )
