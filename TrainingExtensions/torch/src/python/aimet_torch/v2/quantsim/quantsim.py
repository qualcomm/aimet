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
""" Top level API for performing quantization simulation of a pytorch model """

import copy
from typing import Union, Tuple, Optional, TypeVar, Any, Callable, overload
import warnings
import itertools
import io
import contextlib
import torch

from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_torch.v1.quantsim import QuantizationSimModel as V1QuantizationSimModel, logger, unquantizable_modules, quantized_modules
from aimet_torch.v2 import nn as aimet_nn
from aimet_torch.v2.nn import BaseQuantizationMixin, QuantizationMixin
from aimet_torch.v2.nn.fake_quant import _legacy_impl
from aimet_torch.quantsim_config.builder import LazyQuantizeWrapper
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine import AffineQuantizerBase
from aimet_torch.v2.quantization.encoding_analyzer import PercentileEncodingAnalyzer
from aimet_torch.v2.utils import patch_attr
from aimet_torch import utils
from aimet_torch.utils import deprecated, _red
from aimet_torch.v2.deepspeed_utils import _register_zero3_forward_hooks


unquantizable_modules = (QuantizerBase, *unquantizable_modules)
quantized_modules = (BaseQuantizationMixin, *quantized_modules)
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


def _convert_to_qmodule(module: torch.nn.Module):
    """
    Helper function to convert all modules to quantized aimet.nn modules.
    """
    if not isinstance(module, (*quantized_modules, *unquantizable_modules, *containers)):
        try:
            module = QuantizationMixin.from_module(module)
        except RuntimeError as e:
            try:
                module = _legacy_impl.FakeQuantizationMixin.from_module(module)
            except RuntimeError:
                if not tuple(module.children()):
                    raise e # pylint: disable=raise-missing-from

    for name, child in module.named_children():
        setattr(module, name, _convert_to_qmodule(child))

    return module


class QuantizationSimModel(V1QuantizationSimModel):
    """
    Overriden QuantizationSimModel that does off-target quantization simulation using v2 quantsim blocks.
    """
    def __init__(self, # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
                 model: torch.nn.Module,
                 dummy_input: Union[torch.Tensor, Tuple],
                 quant_scheme: Union[str, QuantScheme] = None, # NOTE: Planned to be deprecated
                 rounding_mode: Optional[str] = None, # NOTE: Planned to be deprecated
                 default_output_bw: int = 8,
                 default_param_bw: int = 8,
                 in_place: bool = False,
                 config_file: Optional[str] = None,
                 default_data_type: QuantizationDataType = QuantizationDataType.int):
        if not quant_scheme:
            old_default = QuantScheme.post_training_tf_enhanced
            new_default = QuantScheme.training_range_learning_with_tf_init
            msg = _red(f"The default value of 'quant_scheme' will change from '{old_default}' "
                       f"to '{new_default}' in the later versions. "
                       "If you wish to maintain the legacy behavior in the future, "
                       f"please explicitly pass 'quant_scheme={old_default}'")
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            quant_scheme = old_default

        if rounding_mode:
            if rounding_mode == 'nearest':
                warnings.warn(_red("Passing rounding_mode='nearest' is no longer needed "\
                                   "and will be deprecated soon in the later versions."),
                              DeprecationWarning, stacklevel=2)
            else:
                raise TypeError("'rounding_mode' parameter is no longer supported.")

        qmodules = {
            name: module for name, module in model.named_modules()
            if isinstance(module, BaseQuantizationMixin)
        }
        quantizers = {
            name: module for name, module in model.named_modules()
            if isinstance(module, QuantizerBase)
        }

        if isinstance(model, BaseQuantizationMixin):
            problem = f"the model itself is already a quantized module of type {type(model)}."
        elif isinstance(model, QuantizerBase):
            problem = f"the model itself is already a quantizer object of type {type(model)}."
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

        model = _convert_to_qmodule(model)

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
            super().__init__(model, dummy_input, quant_scheme,
                             rounding_mode='nearest',
                             default_output_bw=default_output_bw,
                             default_param_bw=default_param_bw,
                             in_place=in_place,
                             config_file=config_file,
                             default_data_type=default_data_type)

        # Quantization parameters are placed on cpu by default.
        # Move them to cuda device as necessary

        default_device = torch.device('cpu')

        for param_or_buffer in itertools.chain(self.model.parameters(), self.model.buffers()):
            if param_or_buffer.device.type != 'cpu':
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
                param_or_buffer = next(iter(itertools.chain(module.parameters(recurse=False),
                                                            module.buffers(recurse=False))))
                device = param_or_buffer.device
            except StopIteration:
                # If the original module has no parameter, use default device
                device = default_device

            # Set quantization parameters to the device of the original module
            module.to(device=device)


    @overload
    def compute_encodings(self, forward_pass_callback: Callable[[torch.nn.Module], Any]): # pylint: disable=arguments-differ
        ...

    T = TypeVar('T')

    @overload
    def compute_encodings(self,
                          forward_pass_callback: Callable[[torch.nn.Module, T], Any],
                          forward_pass_callback_args: T):
        ...

    del T

    def compute_encodings(self, forward_pass_callback, forward_pass_callback_args=_NOT_SPECIFIED):
        """
        Computes encodings for all quantization sim nodes in the model. It is also used to find initial encodings for
        Range Learning

        :param forward_pass_callback: A callback function that simply runs forward passes on the model. This callback
            function should use representative data for the forward pass, so the calculated encodings work for all
            data samples. This callback internally chooses the number of data samples it wants to use for calculating
            encodings.
        :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
            the user to determine the type of this parameter. E.g. could be simply an integer representing the number
            of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
            If set to None, forward_pass_callback will be invoked with no parameters.
        :return: None

        """
        if forward_pass_callback_args is _NOT_SPECIFIED:
            args = (self.model,)
        else:
            args = (self.model, forward_pass_callback_args)

        # Run forward iterations so we can collect statistics to compute the appropriate encodings
        with utils.in_eval_mode(self.model), torch.no_grad():
            with aimet_nn.compute_encodings(self.model):
                _ = forward_pass_callback(*args)

    def export(self, path: str, filename_prefix: str, dummy_input: Union[torch.Tensor, Tuple],
               *args, **kwargs):
        if isinstance(dummy_input, torch.Tensor):
            dummy_input = (dummy_input,)

        @torch.no_grad()
        def concretize_block_size(qtzr, inp):
            """
            Fill in block sizes for dimensions with block size -1
            """
            inp, = inp
            dims = len(qtzr.block_size)
            input_shape = inp.shape[-dims:]
            scale_shape = qtzr.get_scale().shape[-dims:]
            block_size = qtzr.block_size

            concrete_block_size = tuple(inp_size//scale_size if blk_size == -1 else blk_size
                                        for inp_size, scale_size, blk_size
                                        in zip(input_shape, scale_shape, block_size))
            ctx = patch_attr(qtzr, 'block_size', concrete_block_size)
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

                return super().export(path, filename_prefix, dummy_input, *args, **kwargs)

        finally:
            for h in handles:
                h.remove()

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
        stream = io.StringIO(newline='\n')
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

    @staticmethod
    def compute_layer_encodings_for_sim(sim: 'QuantizationSimModel'):
        raise NotImplementedError("QuantizationSimModel.compute_layer_encodings_for_sim has been removed.")

    @staticmethod
    def prepare_sim_for_compute_encodings(sim: 'QuantizationSimModel'):
        logger.warning("QuantizationSimModel.prepare_sim_for_compute_encodings has been deprecated and is no longer necessary. "
                       "Any calls can be safely removed.")

    @classmethod
    def set_mode_for_recurrent_module(cls, layer, name: str):
        raise NotImplementedError("QuantizationSimModel.set_mode_for_recurrent_module has been removed.")

    @staticmethod
    def save_model_with_embedded_quantization_nodes(sim_model, path: str, filename_prefix: str,
                                                    dummy_input, onnx_export_args=None,
                                                    export_to_torchscript=False, is_conditional=False):
        raise NotImplementedError("QuantizationSimModel.save_model_with_embedded_quantization_nodes has been removed.")

    @staticmethod
    def _replace_quantization_wrapper_with_native_torch_quantization_nodes(quant_sim_model, device: torch.device):
        raise NotImplementedError()

    @classmethod
    @torch.no_grad()
    def _apply_qdq_to_model_parameters(cls, model: torch.nn.Module):
        """
        Applies quant-dequant to the parameters of a PyTorch model
        to avoid rounding error during weight quantization.

        :param model: The PyTorch model whose parameters will be quant-dequantized.
        """
        for module in model.modules():
            if isinstance(module, BaseQuantizationMixin):
                # pylint: disable=protected-access
                module._patch_quantized_parameters()
                if isinstance(module, QuantizationMixin):
                    module._patch_dequantized_parameters()
                cls._update_parameters_by_attr(module)

    @deprecated(f'Use {V1QuantizationSimModel.named_qmodules.__qualname__} instead.')
    def quant_wrappers(self): # pylint: disable=missing-docstring
        return super().quant_wrappers()

    # Overrides V1QuantizationSimModel._add_quantization_wrappers
    def _add_quantization_wrappers(self, module, num_inout_tensors, default_data_type):
        # pylint: disable=protected-access
        for name, child in module.named_children():
            if isinstance(child, BaseQuantizationMixin):
                child_wrapper = self._create_quantizer_module(child, num_inout_tensors, default_data_type)
                setattr(module, name, child_wrapper)
                child = child_wrapper._module_to_wrap
            self._add_quantization_wrappers(child, num_inout_tensors, default_data_type)

    # Overrides V1QuantizationSimModel._realize_quant_wrappers_in_model
    def _realize_quant_wrappers_in_model(self, model: torch.nn.Module):
        for name, child in model.named_children():
            if isinstance(child, LazyQuantizeWrapper):
                child = child.realize_v2_wrapper()
                setattr(model, name, child)
            self._realize_quant_wrappers_in_model(child)
