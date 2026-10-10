# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
from typing import Optional, List, Tuple, Dict
import math

import torch
from aimet_onnx.common.quantsim import _get_minimum_scale
from aimet_onnx.adascale.utils import (
    derive_symmetric_qmin_qmax,
    validate_arguments,
    is_numerically_stable,
    is_grid_representable,
    reshape_tensor_for_blocks,
    get_encoding_shape_with_blocks,
    reduce,
    get_symmetric_offset,
)


class RoundManual(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, x):  # pylint: disable=arguments-differ, abstract-method, unused-argument
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ, unused-argument
        return grad_output


class WeightQdq(torch.nn.Module):
    """
    Light weight QDQ module for weight quantization (supports symmetric mode only)
    """

    def __init__(
        self,
        weight_tensor: torch.Tensor,
        enc_shape: tuple,
        bitwidth: int,
        block_size=None,
        zero_point_shift=None,
    ):
        super().__init__()
        self.shape = tuple(enc_shape)
        self.bitwidth = bitwidth
        self.qmin, self.qmax = derive_symmetric_qmin_qmax(bitwidth=bitwidth)
        self.block_size = block_size
        self.zero_point_shift = zero_point_shift or 0.0

        min_tensor, max_tensor = self.compute_min_max_tensors(
            weight_tensor, self.shape, self._get_num_steps()
        )

        self.register_parameter("min", torch.nn.Parameter(min_tensor))
        self.register_parameter("max", torch.nn.Parameter(max_tensor))

    @staticmethod
    def quantize_dequantize(
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
        qmin: int,
        qmax: int,
        block_size: Optional[List] = None,
        zero_point_shift: float = 0.0,
    ) -> torch.Tensor:
        """
        Performs differentiable quantize-dequantize given scale, offset, and quantization range.

        :param tensor: Tensor to quantize
        :param scale: Scale factor for quantization
        :param offset: Offset value for quantization
        :param qmin: Minimum value of the quantization range
        :param qmax: Maximum value of the quantization range
        :param block_size: Block sizes per dimension
        :param zero_point_shift: Shift tensor by an amount proportional to scale during quantize dequantize
        """
        validate_arguments(tensor, scale, qmin, qmax, block_size)

        output_dtype = internal_dtype = tensor.dtype

        if not is_numerically_stable(internal_dtype, qmin, qmax):
            internal_dtype = torch.float32
            if not is_numerically_stable(internal_dtype, qmin, qmax):
                internal_dtype = torch.float64

        if not is_grid_representable(internal_dtype, qmin, qmax):
            msg = f"{internal_dtype} is unable to represent quantized output of range [{qmin}, {qmax}]."
            raise RuntimeError(msg)

        orig_tensor_shape = tensor.shape
        tensor = reshape_tensor_for_blocks(tensor, scale.shape, block_size)
        scale = scale.view(get_encoding_shape_with_blocks(scale.shape, block_size)).to(
            internal_dtype
        )
        offset = offset.view(get_encoding_shape_with_blocks(offset.shape, block_size))
        shifted_tensor = tensor
        if zero_point_shift != 0.0:
            shifted_tensor = torch.sub(tensor, scale, alpha=zero_point_shift)

        # QDQ
        x_round = RoundManual.apply(shifted_tensor.to(scale.dtype) / scale).sub_(offset)
        x_quant = x_round.clamp_(qmin, qmax)
        x_qdq = x_quant.add_(offset).mul_(scale)

        if zero_point_shift != 0.0:
            x_qdq = torch.add(x_qdq, scale, alpha=zero_point_shift)

        return x_qdq.to(output_dtype).view(orig_tensor_shape)

    @staticmethod
    def compute_min_max_tensors(weight_tensor, shape, num_steps):
        """
        compute encodings of weight tensor (instead of EncodingAnalyzer)
        """
        min_tensor = reduce(weight_tensor, shape=shape, reduce_op=torch.min).values
        max_tensor = reduce(weight_tensor, shape=shape, reduce_op=torch.max).values

        # enforces that 0 is within the min/max
        min_with_zero = torch.clamp(min_tensor, max=0)
        max_with_zero = torch.clamp(max_tensor, min=0)

        minimum_scale = _get_minimum_scale(num_steps)

        # adjusts any min/max pairing that are too close
        tensor_diff = (max_with_zero - min_with_zero) / num_steps
        adjustment_step = minimum_scale * (tensor_diff < minimum_scale)

        updated_max = max_with_zero + math.floor(num_steps / 2) * adjustment_step
        updated_min = min_with_zero - math.ceil(num_steps / 2) * adjustment_step

        num_pos_steps = math.floor(num_steps / 2)
        num_neg_steps = math.ceil(num_steps / 2)

        delta = torch.maximum(updated_max / num_pos_steps, -updated_min / num_neg_steps)
        offset = -1 * num_neg_steps
        updated_min = offset * delta
        updated_max = num_pos_steps * delta

        updated_max = torch.clamp(
            updated_max, max=torch.finfo(min_tensor.dtype).max
        ).to(min_tensor.dtype)
        updated_min = torch.clamp(
            updated_min, min=torch.finfo(max_tensor.dtype).min
        ).to(max_tensor.dtype)

        return updated_min, updated_max

    def get_min(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization min to be used for forward pass.

        NOTE: self.min may not be equal to self.get_min().
              self.get_min() returns slightly recalibrated version of self.min.

        :param dtype: dtype of the computed min. Use of self.min.dtype by default.
        :return: Quantization min
        """
        return self.get_scale(dtype) * (
            self.get_offset(dtype) + self.qmin + self.zero_point_shift
        )

    def get_max(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization max to be used for forward pass.

        NOTE: self.max may not be equal to self.get_max()
              self.get_max() returns slightly recalibrated version of self.max.

        :param dtype: dtype of the computed max. Use of self.min.dtype by default.
        :return: Quantization max
        """
        return self.get_scale(dtype) * (
            self.get_offset(dtype) + self.qmax + self.zero_point_shift
        )

    def get_scale(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization scale to be used for forward pass.
        Return None if the quantizer is not initialized yet.

        Args:
            dtype (torch.dtype): dtype of the computed scale

        Returns:
            Quantization scale
        """

        dtype = dtype or torch.float32

        num_steps = self.qmax - self.qmin
        scale = (self.max.to(dtype) - self.min.to(dtype)) / num_steps

        return torch.clamp_min(
            scale.to(dtype), _get_minimum_scale(self.qmax - self.qmin)
        )

    def get_offset(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization offset to be used for forward pass.
        Return None if the quantizer is not initialized yet.

        Args:
            dtype (torch.dtype): dtype of the computed offset

        Returns:
            Quantization offset
        """

        dtype = dtype or torch.float32
        device = next(p.device for p in self.parameters())
        offset = get_symmetric_offset(self.qmin, self.qmax, self.shape, dtype, device)
        return offset.to(dtype)

    @torch.no_grad()
    def set_range(self, min_tensor: torch.Tensor, max_tensor: torch.Tensor):
        """
        Set quantization parameters to the given min-max range
        """
        self.min.copy_(min_tensor)
        self.max.copy_(max_tensor)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.as_subclass(torch.Tensor)
        output = self.quantize_dequantize(
            input_tensor,
            self.get_scale(),
            self.get_offset(),
            self.qmin,
            self.qmax,
            block_size=self.block_size,
            zero_point_shift=self.zero_point_shift,
        )
        return output

    def _get_num_steps(self) -> int:
        return self.qmax - self.qmin


class AdaScaleLinearWeightQdq(WeightQdq):
    """Only for linear layers"""

    beta: torch.nn.Parameter
    gamma: torch.nn.Parameter
    s2: torch.nn.Parameter
    s3: torch.nn.Parameter

    def __init__(
        self,
        weight_tensor: torch.Tensor,
        enc_shape: tuple,
        bitwidth: int,
        block_size=None,
        zero_point_shift=None,
    ):
        super().__init__(
            weight_tensor, enc_shape, bitwidth, block_size, zero_point_shift
        )
        self.register_parameter("beta", torch.nn.Parameter(torch.zeros(self.shape)))
        self.register_parameter("gamma", torch.nn.Parameter(torch.zeros(self.shape)))

        if block_size is not None:
            self.register_parameter(
                "s2",
                torch.nn.Parameter(
                    reshape_tensor_for_blocks(
                        torch.zeros(weight_tensor.shape), enc_shape, self.block_size
                    ).squeeze(1)
                ),
            )
            self.register_parameter(
                "s3", torch.nn.Parameter(torch.zeros(self.shape).unsqueeze(-1))
            )
        else:
            self.register_parameter(
                "s2", torch.nn.Parameter(torch.zeros(weight_tensor.shape))
            )
            self.register_parameter("s3", torch.nn.Parameter(torch.zeros(enc_shape)))

        self.min.requires_grad = self.max.requires_grad = False
        self.beta.requires_grad = self.gamma.requires_grad = True
        self.s2.requires_grad = self.s3.requires_grad = True

    def get_adascale_trainable_parameters(self):
        """Method to query all the trainable parameters of AdaScale QDQ"""
        return [self.beta, self.gamma], self._get_learnable_scales()

    def get_scale(self, dtype=None) -> Optional[torch.Tensor]:
        dtype = dtype or torch.float32
        scale = (
            torch.exp(self.gamma) * self.max.to(dtype)
            - torch.exp(self.beta) * self.min.to(dtype)
        ) / self._get_num_steps()
        return scale

    def get_offset(self, dtype=None) -> Optional[torch.Tensor]:
        dtype = dtype or torch.float32
        return torch.zeros_like(self.min, requires_grad=False, dtype=dtype)

    def get_folded_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Return the folded weight of the layer. This method along with get_qdq can be used to convert AdaScale
        QDQ object into regular QDQ object
        """
        for scale in self._get_learnable_scales():
            weight = weight / torch.exp(scale)
        return weight

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs QDQ on the input tensor based on the learnt scales by using the parameters min and max

        :param input_tensor: Input tensor to be QDQ
        :return: Dequantized tensor after applying AdaScale QDQ
        """
        # AdaScale trainable params (s2/s3/beta/gamma) stay in fp32 for Adam
        # numerical stability; the block outputs remain in the weight's dtype
        # (fp16/bf16 in low-precision graphs). Cast back so downstream
        # Linear/Conv sees a matching-dtype weight.
        orig_dtype = input_tensor.dtype
        for scale in self._get_learnable_scales():
            input_tensor = input_tensor / torch.exp(scale)
        return super().forward(input_tensor.to(orig_dtype))

    def _get_learnable_scales(self) -> list[torch.Tensor]:
        return [self.s2, self.s3]


class AdaScaleConvWeightQdq(WeightQdq):
    """Only for linear layers"""

    beta: torch.nn.Parameter
    gamma: torch.nn.Parameter
    s2: torch.nn.Parameter
    s3: torch.nn.Parameter
    s4: torch.nn.Parameter

    def __init__(
        self,
        weight_tensor: torch.Tensor,
        enc_shape: tuple,
        bitwidth: int,
        block_size=None,
        zero_point_shift=None,
    ):
        super().__init__(
            weight_tensor, enc_shape, bitwidth, block_size, zero_point_shift
        )
        self.register_parameter("beta", torch.nn.Parameter(torch.zeros(self.shape)))
        self.register_parameter("gamma", torch.nn.Parameter(torch.zeros(self.shape)))

        if block_size is not None:
            self.register_parameter(
                "s2",
                torch.nn.Parameter(
                    reshape_tensor_for_blocks(
                        torch.zeros(weight_tensor.shape), enc_shape, self.block_size
                    ).squeeze(1)
                ),
            )
        else:
            self.register_parameter(
                "s2", torch.nn.Parameter(torch.zeros(weight_tensor.shape))
            )

        out_ch, in_ch, _, _ = weight_tensor.shape
        self.register_parameter(
            "s3", torch.nn.Parameter(torch.zeros((out_ch, 1, 1, 1)))
        )
        self.register_parameter("s4", torch.nn.Parameter(torch.zeros((1, in_ch, 1, 1))))

        self.min.requires_grad = self.max.requires_grad = False
        self.beta.requires_grad = self.gamma.requires_grad = True
        self.s2.requires_grad = self.s3.requires_grad = self.s4.requires_grad = True

    def get_adascale_trainable_parameters(self):
        """Method to query all the trainable parameters of AdaScale QDQ"""
        return [self.beta, self.gamma], self._get_learnable_scales()

    def get_scale(self, dtype=None) -> Optional[torch.Tensor]:
        dtype = dtype or torch.float32
        scale = (
            torch.exp(self.gamma) * self.max.to(dtype)
            - torch.exp(self.beta) * self.min.to(dtype)
        ) / self._get_num_steps()
        return scale

    def get_offset(self, dtype=None) -> Optional[torch.Tensor]:
        dtype = dtype or torch.float32
        return torch.zeros_like(self.min, requires_grad=False, dtype=dtype)

    def get_folded_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Return the folded weight of the layer. This method along with get_qdq can be used to convert AdaScale
        QDQ object into regular QDQ object
        """
        for scale in self._get_learnable_scales():
            weight = weight / torch.exp(scale)
        return weight

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs QDQ on the input tensor based on the learnt scales by using the parameters min and max

        :param input_tensor: Input tensor to be QDQ
        :return: Dequantized tensor after applying AdaScale QDQ
        """
        orig_dtype = input_tensor.dtype
        for scale in self._get_learnable_scales():
            input_tensor = input_tensor / torch.exp(scale)
        return super().forward(input_tensor.to(orig_dtype))

    def _get_learnable_scales(self) -> list[torch.Tensor]:
        return [self.s2, self.s3, self.s4]


class QuantizedLinear(torch.nn.Linear):
    """
    Lightweight quantized wrapper over an existing torch.nn.Linear.

    Args:
        original_module (torch.nn.Linear): Pre-existing linear layer to wrap.
        bitwidth (int): Weight quantization bitwidth.
        block_size: Optional block size for block-wise encodings.
        zero_point_shift: Optional zero point shift factor.
        enc_shape (tuple): Encoding tensor shape; defaults to (out_features, 1).
    """

    def __init__(
        self,
        original_module: torch.nn.Linear,
        *,
        bitwidth: int = 4,
        block_size=None,
        zero_point_shift=None,
        enc_shape=None,
    ):
        super().__init__(
            original_module.in_features,
            original_module.out_features,
            bias=original_module.bias is not None,
            device=original_module.weight.device,
            dtype=original_module.weight.dtype,
        )

        # Reuse (share) existing parameters
        self.weight = original_module.weight
        if original_module.bias is not None:
            self.bias = original_module.bias

        enc_shape = enc_shape or (self.weight.shape[0], 1)
        self.param_quantizers = torch.nn.ModuleDict()
        self.param_quantizers["weight"] = WeightQdq(
            self.weight,
            bitwidth=bitwidth,
            block_size=block_size,
            zero_point_shift=zero_point_shift,
            enc_shape=enc_shape,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        qdq = self.param_quantizers["weight"]
        w = qdq(self.weight) if qdq is not None else self.weight
        return torch.nn.functional.linear(input, w, self.bias)


class QuantizedConv2d(torch.nn.Conv2d):
    """
    Lightweight quantized wrapper over an existing torch.nn.Conv2d.

    Args:
        original_module (torch.nn.Conv2d): Pre-existing linear layer to wrap.
        bitwidth (int): Weight quantization bitwidth.
        block_size: Optional block size for block-wise encodings.
        zero_point_shift: Optional zero point shift factor.
        enc_shape (tuple): Encoding tensor shape; defaults to (out_features, 1).
    """

    def __init__(
        self,
        original_module: torch.nn.Conv2d,
        *,
        bitwidth: int = 4,
        block_size=None,
        zero_point_shift=None,
        enc_shape=None,
    ):
        super().__init__(
            in_channels=original_module.in_channels,
            out_channels=original_module.out_channels,
            kernel_size=original_module.kernel_size,
            padding=original_module.padding,
            dilation=original_module.dilation,
            groups=original_module.groups,
            stride=original_module.stride,
            bias=original_module.bias is not None,
            device=original_module.weight.device,
            dtype=original_module.weight.dtype,
        )

        # Reuse (share) existing parameters
        self.weight = original_module.weight
        if original_module.bias is not None:
            self.bias = original_module.bias

        enc_shape = enc_shape or (self.weight.shape[0], 1, 1, 1)
        self.param_quantizers = torch.nn.ModuleDict()
        self.param_quantizers["weight"] = WeightQdq(
            self.weight,
            bitwidth=bitwidth,
            block_size=block_size,
            zero_point_shift=zero_point_shift,
            enc_shape=enc_shape,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        qdq = self.param_quantizers["weight"]
        w = qdq(self.weight) if qdq is not None else self.weight
        return torch.nn.functional.conv2d(
            input, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def add_qlinear_layers(
    model: torch.nn.Module,
    bitwidth: int = 4,
    block_size=None,
    zero_point_shift=None,
    per_module_bitwidth: Optional[Dict[str, int]] = None,
) -> torch.nn.Module:
    """
    :param per_module_bitwidth: Optional mapping of dotted module name (as seen via
        ``model.named_modules()``) to a bitwidth override. If passed, every
        Linear/Conv2d module MUST have an entry here -- a missing entry raises
        ``KeyError`` rather than silently falling back to ``bitwidth``. If not
        passed (``None``), every Linear/Conv2d uses the flat ``bitwidth`` arg.
    """

    def _module_bitwidth(name: str) -> int:
        if per_module_bitwidth is None:
            return bitwidth
        return per_module_bitwidth[name]

    def _convert_to_qmodule(module: torch.nn.Module, name: str = ""):
        if isinstance(module, torch.nn.Linear):
            enc_shape = (module.weight.shape[0], 1)
            qmodule = QuantizedLinear(
                module,
                enc_shape=enc_shape,
                bitwidth=_module_bitwidth(name),
                block_size=block_size,
                zero_point_shift=zero_point_shift,
            )
            return qmodule

        elif isinstance(module, torch.nn.Conv2d):
            enc_shape = (module.weight.shape[0], 1, 1, 1)
            qmodule = QuantizedConv2d(
                module,
                enc_shape=enc_shape,
                bitwidth=_module_bitwidth(name),
                block_size=block_size,
                zero_point_shift=zero_point_shift,
            )
            return qmodule

        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            setattr(module, child_name, _convert_to_qmodule(child, full_name))
        return module

    model = _convert_to_qmodule(model)
    return model


def replace_with_adascale_quantizers(model: torch.nn.Module) -> torch.nn.Module:
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            m.param_quantizers["weight"] = AdaScaleLinearWeightQdq(
                weight_tensor=m.weight,
                enc_shape=m.param_quantizers["weight"].shape,
                bitwidth=m.param_quantizers["weight"].bitwidth,
                block_size=m.param_quantizers["weight"].block_size,
                zero_point_shift=m.param_quantizers["weight"].zero_point_shift,
            )

        elif isinstance(m, QuantizedConv2d):
            m.param_quantizers["weight"] = AdaScaleConvWeightQdq(
                weight_tensor=m.weight,
                enc_shape=m.param_quantizers["weight"].shape,
                bitwidth=m.param_quantizers["weight"].bitwidth,
                block_size=m.param_quantizers["weight"].block_size,
                zero_point_shift=m.param_quantizers["weight"].zero_point_shift,
            )


def get_adascale_trainable_params(
    non_leaf_module: torch.nn.Module,
) -> Tuple[List, List]:
    """Get all the adascale scale params present in the non-leaf module"""
    all_scale_parameters = []
    all_beta_gamma_parameters = []
    for module in non_leaf_module.modules():
        if (
            isinstance(module, QuantizedLinear)
            and isinstance(module.param_quantizers["weight"], AdaScaleLinearWeightQdq)
        ) or (
            isinstance(module, QuantizedConv2d)
            and isinstance(module.param_quantizers["weight"], AdaScaleConvWeightQdq)
        ):
            beta_gamma_params, scale_parameters = module.param_quantizers[
                "weight"
            ].get_adascale_trainable_parameters()
            all_beta_gamma_parameters.extend(beta_gamma_params)
            all_scale_parameters.extend(scale_parameters)
    return all_beta_gamma_parameters, all_scale_parameters
