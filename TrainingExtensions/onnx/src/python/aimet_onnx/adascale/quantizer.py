# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
from typing import Optional, List, Tuple, Dict
import functools
import math

import torch
import torch.utils.checkpoint
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.common.quantsim import _get_minimum_scale
from aimet_onnx.qc_quantize_op import QcQuantizeOp
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


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)


def _is_compiling() -> bool:
    # torch.compiler.is_compiling is new in torch 2.7; nothing compiles before that here
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    return bool(is_compiling and is_compiling())


def _assume_constant_result(fn):
    # Read once at trace time and baked into the compiled graph, rather than traced: these
    # checks read tensor values, which would otherwise split the graph at every branch on them
    assume = getattr(getattr(torch, "compiler", None), "assume_constant_result", None)
    return assume(fn) if assume is not None else fn


_cached_numerically_stable = functools.lru_cache(None)(is_numerically_stable)


@_assume_constant_result
def _numerically_stable(dtype: torch.dtype, qmin: int, qmax: int) -> bool:
    return bool(_cached_numerically_stable(dtype, qmin, qmax))


@_assume_constant_result
def _grid_representable(dtype: torch.dtype, qmin: int, qmax: int) -> bool:
    # is_grid_representable is cached already
    return bool(is_grid_representable(dtype, qmin, qmax))


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

        # Keep min/max in fp32 even for fp16/bf16 weights; bf16 would round the
        # encodings loaded from the sim by up to ~0.2%.
        min_tensor, max_tensor = self.compute_min_max_tensors(
            weight_tensor.detach().float(), self.shape, self._get_num_steps()
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
        # Skipped while compiling, which can't trace the errors it raises: a shape or dtype
        # mismatch then fails in the compiled ops instead, with a less specific message
        if not _is_compiling():
            validate_arguments(tensor, scale, qmin, qmax, block_size)

        output_dtype = internal_dtype = tensor.dtype

        if not _numerically_stable(internal_dtype, qmin, qmax):
            internal_dtype = torch.float32
            if not _numerically_stable(internal_dtype, qmin, qmax):
                internal_dtype = torch.float64

        if not _grid_representable(internal_dtype, qmin, qmax):
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
        return self._compute_scale(
            self.gamma, self.beta, self.min, self.max, self._get_num_steps(), dtype
        )

    @staticmethod
    def _compute_scale(gamma, beta, min_tensor, max_tensor, num_steps, dtype=None):
        """Scale from the learnt ``gamma``/``beta`` and the initial range. Elementwise, so it
        also takes these parameters of several quantizers stacked along a new leading dim."""
        dtype = dtype or torch.float32
        return (
            torch.exp(gamma) * max_tensor.to(dtype)
            - torch.exp(beta) * min_tensor.to(dtype)
        ) / num_steps

    def get_offset(self, dtype=None) -> Optional[torch.Tensor]:
        dtype = dtype or torch.float32
        return torch.zeros_like(self.min, requires_grad=False, dtype=dtype)

    def get_folded_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Return the folded weight of the layer. This method along with get_qdq can be used to convert AdaScale
        QDQ object into regular QDQ object
        """
        return self._fold(weight, self._get_learnable_scales())

    @staticmethod
    def _fold(weight: torch.Tensor, scales: List[torch.Tensor]) -> torch.Tensor:
        for scale in scales:
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
        input_tensor = self._fold(input_tensor, self._get_learnable_scales())
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
        self.param_quantizers["weight"] = AdaScaleLinearWeightQdq(
            self.weight,
            bitwidth=bitwidth,
            block_size=block_size,
            zero_point_shift=zero_point_shift,
            enc_shape=enc_shape,
        )
        # QDQ'd weight precomputed by _BatchedWeightQdq for the next call, used once
        self._batched_weight: Optional[torch.Tensor] = None
        # Stacked tensor of its group's weights that this layer's weight views, once batched
        self._stacked_weights: Optional[torch.Tensor] = None
        # The _BatchedWeightQdq batching this layer, if any
        self._batched_by = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        if self._batched_weight is not None:
            w, self._batched_weight = self._batched_weight, None
        else:
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
        self.param_quantizers["weight"] = AdaScaleConvWeightQdq(
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
    per_module_quantizers: Dict[str, QcQuantizeOp],
) -> torch.nn.Module:
    """
    :param per_module_quantizers: Mapping of dotted module name (as seen via
        ``model.named_modules()``) to its ONNX quantizer. Bitwidth and initialized
        encoding ranges are copied from the source quantizer. Every Linear/Conv2d
        module must have an entry.
    """

    def _initialize_quantizer(qmodule: torch.nn.Module, source_quantizer: QcQuantizeOp):
        qdq = qmodule.param_quantizers["weight"]
        # Use WeightQDQ quantizer for frozen quantizer for efficiency
        if source_quantizer.is_encoding_frozen():
            weight_qdq = WeightQdq(
                weight_tensor=qmodule.weight,
                enc_shape=qdq.shape,
                bitwidth=qdq.bitwidth,
            )
            qmodule.param_quantizers["weight"] = weight_qdq
            qdq = weight_qdq
            qdq.requires_grad_(False)

        encodings = source_quantizer.get_encodings()
        if encodings is not None:
            min_values = [encoding.min for encoding in encodings]
            max_values = [encoding.max for encoding in encodings]
            min_tensor = torch.as_tensor(
                min_values, dtype=qdq.min.dtype, device=qdq.min.device
            ).reshape(qdq.shape)
            max_tensor = torch.as_tensor(
                max_values, dtype=qdq.max.dtype, device=qdq.max.device
            ).reshape(qdq.shape)
            qdq.set_range(min_tensor, max_tensor)

    def _convert_to_qmodule(module: torch.nn.Module, name: str = ""):
        if isinstance(module, torch.nn.Linear):
            source_quantizer = per_module_quantizers[name]
            enc_shape = (module.weight.shape[0], 1)
            qmodule = QuantizedLinear(
                module,
                enc_shape=enc_shape,
                bitwidth=source_quantizer.bitwidth,
            )
            _initialize_quantizer(qmodule, source_quantizer)
            return qmodule

        elif isinstance(module, torch.nn.Conv2d):
            source_quantizer = per_module_quantizers[name]
            enc_shape = (module.weight.shape[0], 1, 1, 1)
            qmodule = QuantizedConv2d(
                module,
                enc_shape=enc_shape,
                bitwidth=source_quantizer.bitwidth,
            )
            _initialize_quantizer(qmodule, source_quantizer)
            return qmodule

        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            setattr(module, child_name, _convert_to_qmodule(child, full_name))
        return module

    model = _convert_to_qmodule(model)
    return model


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


class _GatedStack(torch.autograd.Function):  # pylint: disable=abstract-method
    """
    ``torch.stack`` of one parameter of every layer in a group, whose backward gives a gradient
    only to the layers in ``used``: the layers whose QDQ'd weight the forward that stacked them
    used. ``used`` is filled in after that forward has run, before its backward.

    A layer the forward didn't use (an expert in an If branch it didn't take) would otherwise
    get a zero gradient through the stack, which Adam steps on. Without batching it gets no
    gradient at all, and that is what this gives it, however many forwards share a backward and
    however many backwards accumulate before a step.
    """

    @staticmethod
    def forward(ctx, used, *tensors):  # pylint: disable=arguments-differ
        ctx.used = used
        return torch.stack(tensors)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        return (
            None,
            *(
                grad if i in ctx.used else None
                for i, grad in enumerate(grad_output.unbind(0))
            ),
        )


class _BatchedWeightQdq:
    """
    Computes the AdaScale weight QDQ of a block's linear layers in one stacked call per group
    of layers with the same weight shape, dtype and quantization, rather than one call per
    layer. A mixture-of-experts block has hundreds of such layers (one per expert projection),
    and its training step is otherwise dominated by launching their small QDQ kernels.

    Before each forward of the block, every grouped layer is handed its slice of the group's
    QDQ'd weights, which it uses for its next call instead of running its own quantizer. The
    ops are the same, elementwise, as :meth:`AdaScaleLinearWeightQdq.forward`, so outputs and
    gradients are unchanged. Layers with a frozen (non-AdaScale) quantizer, block-wise
    encodings or no other layer to group with keep running their own quantizer.

    Memory, compared with each layer running its own QDQ:

    - A group's weights are stored once, as one stacked tensor: each layer's weight becomes a
      view of its slice, so stacking them for the QDQ copies nothing. The layers keep their own
      AdaScale parameters, so the optimizer and copying the results back are unaffected.
    - The stacked QDQ is recomputed in backward rather than saving its intermediates, which
      are several times the size of the weights.
    """

    def __init__(self, block: torch.nn.Module, compiled: bool = False):
        """
        :param block: Block whose linear layers to batch
        :param compiled: Run the stacked QDQ through ``torch.compile``. Faster, but not
            bit-identical to the per-layer QDQ: see :func:`_compiled_stacked_qdq`.
        """
        self._compiled = compiled
        groups: Dict[tuple, List[QuantizedLinear]] = {}
        for module in block.modules():
            if not isinstance(module, QuantizedLinear):
                continue
            qdq = module.param_quantizers["weight"]
            if (
                not isinstance(qdq, AdaScaleLinearWeightQdq)
                or qdq.block_size is not None
            ):
                continue
            key = (
                tuple(module.weight.shape),
                module.weight.dtype,
                module.weight.device,
                qdq.bitwidth,
                qdq.shape,
                qdq.zero_point_shift,
            )
            groups.setdefault(key, []).append(module)
        self.groups = [layers for layers in groups.values() if len(layers) > 1]
        for layers in self.groups:
            for layer in layers:
                if layer._batched_by is not None:  # pylint: disable=protected-access
                    raise RuntimeError(
                        "The block is already batched: remove() the earlier "
                        "_BatchedWeightQdq first. Two at once would each think the other's "
                        "layers didn't run, and give them the wrong gradients."
                    )
                layer._batched_by = self  # pylint: disable=protected-access
        self._weights = [self._store_once(layers) for layers in self.groups]
        # Per group, the indices of the layers whose QDQ'd weight the current forward used
        self._used: List[set] = [set() for _ in self.groups]
        self._handles = []
        if self.groups:
            self._handles = [
                block.register_forward_pre_hook(self._precompute),
                block.register_forward_hook(self._release),
            ]

    @staticmethod
    def _store_once(layers: List["QuantizedLinear"]) -> torch.Tensor:
        """Move the group's weights into one stacked tensor; each layer's weight becomes a
        view of its slice, so it is copied back and loaded as before. Weights already stored
        that way (the block was batched before) are kept where they are."""
        first = layers[0].weight
        stored = getattr(layers[0], "_stacked_weights", None)
        if (
            stored is not None
            and stored.shape == (len(layers), *first.shape)
            and all(
                getattr(layer, "_stacked_weights", None) is stored
                and layer.weight.data_ptr() == weight.data_ptr()
                for layer, weight in zip(layers, stored.unbind(0))
            )
        ):
            return stored
        weights = torch.empty(
            (len(layers), *first.shape), dtype=first.dtype, device=first.device
        )
        for layer, weight in zip(layers, weights.unbind(0)):
            weight.copy_(layer.weight.detach())
            layer.weight = torch.nn.Parameter(
                weight, requires_grad=layer.weight.requires_grad
            )
            layer._stacked_weights = weights  # pylint: disable=protected-access
        return weights

    @property
    def num_batched(self) -> int:
        """Number of layers whose QDQ is batched."""
        return sum(len(layers) for layers in self.groups)

    def remove(self):
        """Stop batching: remove the hooks and drop any precomputed weights. The layers keep
        their weights, as views of the stacked tensor."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._release()
        for layers in self.groups:
            for layer in layers:
                if layer._batched_by is self:  # pylint: disable=protected-access
                    layer._batched_by = None  # pylint: disable=protected-access

    def _precompute(self, *_):
        for layers, weights in zip(self.groups, self._weights):
            for layer, weight in zip(layers, weights.unbind(0)):
                if layer.weight.data_ptr() != weight.data_ptr():
                    raise RuntimeError(
                        "A batched layer's weight no longer views its group's stacked weights "
                        "(was the block moved to another device?), so the batched QDQ would "
                        "use stale weights. Batch the block after moving it."
                    )
        # Fresh sets: a backward still to come for an earlier forward keeps that forward's
        self._used = [set() for _ in self.groups]
        for layers, weights, used in zip(self.groups, self._weights, self._used):
            if self._compiled:
                # The compiled kernel recomputes its intermediates in backward itself
                qdq = self._qdq(layers, weights, used, compiled=True)
            else:
                qdq = torch.utils.checkpoint.checkpoint(
                    self._qdq, layers, weights, used, use_reentrant=False
                )
            # unbind rather than indexing: the backward of each index would write a
            # zero-filled gradient the size of the whole stack
            for layer, weight in zip(layers, qdq.unbind(0)):
                layer._batched_weight = weight  # pylint: disable=protected-access

    def _release(self, *_):
        # A layer whose weight is still there didn't run, e.g. in an If branch the forward
        # didn't take
        for layers, used in zip(self.groups, self._used):
            for i, layer in enumerate(layers):
                if layer._batched_weight is None:  # pylint: disable=protected-access
                    used.add(i)
                layer._batched_weight = None  # pylint: disable=protected-access

    @staticmethod
    def _qdq(
        layers: List["QuantizedLinear"],
        weights: torch.Tensor,
        used: set,
        compiled: bool = False,
    ) -> torch.Tensor:
        qdqs = [layer.param_quantizers["weight"] for layer in layers]
        first = qdqs[0]

        def stack(attr):
            # Stacked (and gated) outside the compiled function: the gate reads ``used`` when
            # backward runs, which a compiled graph would read once, at trace time
            return _GatedStack.apply(used, *(getattr(qdq, attr) for qdq in qdqs))

        def stack_constant(attr):
            return torch.stack([getattr(qdq, attr) for qdq in qdqs])

        if compiled:
            return _compiled_stacked_qdq()(
                weights,
                stack("s2"),
                stack("s3"),
                stack("gamma"),
                stack("beta"),
                stack_constant("min"),
                stack_constant("max"),
                first.qmin,
                first.qmax,
                float(first.zero_point_shift),
            )
        # The two steps of _stacked_qdq, called one after the other here so that the stacked
        # s2 and s3 (each the size of the weights, in float32) are freed once folded, rather
        # than held through the quantize-dequantize
        weight, scale = _fold_and_scale(
            weights,
            stack("s2"),
            stack("s3"),
            stack("gamma"),
            stack("beta"),
            stack_constant("min"),
            stack_constant("max"),
            first._get_num_steps(),  # pylint: disable=protected-access
        )
        return _quantize_dequantize(
            weight, scale, first.qmin, first.qmax, float(first.zero_point_shift)
        )


def _fold_and_scale(
    weights: torch.Tensor,
    s2: torch.Tensor,
    s3: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    min_tensor: torch.Tensor,
    max_tensor: torch.Tensor,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stacked weights with the learnt scales folded in, and the scale from the learnt
    gamma/beta, as :meth:`AdaScaleLinearWeightQdq.forward` computes them for one layer."""
    weight = AdaScaleLinearWeightQdq._fold(  # pylint: disable=protected-access
        weights, [s2, s3]
    ).to(weights.dtype)
    scale = AdaScaleLinearWeightQdq._compute_scale(  # pylint: disable=protected-access
        gamma, beta, min_tensor, max_tensor, num_steps
    )
    return weight, scale


def _quantize_dequantize(
    weight: torch.Tensor,
    scale: torch.Tensor,
    qmin: int,
    qmax: int,
    zero_point_shift: float,
) -> torch.Tensor:
    """:meth:`WeightQdq.quantize_dequantize` of folded stacked weights, symmetrically."""
    offset = torch.zeros_like(scale, requires_grad=False)
    if not _is_compiling():
        # Strips a tensor subclass of the weight; a compiled graph only sees plain tensors
        weight = weight.as_subclass(torch.Tensor)
    return WeightQdq.quantize_dequantize(
        weight, scale, offset, qmin, qmax, zero_point_shift=zero_point_shift
    )


def _stacked_qdq(
    weights: torch.Tensor,
    s2: torch.Tensor,
    s3: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    min_tensor: torch.Tensor,
    max_tensor: torch.Tensor,
    qmin: int,
    qmax: int,
    zero_point_shift: float,
) -> torch.Tensor:
    """
    AdaScale weight QDQ of a group's stacked weights and quantizer parameters, as one function
    to compile: :func:`_fold_and_scale`, then :func:`_quantize_dequantize`. The eager batched
    QDQ runs the same two functions.
    """
    weight, scale = _fold_and_scale(
        weights, s2, s3, gamma, beta, min_tensor, max_tensor, qmax - qmin
    )
    return _quantize_dequantize(weight, scale, qmin, qmax, zero_point_shift)


# Distinct weight-group shapes one compiled QDQ may specialize for before dynamo stops
# recompiling. A mixture-of-experts model has a few (3 on Qwen3-30B-A3B); dynamo's default of 8
# is per compiled function, across every model the process runs AdaScale on.
_COMPILE_RECOMPILE_LIMIT = 64

_COMPILED_STACKED_QDQ = None


def _reset_compiled_stacked_qdq():
    """Forget the compiled weight QDQ, so the next AdaScale run compiles afresh and the shapes
    of models optimized earlier in the process don't count towards the recompile limit."""
    global _COMPILED_STACKED_QDQ  # pylint: disable=global-statement
    _COMPILED_STACKED_QDQ = None


def _compiled_stacked_qdq():
    """
    :func:`_stacked_qdq` compiled with inductor, falling back to the eager function for good
    if compiling fails (e.g. no Triton).

    The fused kernel reads and writes the stacked weights once rather than once per op, so the
    QDQ of a mixture-of-experts block's expert weights is about 4x faster. It is not
    bit-identical to the eager ops: ``emulate_precision_casts`` keeps the intermediate bf16
    roundings of the eager path, but fused multiply-adds still leave a few weights (65 of 606M
    on a Qwen3-30B-A3B block) one quantization step apart.
    """
    global _COMPILED_STACKED_QDQ  # pylint: disable=global-statement
    if _COMPILED_STACKED_QDQ is None:
        import torch._dynamo.config as dynamo_config  # pylint: disable=import-outside-toplevel
        import torch._inductor.config as inductor_config  # pylint: disable=import-outside-toplevel

        compiled = torch.compile(_stacked_qdq, dynamic=False, fullgraph=True)
        state = {"failed": False}

        def run(*args):
            if state["failed"]:
                return _stacked_qdq(*args)
            try:
                with (
                    inductor_config.patch(emulate_precision_casts=True),
                    dynamo_config.patch(recompile_limit=_COMPILE_RECOMPILE_LIMIT),
                ):
                    return compiled(*args)
            except Exception as exc:  # pylint: disable=broad-except
                # e.g. no Triton, or more distinct weight shapes than dynamo recompiles for
                _logger.warning(
                    "Compiling the AdaScale weight QDQ failed (%s: %s); using eager QDQ "
                    "from here on.",
                    type(exc).__name__,
                    str(exc).splitlines()[0] if str(exc) else "",
                )
                state["failed"] = True
                return _stacked_qdq(*args)

        _COMPILED_STACKED_QDQ = run
    return _COMPILED_STACKED_QDQ
