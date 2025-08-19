# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

import functools
import contextlib

import torch
from torch import nn

from .transform_ops import TransformOp

from aimet_torch.v2.utils import patch_attr
from aimet_torch.v2.nn import compute_param_encodings
from aimet_torch.v2.nn.true_quant import QuantizationMixin


# pylint: disable=abstract-method
class TransformationMixin(torch.nn.Module):
    cls_to_tcls: dict = {}
    tcls_to_cls: dict = {}

    def __init__(self, module: torch.nn.Module):  # pylint: disable=protected-access
        # pylint: disable=super-init-not-called
        super().__new__(self.__class__)  # pylint: disable=no-value-for-parameter
        self.__dict__.update(module.__dict__)
        self._modules = module._modules.copy()
        self._parameters = module._parameters.copy()
        self._buffers = module._buffers.copy()

        self.right_hand_transforms: nn.ModuleList = nn.ModuleList()
        self.left_hand_transforms: nn.ModuleList = nn.ModuleList()

    def add_right_hand_transform(self, transform: TransformOp):
        # we can't interleave mergeable and non-mergeable transforms
        # all non-mergeable RHS transforms should be applied AFTER all mergeable RHS transforms
        if (
            transform.mergeable
            and len(self.right_hand_transforms) > 0
            and not self.right_hand_transforms[-1].mergeable
        ):
            # only an issue if a mergeable transform follows a non-mergeable transform
            raise RuntimeError(
                "Cannot add mergeable transform to RHS after non-mergeable transform."
            )

        if isinstance(self, QuantizationMixin) and not transform.mergeable:
            transform = QuantizationMixin.from_module(transform)

        self.right_hand_transforms.append(transform)

    def add_left_hand_transform(self, transform: TransformOp):
        # we can't interleave mergeable and non-mergeable transforms
        # all non-mergeable LHS transforms should be applied BEFORE all mergeable LHS transforms
        if (
            transform.mergeable
            and len(self.left_hand_transforms) > 0
            and not self.left_hand_transforms[0].mergeable
        ):
            # only an issue if a mergeable transform is before a non-mergeable transform
            raise RuntimeError(
                "Cannot add mergeable transform to LHS before non-mergeable transform."
            )

        if isinstance(self, QuantizationMixin) and not transform.mergeable:
            transform = QuantizationMixin.from_module(transform)

        self.left_hand_transforms.insert(0, transform)

    def _compute_merged_params(self):
        mergeable_left_hand_transforms = [
            transform for transform in self.left_hand_transforms if transform.mergeable
        ]
        mergeable_right_hand_transforms = [
            transform for transform in self.right_hand_transforms if transform.mergeable
        ]

        transformed_weight = functools.reduce(
            lambda weight, transform: transform.right_hand_merge(weight),
            mergeable_right_hand_transforms,
            functools.reduce(
                lambda weight, transform: transform.left_hand_merge(weight),
                mergeable_left_hand_transforms[::-1],
                self.weight.data,
            ),
        )

        if getattr(self, "bias", None) is not None:
            transformed_bias = functools.reduce(
                lambda bias, transform: transform.right_hand_merge(bias),
                mergeable_right_hand_transforms,
                functools.reduce(
                    lambda bias, transform: transform.left_hand_merge(bias),
                    mergeable_left_hand_transforms[::-1],
                    self.bias.data,
                ),
            )
        else:
            transformed_bias = None

        return transformed_weight, transformed_bias

    def merge(self):
        transformed_weight, transformed_bias = self._compute_merged_params()

        setattr(self, "weight", nn.Parameter(transformed_weight))
        if transformed_bias is not None:
            setattr(self, "bias", nn.Parameter(transformed_bias))

        self.right_hand_transforms = torch.nn.ModuleList(
            [
                transform
                for transform in self.right_hand_transforms
                if not transform.mergeable
            ]
        )
        self.left_hand_transforms = torch.nn.ModuleList(
            [
                transform
                for transform in self.left_hand_transforms
                if not transform.mergeable
            ]
        )

    def _patch_transformed_parameters(self):
        transformed_weight, transformed_bias = self._compute_merged_params()
        stack = contextlib.ExitStack()
        stack.enter_context(
            patch_attr(self, "weight", nn.Parameter(transformed_weight))
        )
        stack.enter_context(patch_attr(self, "left_hand_transforms", nn.ModuleList()))
        stack.enter_context(patch_attr(self, "right_hand_transforms", nn.ModuleList()))
        if transformed_bias is not None:
            stack.enter_context(
                patch_attr(self, "bias", nn.Parameter(transformed_bias))
            )
        return stack

    def get_original_module(self):
        if len(self.right_hand_transforms) > 0 or len(self.left_hand_transforms) > 0:
            raise RuntimeError(
                "Cannot obtain original module with unmerged transforms."
            )

        original_cls = type(self)
        if isinstance(self, QuantizationMixin):
            original_cls = QuantizationMixin.qcls_to_cls[original_cls]
        original_cls = self.tcls_to_cls[original_cls]
        if isinstance(self, QuantizationMixin):
            original_cls = QuantizationMixin.cls_to_qcls[original_cls]

        orig_module = super().__new__(original_cls)  # pylint: disable=no-value-for-parameter
        orig_module.__dict__ = self.__dict__.copy()
        orig_module.__dict__.pop("forward", None)

        orig_module._parameters = self._parameters.copy()  # pylint: disable=protected-access
        orig_module._buffers = self._buffers.copy()  # pylint: disable=protected-access
        orig_module._modules = self._modules.copy()  # pylint: disable=protected-access

        return orig_module

    @classmethod
    def implements(cls, module_cls):
        def wrapper(quantized_cls):
            cls.cls_to_tcls[module_cls] = quantized_cls
            cls.tcls_to_cls[quantized_cls] = module_cls
            return quantized_cls

        return wrapper

    @classmethod
    def from_module(cls, module: nn.Module):
        if isinstance(module, TransformationMixin):
            # the layer has already been converted to its Transformed equivalent
            return module

        module_cls = type(module)
        if isinstance(module, QuantizationMixin):
            module_cls = QuantizationMixin.qcls_to_cls[module_cls]
        transformed_cls = cls.cls_to_tcls[module_cls]
        if isinstance(module, QuantizationMixin):
            transformed_cls = QuantizationMixin.cls_to_qcls[transformed_cls]

        return transformed_cls(module)


class _NullaryOrUnaryTransformedLayer(TransformationMixin, nn.Module):
    # pylint: disable=redefined-builtin
    # pylint: disable=arguments-differ
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        transformed_input = (
            functools.reduce(
                lambda x, transform: transform(x), self.left_hand_transforms, input
            )
            if len(self.left_hand_transforms) > 0
            else input
        )
        output = super().forward(transformed_input)
        transformed_output = (
            functools.reduce(
                lambda x, transform: transform(x), self.right_hand_transforms, output
            )
            if len(self.right_hand_transforms) > 0
            else output
        )
        return transformed_output


class _NullaryOrUnaryQuantizedTransformedLayer(
    QuantizationMixin, _NullaryOrUnaryTransformedLayer
):
    def __quant_init__(self):
        pass

    # pylint: disable=redefined-builtin
    # pylint: disable=arguments-differ
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 1) apply non mergeable left hand transforms (they will have their own quantizers present)
        non_mergeable_left_hand_transforms = [
            transform
            for transform in self.left_hand_transforms
            if not transform.mergeable
        ]
        transformed_input = (
            functools.reduce(
                lambda x, transform: transform.forward(x),
                non_mergeable_left_hand_transforms,
                input,
            )
            if len(non_mergeable_left_hand_transforms) > 0
            else input
        )

        # 2) quantize inputs
        if len(self.input_quantizers) > 0 and self.input_quantizers[0]:
            transformed_input = self.input_quantizers[0](transformed_input)

        # 3) Forward
        with contextlib.ExitStack() as stack:
            stack.enter_context(self._patch_transformed_parameters())
            stack.enter_context(self._patch_quantized_parameters())
            stack.enter_context(
                patch_attr(self, "input_quantizers", torch.nn.ModuleList())
            )
            stack.enter_context(
                patch_attr(self, "output_quantizers", torch.nn.ModuleList())
            )
            stack.enter_context(
                patch_attr(self, "param_quantizers", torch.nn.ModuleDict())
            )

            transformed_output = super().forward(transformed_input)

        # 4) Determine non-mergeable right hand transforms
        non_mergeable_right_hand_transforms = [
            transform
            for transform in self.right_hand_transforms
            if not transform.mergeable
        ]

        # 5) quantize outputs
        if self.output_quantizers[0]:
            transformed_output = self.output_quantizers[0](transformed_output)

        # 6) Apply non-mergeable right hand transforms
        transformed_output = (
            functools.reduce(
                lambda x, transform: transform.forward(x),
                non_mergeable_right_hand_transforms,
                transformed_output,
            )
            if len(non_mergeable_right_hand_transforms) > 0
            else transformed_output
        )
        return transformed_output


@TransformationMixin.implements(nn.Linear)
class TransformedLinear(_NullaryOrUnaryTransformedLayer, nn.Linear):
    pass


@QuantizationMixin.implements(TransformedLinear)
class QuantizedTransformedLinear(
    _NullaryOrUnaryQuantizedTransformedLayer, TransformedLinear
):
    pass


@TransformationMixin.implements(nn.Embedding)
class TransformedEmbedding(_NullaryOrUnaryTransformedLayer, nn.Embedding):
    def _compute_merged_params(self):
        weight = self.weight.data

        if len(self.left_hand_transforms) > 0:
            transformed_input = functools.reduce(
                lambda x, transform: transform(x),
                self.left_hand_transforms,
                torch.eye(self.weight.data.shape[0], device=self.weight.data.device),
            )
            weight = transformed_input @ weight

        if len(self.right_hand_transforms) > 0:
            weight = functools.reduce(
                lambda x, transform: transform(x), self.right_hand_transforms, weight
            )

        return weight, None


@QuantizationMixin.implements(TransformedEmbedding)
class QuantizedTransformedEmbedding(
    _NullaryOrUnaryQuantizedTransformedLayer, TransformedEmbedding
):
    pass


def remove_all_transforms(model: torch.nn.Module):
    with contextlib.ExitStack() as stack:
        for _, module in model.named_modules():
            if isinstance(module, TransformationMixin):
                stack.enter_context(
                    patch_attr(module, "left_hand_transforms", nn.ModuleList())
                )
                stack.enter_context(
                    patch_attr(module, "right_hand_transforms", nn.ModuleList())
                )
    return stack


# pylint: disable=protected-access
def merge_transforms(model: torch.nn.Module):
    with contextlib.ExitStack() as stack:
        for _, module in model.named_modules():
            if isinstance(module, TransformationMixin):
                stack.enter_context(module._patch_transformed_parameters())
    return stack


def recompute_param_encodings_for_transformed_layers(model: torch.nn.Module):
    # Temporarily merge weights and recompute param encodings
    with merge_transforms(model):
        compute_param_encodings(model)


def get_trainable_transform_params(model: torch.nn.Module):
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, TransformationMixin):
            state_dict.update(module.state_dict(prefix=f"{name}."))
    return state_dict
