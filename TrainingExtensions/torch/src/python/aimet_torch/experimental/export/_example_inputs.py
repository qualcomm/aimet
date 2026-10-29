# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=protected-access

from typing import Callable
import torch
from torch._ops import OpOverload


_example_inputs_registry: dict[OpOverload, Callable[[], tuple[tuple, dict]]] = {}


def register_example_inputs(op: OpOverload):
    def decorator(func: Callable[[], tuple[tuple, dict]]):
        _example_inputs_registry[op] = func
        return func

    return decorator


@register_example_inputs(torch.ops.aten._adaptive_avg_pool2d.default)
def _adaptive_avg_pool2d_example_inputs():
    return (torch.randn(1, 3, 8, 8), [4, 4]), {}


@register_example_inputs(torch.ops.aten._log_softmax.default)
def _log_softmax_example_inputs():
    return (torch.randn(2, 3), 1, False), {}


@register_example_inputs(torch.ops.aten._native_batch_norm_legit.default)
def _native_batch_norm_legit_example_inputs():
    return (torch.randn(2, 3, 4, 4),), {
        "weight": torch.ones(3),
        "bias": torch.zeros(3),
        "running_mean": torch.zeros(3),
        "running_var": torch.ones(3),
        "training": False,
        "momentum": 0.1,
        "eps": 1e-5,
    }


@register_example_inputs(torch.ops.aten._native_batch_norm_legit_no_training.default)
def _native_batch_norm_legit_no_training_example_inputs():
    args, kwargs = _native_batch_norm_legit_example_inputs()
    kwargs.pop("training")
    return args, kwargs


@register_example_inputs(torch.ops.aten._prelu_kernel.default)
def _prelu_kernel_example_inputs():
    return (torch.randn(2, 3, 4, 4), torch.tensor([0.25])), {}


@register_example_inputs(torch.ops.aten._safe_softmax.default)
def _safe_softmax_example_inputs():
    return (torch.randn(2, 3), 1, None), {}


@register_example_inputs(torch.ops.aten._softmax.default)
def _softmax_example_inputs():
    return (torch.randn(2, 3), 1, False), {}


@register_example_inputs(torch.ops.aten.abs.default)
def _abs_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.absolute.default)
def _absolute_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.adaptive_avg_pool2d.default)
def _adaptive_avg_pool2d_op_example_inputs():
    return (torch.randn(1, 3, 8, 8), [4, 4]), {}


@register_example_inputs(torch.ops.aten.add.Tensor)
def _add_tensor_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3)), {}


@register_example_inputs(torch.ops.aten.addmm.default)
def _addmm_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 4), torch.randn(4, 3)), {}


@register_example_inputs(torch.ops.aten.amax.default)
def _amax_example_inputs():
    return (torch.randn(2, 3, 4), [1]), {}


@register_example_inputs(torch.ops.aten.amin.default)
def _amin_example_inputs():
    return (torch.randn(2, 3, 4), [1]), {}


@register_example_inputs(torch.ops.aten.avg_pool1d.default)
def _avg_pool1d_example_inputs():
    return (torch.randn(1, 3, 10), [3]), {}


@register_example_inputs(torch.ops.aten.avg_pool2d.default)
def _avg_pool2d_example_inputs():
    return (torch.randn(1, 3, 8, 8), [2, 2]), {}


@register_example_inputs(torch.ops.aten.avg_pool3d.default)
def _avg_pool3d_example_inputs():
    return (torch.randn(1, 3, 4, 4, 4), [2, 2, 2]), {}


@register_example_inputs(torch.ops.aten.batch_norm.default)
def _batch_norm_example_inputs():
    return (torch.randn(2, 3, 4, 4),), {
        "weight": torch.ones(3),
        "bias": torch.zeros(3),
        "running_mean": torch.ones(3),
        "running_var": torch.zeros(3),
        "training": False,
        "momentum": 0.1,
        "eps": 1e-5,
        "cudnn_enabled": False,
    }


@register_example_inputs(torch.ops.aten.bmm.default)
def _bmm_example_inputs():
    return (torch.randn(5, 3, 4), torch.randn(5, 4, 2)), {}


@register_example_inputs(torch.ops.aten.cat.default)
def _cat_example_inputs():
    return ([torch.randn(2, 3), torch.randn(2, 3)], 0), {}


@register_example_inputs(torch.ops.aten.channel_shuffle.default)
def _channel_shuffle_example_inputs():
    return (torch.randn(1, 4, 8, 8), 2), {}


@register_example_inputs(torch.ops.aten.clamp.default)
def _clamp_example_inputs():
    return (torch.randn(2, 3), -1.0, 1.0), {}


@register_example_inputs(torch.ops.aten.clip.default)
def _clip_example_inputs():
    return (torch.randn(2, 3), -1.0, 1.0), {}


@register_example_inputs(torch.ops.aten.col2im.default)
def _col2im_example_inputs():
    return (
        torch.randn(1, 27, 4),
        [4, 4],
        [3, 3],
        [1, 1],
        [0, 0],
        [1, 1],
    ), {}


@register_example_inputs(torch.ops.aten.constant_pad_nd.default)
def _constant_pad_nd_example_inputs():
    return (torch.randn(1, 3, 4, 4), [1, 1, 1, 1], 0.0), {}


@register_example_inputs(torch.ops.aten.conv1d.default)
def _conv1d_example_inputs():
    return (torch.randn(1, 3, 8),), {
        "weight": torch.randn(4, 3, 3),
        "bias": torch.randn(4),
        "stride": [1],
        "padding": [0],
        "dilation": [1],
        "groups": 1,
    }


@register_example_inputs(torch.ops.aten.conv2d.default)
def _conv2d_example_inputs():
    return (torch.randn(1, 3, 8, 8),), {
        "weight": torch.randn(4, 3, 3, 3),
        "bias": torch.randn(4),
        "stride": [1, 1],
        "padding": [0, 0],
        "dilation": [1, 1],
        "groups": 1,
    }


@register_example_inputs(torch.ops.aten.conv3d.default)
def _conv3d_example_inputs():
    return (torch.randn(1, 3, 4, 4, 4),), {
        "weight": torch.randn(4, 3, 3, 3, 3),
        "bias": torch.randn(4),
        "stride": [1, 1, 1],
        "padding": [0, 0, 0],
        "dilation": [1, 1, 1],
        "groups": 1,
    }


@register_example_inputs(torch.ops.aten.conv_transpose1d.default)
def _conv_transpose1d_example_inputs():
    return (torch.randn(1, 3, 8),), {
        "weight": torch.randn(3, 4, 3),
        "bias": torch.randn(4),
        "stride": [1],
        "padding": [0],
        "output_padding": [0],
        "groups": 1,
        "dilation": [1],
    }


@register_example_inputs(torch.ops.aten.conv_transpose2d.input)
def _conv_transpose2d_input_example_inputs():
    return (torch.randn(1, 3, 8, 8),), {
        "weight": torch.randn(3, 4, 3, 3),
        "bias": torch.randn(4),
        "stride": [1, 1],
        "padding": [0, 0],
        "output_padding": [0, 0],
        "groups": 1,
        "dilation": [1, 1],
    }


@register_example_inputs(torch.ops.aten.conv_transpose3d.input)
def _conv_transpose3d_input_example_inputs():
    return (torch.randn(1, 3, 4, 4, 4),), {
        "weight": torch.randn(3, 4, 3, 3, 3),
        "bias": torch.randn(4),
        "stride": [1, 1, 1],
        "padding": [0, 0, 0],
        "output_padding": [0, 0, 0],
        "groups": 1,
        "dilation": [1, 1, 1],
    }


@register_example_inputs(torch.ops.aten.convolution.default)
def _convolution_example_inputs():
    return (torch.randn(1, 3, 8, 8),), {
        "weight": torch.randn(4, 3, 3, 3),
        "bias": None,
        "stride": [1, 1],
        "padding": [0, 0],
        "dilation": [1, 1],
        "transposed": False,
        "output_padding": [0, 0],
        "groups": 1,
    }


@register_example_inputs(torch.ops.aten.copy.default)
def _copy_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3), False), {}


@register_example_inputs(torch.ops.aten.detach.default)
def _detach_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.div.Tensor)
def _div_tensor_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3)), {}


@register_example_inputs(torch.ops.aten.div.Tensor_mode)
def _div_tensor_mode_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3)), {"rounding_mode": "floor"}


@register_example_inputs(torch.ops.aten.dot.default)
def _dot_example_inputs():
    return (torch.randn(5), torch.randn(5)), {}


@register_example_inputs(torch.ops.aten.elu.default)
def _elu_example_inputs():
    return (torch.randn(2, 3), 1.0, 1.0, 1.0), {}


@register_example_inputs(torch.ops.aten.embedding.default)
def _embedding_example_inputs():
    return (
        torch.randn(10, 3),
        torch.tensor([0, 2, 5]),
        -1,
        False,
        False,
    ), {}


@register_example_inputs(torch.ops.aten.eq.Tensor)
def _eq_tensor_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3)), {}


@register_example_inputs(torch.ops.aten.exp.default)
def _exp_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.gather.default)
def _gather_example_inputs():
    return (
        torch.randn(2, 3),
        1,
        torch.tensor([[0, 1, 2], [2, 1, 0]]),
    ), {}


@register_example_inputs(torch.ops.aten.ge.Tensor)
def _ge_tensor_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3)), {}


@register_example_inputs(torch.ops.aten.gelu.default)
def _gelu_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.grid_sampler_2d.default)
def _grid_sampler_2d_example_inputs():
    return (
        torch.randn(1, 3, 8, 8),
        torch.randn(1, 4, 4, 2),
        0,
        0,
        True,
    ), {}


@register_example_inputs(torch.ops.aten.group_norm.default)
def _group_norm_example_inputs():
    return (
        torch.randn(2, 4, 5),
        2,
        torch.ones(4),
        torch.zeros(4),
        1e-5,
    ), {}


@register_example_inputs(torch.ops.aten.gt.Tensor)
def _gt_tensor_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3)), {}


@register_example_inputs(torch.ops.aten.hardswish.default)
def _hardswish_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.hardtanh.default)
def _hardtanh_example_inputs():
    return (torch.randn(2, 3), -1.0, 1.0), {}


@register_example_inputs(torch.ops.aten.im2col.default)
def _im2col_example_inputs():
    return (
        torch.randn(1, 3, 4, 4),
        [3, 3],
        [1, 1],
        [0, 0],
        [1, 1],
    ), {}


@register_example_inputs(torch.ops.aten.index_add.default)
def _index_add_example_inputs():
    return (
        torch.randn(3, 4),
        0,
        torch.tensor([0, 2]),
        torch.randn(2, 4),
    ), {}


@register_example_inputs(torch.ops.aten.index_copy.default)
def _index_copy_example_inputs():
    return (
        torch.randn(3, 4),
        0,
        torch.tensor([0, 2]),
        torch.randn(2, 4),
    ), {}


# @register_example_inputs(torch.ops.aten.index_fill.default)
# def _index_fill_example_inputs():
#     return (
#         torch.randn(3, 4),
#         0,
#         torch.tensor([0, 2]),
#         1.0,
#     ), {}


@register_example_inputs(torch.ops.aten.index_put.default)
def _index_put_example_inputs():
    return (
        torch.randn(3, 4),
        [torch.tensor([0, 2])],
        torch.randn(2, 4),
        False,
    ), {}


@register_example_inputs(torch.ops.aten.index_select.default)
def _index_select_example_inputs():
    return (
        torch.randn(3, 4),
        0,
        torch.tensor([0, 2]),
    ), {}


@register_example_inputs(torch.ops.aten.instance_norm.default)
def _instance_norm_example_inputs():
    return (torch.randn(2, 3, 4, 4),), {
        "weight": torch.ones(3),
        "bias": torch.zeros(3),
        "running_mean": torch.ones(3),
        "running_var": torch.zeros(3),
        "use_input_stats": True,
        "momentum": 0.1,
        "eps": 1e-5,
        "cudnn_enabled": False,
    }


@register_example_inputs(torch.ops.aten.le.Tensor)
def _le_tensor_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3)), {}


@register_example_inputs(torch.ops.aten.leaky_relu.default)
def _leaky_relu_example_inputs():
    return (torch.randn(2, 3), 0.01), {}


@register_example_inputs(torch.ops.aten.linear.default)
def _linear_example_inputs():
    return (torch.randn(2, 3),), {"weight": torch.randn(4, 3), "bias": torch.randn(4)}


@register_example_inputs(torch.ops.aten.log.default)
def _log_example_inputs():
    return (torch.randn(2, 3).abs() + 1,), {}


# @register_example_inputs(torch.ops.aten.log_softmax.default)
# def _log_softmax_op_example_inputs():
#     return (torch.randn(2, 3), 1), {}


@register_example_inputs(torch.ops.aten.logical_and.default)
def _logical_and_example_inputs():
    return (
        torch.randn(2, 3) > 0,
        torch.randn(2, 3) > 0,
    ), {}


@register_example_inputs(torch.ops.aten.logical_not.default)
def _logical_not_example_inputs():
    return (torch.randn(2, 3) > 0,), {}


@register_example_inputs(torch.ops.aten.logical_or.default)
def _logical_or_example_inputs():
    return (
        torch.randn(2, 3) > 0,
        torch.randn(2, 3) > 0,
    ), {}


@register_example_inputs(torch.ops.aten.logical_xor.default)
def _logical_xor_example_inputs():
    return (
        torch.randn(2, 3) > 0,
        torch.randn(2, 3) > 0,
    ), {}


@register_example_inputs(torch.ops.aten.lt.Tensor)
def _lt_tensor_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3)), {}


@register_example_inputs(torch.ops.aten.masked_fill.Scalar)
def _masked_fill_scalar_example_inputs():
    return (
        torch.randn(2, 3),
        torch.randn(2, 3) > 0,
        0.0,
    ), {}


@register_example_inputs(torch.ops.aten.masked_scatter.default)
def _masked_scatter_example_inputs():
    return (
        torch.randn(2, 3),
        torch.arange(6).reshape(2, 3) < 3,
        torch.randn(3),
    ), {}


@register_example_inputs(torch.ops.aten.matmul.default)
def _matmul_example_inputs():
    return (torch.randn(2, 3), torch.randn(3, 4)), {}


@register_example_inputs(torch.ops.aten.mean.default)
def _mean_example_inputs():
    return (torch.randn(2, 3, 4),), {}


@register_example_inputs(torch.ops.aten.mm.default)
def _mm_example_inputs():
    return (torch.randn(2, 3), torch.randn(3, 4)), {}


@register_example_inputs(torch.ops.aten.mul.Tensor)
def _mul_tensor_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3)), {}


@register_example_inputs(torch.ops.aten.mv.default)
def _mv_example_inputs():
    return (torch.randn(3, 4), torch.randn(4)), {}


@register_example_inputs(torch.ops.aten.narrow.default)
def _narrow_example_inputs():
    return (torch.randn(3, 4, 5), 1, 1, 2), {}


@register_example_inputs(torch.ops.aten.native_batch_norm.default)
def _native_batch_norm_example_inputs():
    return (
        torch.randn(2, 3, 4, 4),
        torch.ones(3),
        torch.zeros(3),
        torch.ones(3),
        torch.zeros(3),
        False,
        0.1,
        1e-5,
    ), {}


@register_example_inputs(torch.ops.aten.native_group_norm.default)
def _native_group_norm_example_inputs():
    return (
        torch.randn(2, 4, 5),
        torch.ones(4),
        torch.zeros(4),
        2,
        4,
        5,
        2,
        1e-5,
    ), {}


@register_example_inputs(torch.ops.aten.native_layer_norm.default)
def _native_layer_norm_example_inputs():
    return (torch.randn(2, 3, 4),), {
        "normalized_shape": [4],
        "weight": torch.ones(4),
        "bias": torch.zeros(4),
        "eps": 1e-5,
    }


@register_example_inputs(torch.ops.aten.neg.default)
def _neg_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.pixel_shuffle.default)
def _pixel_shuffle_example_inputs():
    return (torch.randn(1, 4, 8, 8), 2), {}


@register_example_inputs(torch.ops.aten.pixel_unshuffle.default)
def _pixel_unshuffle_example_inputs():
    return (torch.randn(1, 4, 8, 8), 2), {}


@register_example_inputs(torch.ops.aten.pow.Scalar)
def _pow_scalar_example_inputs():
    return (2.0, torch.randn(2, 3)), {}


@register_example_inputs(torch.ops.aten.pow.Tensor_Scalar)
def _pow_tensor_scalar_example_inputs():
    return (torch.randn(2, 3), 2.0), {}


@register_example_inputs(torch.ops.aten.pow.Tensor_Tensor)
def _pow_tensor_tensor_example_inputs():
    return (torch.randn(2, 3), torch.ones(2, 3)), {}


@register_example_inputs(torch.ops.aten.prelu.default)
def _prelu_example_inputs():
    return (
        torch.randn(2, 3, 4, 4),
        torch.ones(3) * 0.25,
    ), {}


@register_example_inputs(torch.ops.aten.reciprocal.default)
def _reciprocal_example_inputs():
    return (torch.randn(2, 3) + 2.0,), {}


@register_example_inputs(torch.ops.aten.reflection_pad1d.default)
def _reflection_pad1d_example_inputs():
    return (torch.randn(1, 3, 8), [2, 2]), {}


@register_example_inputs(torch.ops.aten.reflection_pad2d.default)
def _reflection_pad2d_example_inputs():
    return (
        torch.randn(1, 3, 4, 4),
        [1, 1, 1, 1],
    ), {}


@register_example_inputs(torch.ops.aten.reflection_pad3d.default)
def _reflection_pad3d_example_inputs():
    return (
        torch.randn(1, 3, 4, 4, 4),
        [1, 1, 1, 1, 1, 1],
    ), {}


@register_example_inputs(torch.ops.aten.relu6.default)
def _relu6_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.relu.default)
def _relu_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.repeat.default)
def _repeat_example_inputs():
    return (torch.randn(2, 3), [2, 1]), {}


# @register_example_inputs(torch.ops.aten.repeat_interleave.default)
# def _repeat_interleave_example_inputs():
#     return (torch.randn(2, 3), 2, 0), {}


@register_example_inputs(torch.ops.aten.replication_pad1d.default)
def _replication_pad1d_example_inputs():
    return (torch.randn(1, 3, 8), [2, 2]), {}


@register_example_inputs(torch.ops.aten.replication_pad2d.default)
def _replication_pad2d_example_inputs():
    return (
        torch.randn(1, 3, 4, 4),
        [1, 1, 1, 1],
    ), {}


@register_example_inputs(torch.ops.aten.replication_pad3d.default)
def _replication_pad3d_example_inputs():
    return (
        torch.randn(1, 3, 4, 4, 4),
        [1, 1, 1, 1, 1, 1],
    ), {}


@register_example_inputs(torch.ops.aten.rms_norm.default)
def _rms_norm_example_inputs():
    return (torch.randn(2, 3, 4),), {
        "normalized_shape": [4],
        "weight": torch.ones(4),
        "eps": 1e-5,
    }


@register_example_inputs(torch.ops.aten.rsqrt.default)
def _rsqrt_example_inputs():
    return (torch.randn(2, 3).abs(),), {}


# @register_example_inputs(torch.ops.aten.scatter.default)
# def _scatter_example_inputs():
#     return (
#         torch.randn(2, 3),
#         1,
#         torch.tensor([[0, 1, 2], [2, 1, 0]]),
#         torch.randn(2, 3),
#     ), {}


@register_example_inputs(torch.ops.aten.scatter_add.default)
def _scatter_add_example_inputs():
    return (
        torch.randn(2, 3),
        1,
        torch.tensor([[0, 1, 2], [2, 1, 0]]),
        torch.randn(2, 3),
    ), {}


# @register_example_inputs(torch.ops.aten.scatter_reduce.default)
# def _scatter_reduce_example_inputs():
#     return (
#         torch.randn(2, 3),
#         1,
#         torch.tensor([[0, 1, 2], [2, 1, 0]]),
#         torch.randn(2, 3),
#         "sum",
#     ), {}


# @register_example_inputs(torch.ops.aten.select.default)
# def _select_example_inputs():
#     return (torch.randn(2, 3, 4), 1, 1), {}


@register_example_inputs(torch.ops.aten.sigmoid.default)
def _sigmoid_example_inputs():
    return (torch.randn(2, 3),), {}


# @register_example_inputs(torch.ops.aten.slice.default)
# def _slice_example_inputs():
#     return (torch.randn(2, 3, 4), 1, 0, 2, 1), {}


@register_example_inputs(torch.ops.aten.split.Tensor)
def _split_tensor_example_inputs():
    return (torch.randn(4, 3), 2, 0), {}


@register_example_inputs(torch.ops.aten.split_with_sizes.default)
def _split_with_sizes_example_inputs():
    return (torch.randn(6, 3), [2, 1, 3], 0), {}


@register_example_inputs(torch.ops.aten.split_with_sizes_copy.default)
def _split_with_sizes_copy_example_inputs():
    return (
        torch.randn(6, 3),
        [2, 1, 3],
        0,
    ), {}


@register_example_inputs(torch.ops.aten.sqrt.default)
def _sqrt_example_inputs():
    return (torch.randn(2, 3).abs(),), {}


@register_example_inputs(torch.ops.aten.stack.default)
def _stack_example_inputs():
    return ([torch.randn(2, 3), torch.randn(2, 3)], 0), {}


@register_example_inputs(torch.ops.aten.sub.Tensor)
def _sub_tensor_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3)), {}


@register_example_inputs(torch.ops.aten.sum.default)
def _sum_example_inputs():
    return (torch.randn(2, 3, 4),), {}


@register_example_inputs(torch.ops.aten.tanh.default)
def _tanh_example_inputs():
    return (torch.randn(2, 3),), {}


# @register_example_inputs(torch.ops.aten.unbind.default)
# def _unbind_example_inputs():
#     return (torch.randn(3, 2, 4), 0), {}


@register_example_inputs(torch.ops.aten.upsample_bicubic2d.default)
def _upsample_bicubic2d_example_inputs():
    return (
        torch.randn(1, 3, 4, 4),
        [8, 8],
        True,
        None,
        None,
    ), {}


@register_example_inputs(torch.ops.aten.upsample_linear1d.default)
def _upsample_linear1d_example_inputs():
    return (
        torch.randn(1, 3, 4),
        [8],
        True,
        None,
    ), {}


@register_example_inputs(torch.ops.aten.vdot.default)
def _vdot_example_inputs():
    return (torch.randn(5), torch.randn(5)), {}


@register_example_inputs(torch.ops.aten.where.self)
def _where_self_example_inputs():
    return (
        torch.randn(2, 3) > 0,
        torch.randn(2, 3),
        torch.randn(2, 3),
    ), {}


@register_example_inputs(torch.ops.aten._to_copy.default)
def _to_copy_example_inputs():
    return (torch.randn(2, 3),), {}


# @register_example_inputs(torch.ops.aten.add.Scalar)
# def _add_scalar_example_inputs():
#     return (torch.randn(2, 3), 1.0), {}


@register_example_inputs(torch.ops.aten.alias.default)
def _alias_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.any.default)
def _any_example_inputs():
    return (torch.randint(0, 2, (2, 3)).bool(),), {}


@register_example_inputs(torch.ops.aten.arange.default)
def _arange_example_inputs():
    return (5,), {}


@register_example_inputs(torch.ops.aten.arange.start_step)
def _arange_start_step_example_inputs():
    return (0, 10, 2), {}


@register_example_inputs(torch.ops.aten.argmax.default)
def _argmax_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.argmin.default)
def _argmin_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.as_strided.default)
@register_example_inputs(torch.ops.aten.as_strided_copy.default)
def _as_strided_example_inputs():
    return (torch.randn(4, 4), [2, 2], [4, 1], 0), {}


@register_example_inputs(torch.ops.aten.asin.default)
def _asin_example_inputs():
    return (torch.tensor([-1.0, 0.0, 1.0]),), {}


@register_example_inputs(torch.ops.aten.atan.default)
def _atan_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.ceil.default)
def _ceil_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.clone.default)
def _clone_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.cos.default)
def _cos_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.cumsum.default)
def _cumsum_example_inputs():
    return (torch.randn(2, 3), 1), {}


# @register_example_inputs(torch.ops.aten.detach_copy.default)
# def _detach_copy_example_inputs():
#     return (torch.randn(2, 3),), {}


# @register_example_inputs(torch.ops.aten.div.Scalar)
# def _div_scalar_example_inputs():
#     return (torch.randn(2, 3), 2.0), {}


# @register_example_inputs(torch.ops.aten.div.Tensor_mode)
# def _div_tensor_mode_example_inputs():
#     return (torch.randn(2, 3), torch.randn(2, 3).abs() + 0.1), {"rounding_mode": "floor"}


@register_example_inputs(torch.ops.aten.empty.memory_format)
def _empty_memory_format_example_inputs():
    return ([2, 3],), {}


# @register_example_inputs(torch.ops.aten.eq.Scalar)
# def _eq_scalar_example_inputs():
#     return (torch.randn(2, 3), 0.0), {}


# @register_example_inputs(torch.ops.aten.erf.default)
# def _erf_example_inputs():
#     return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.expand.default)
@register_example_inputs(torch.ops.aten.expand_copy.default)
def _expand_example_inputs():
    return (torch.randn(1, 3), [2, 3]), {}


# @register_example_inputs(torch.ops.aten.fill.Scalar)
# def _fill_scalar_example_inputs():
#     return (torch.randn(2, 3), 1.0), {}


@register_example_inputs(torch.ops.aten.fill.Tensor)
def _fill_tensor_example_inputs():
    return (torch.randn(2, 3), torch.tensor(1.0)), {}


@register_example_inputs(torch.ops.aten.floor.default)
def _floor_example_inputs():
    return (torch.randn(2, 3),), {}


# @register_example_inputs(torch.ops.aten.floor_divide.default)
# def _floor_divide_example_inputs():
#     return (torch.randn(2, 3).abs() + 1, torch.randn(2, 3).abs() + 1), {}


# @register_example_inputs(torch.ops.aten.fmod.Scalar)
# def _fmod_scalar_example_inputs():
#     return (torch.randn(2, 3), 2.0), {}


@register_example_inputs(torch.ops.aten.fmod.Tensor)
def _fmod_tensor_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3).abs() + 0.1), {}


@register_example_inputs(torch.ops.aten.full.default)
def _full_example_inputs():
    return ([2, 3], 1.0), {}


@register_example_inputs(torch.ops.aten.full_like.default)
def _full_like_example_inputs():
    return (torch.randn(2, 3), 1.0), {}


# @register_example_inputs(torch.ops.aten.ge.Scalar)
# def _ge_scalar_example_inputs():
#     return (torch.randn(2, 3), 0.0), {}


@register_example_inputs(torch.ops.aten.glu.default)
def _glu_example_inputs():
    return (torch.randn(2, 4), 1), {}


# @register_example_inputs(torch.ops.aten.gt.Scalar)
# def _gt_scalar_example_inputs():
#     return (torch.randn(2, 3), 0.0), {}


@register_example_inputs(torch.ops.aten.index.Tensor)
def _index_tensor_example_inputs():
    return (torch.randn(4, 4), [torch.tensor([0, 2])]), {}


@register_example_inputs(torch.ops.aten.isinf.default)
def _isinf_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.isnan.default)
def _isnan_example_inputs():
    return (torch.randn(2, 3),), {}


# @register_example_inputs(torch.ops.aten.le.Scalar)
# def _le_scalar_example_inputs():
#     return (torch.randn(2, 3), 0.0), {}


# @register_example_inputs(torch.ops.aten.logit.default)
# def _logit_example_inputs():
#     return (torch.rand(2, 3),), {}


# @register_example_inputs(torch.ops.aten.lt.Scalar)
# def _lt_scalar_example_inputs():
#     return (torch.randn(2, 3), 0.0), {}


@register_example_inputs(torch.ops.aten.max.dim)
def _max_dim_example_inputs():
    return (torch.randn(2, 3), 1, False), {}


@register_example_inputs(torch.ops.aten.max_pool1d_with_indices.default)
def _max_pool1d_with_indices_example_inputs():
    return (torch.randn(1, 3, 10), [3]), {}


@register_example_inputs(torch.ops.aten.max_pool2d_with_indices.default)
def _max_pool2d_with_indices_example_inputs():
    return (torch.randn(1, 3, 8, 8), [2, 2]), {}


@register_example_inputs(torch.ops.aten.max_pool3d_with_indices.default)
def _max_pool3d_with_indices_example_inputs():
    return (torch.randn(1, 3, 4, 4, 4), [2, 2, 2]), {}


@register_example_inputs(torch.ops.aten.mean.dim)
def _mean_dim_example_inputs():
    return (torch.randn(2, 3, 4), [1], False), {}


@register_example_inputs(torch.ops.aten.min.dim)
def _min_dim_example_inputs():
    return (torch.randn(2, 3), 1, False), {}


@register_example_inputs(torch.ops.aten.maximum.default)
@register_example_inputs(torch.ops.aten.minimum.default)
def _minimum_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3)), {}


# @register_example_inputs(torch.ops.aten.mul.Scalar)
# def _mul_scalar_example_inputs():
#     return (torch.randn(2, 3), 2.0), {}


# @register_example_inputs(torch.ops.aten.ne.Scalar)
# def _ne_scalar_example_inputs():
#     return (torch.randn(2, 3), 0.0), {}


@register_example_inputs(torch.ops.aten.ne.Tensor)
def _ne_tensor_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 3)), {}


@register_example_inputs(torch.ops.aten.nonzero.default)
def _nonzero_example_inputs():
    return (torch.arange(0, 3, dtype=torch.float32),), {}


@register_example_inputs(torch.ops.aten.ones.default)
def _ones_example_inputs():
    return ([2, 3],), {}


@register_example_inputs(torch.ops.aten.permute.default)
@register_example_inputs(torch.ops.aten.permute_copy.default)
def _permute_example_inputs():
    return (torch.randn(2, 3, 4), [2, 0, 1]), {}


# @register_example_inputs(torch.ops.aten.remainder.Scalar)
# def _remainder_scalar_example_inputs():
#     return (torch.randn(2, 3), 2.0), {}


# @register_example_inputs(torch.ops.aten.remainder.Tensor)
# def _remainder_tensor_example_inputs():
#     return (torch.randn(2, 3), torch.randn(2, 3).abs() + 0.1), {}


@register_example_inputs(torch.ops.aten.round.default)
def _round_example_inputs():
    return (torch.randn(2, 3),), {}


# @register_example_inputs(torch.ops.aten.rsub.Scalar)
# def _rsub_scalar_example_inputs():
#     return (torch.randn(2, 3), 1.0), {}


@register_example_inputs(torch.ops.aten.scalar_tensor.default)
def _scalar_tensor_example_inputs():
    return (1.0,), {}


# @register_example_inputs(torch.ops.aten.select_copy.int)
# def _select_copy_int_example_inputs():
#     return (torch.randn(2, 3, 4), 1, 0), {}


@register_example_inputs(torch.ops.aten.select_scatter.default)
def _select_scatter_example_inputs():
    return (torch.randn(2, 3), torch.randn(3), 0, 1), {}


@register_example_inputs(torch.ops.aten.sign.default)
def _sign_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.sin.default)
def _sin_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.slice.Tensor)
def _slice_tensor_example_inputs():
    return (torch.randn(2, 3, 4), 1, 0, 2, 1), {}


# @register_example_inputs(torch.ops.aten.slice_copy.Tensor)
# def _slice_copy_tensor_example_inputs():
#     return (torch.randn(2, 3, 4), 1, 0, 2, 1), {}


@register_example_inputs(torch.ops.aten.slice_scatter.default)
def _slice_scatter_example_inputs():
    return (torch.randn(2, 3), torch.randn(2, 2), 1, 0, 2, 1), {}


# @register_example_inputs(torch.ops.aten.split_copy.Tensor)
# def _split_copy_tensor_example_inputs():
#     return (torch.randn(4, 3), 2, 0), {}


@register_example_inputs(torch.ops.aten.squeeze.dim)
@register_example_inputs(torch.ops.aten.squeeze_copy.dim)
def _squeeze_dim_example_inputs():
    return (torch.randn(2, 1, 3), 1), {}


@register_example_inputs(torch.ops.aten.squeeze.dims)
@register_example_inputs(torch.ops.aten.squeeze_copy.dims)
def _squeeze_dims_example_inputs():
    return (torch.randn(2, 1, 3, 1), [1, 3]), {}


# @register_example_inputs(torch.ops.aten.sub.Scalar)
# def _sub_scalar_example_inputs():
#     return (torch.randn(2, 3), 1.0), {}


@register_example_inputs(torch.ops.aten.sum.dim_IntList)
def _sum_dim_intlist_example_inputs():
    return (torch.randn(2, 3, 4), [1]), {}


@register_example_inputs(torch.ops.aten.sym_size.int)
def _sym_size_int():
    return (torch.randn(2, 3, 4), 1), {}


@register_example_inputs(torch.ops.aten.t_copy.default)
def _t_copy_example_inputs():
    return (torch.randn(2, 3),), {}


@register_example_inputs(torch.ops.aten.transpose_copy.int)
def _transpose_copy_int_example_inputs():
    return (torch.randn(2, 3, 4), 0, 1), {}


@register_example_inputs(torch.ops.aten.tril.default)
def _tril_example_inputs():
    return (torch.randn(3, 3),), {}


@register_example_inputs(torch.ops.aten.unbind.int)
@register_example_inputs(torch.ops.aten.unbind_copy.int)
def _unbind_int_example_inputs():
    return (torch.randn(3, 2, 4), 0), {}


@register_example_inputs(torch.ops.aten.unsqueeze.default)
@register_example_inputs(torch.ops.aten.unsqueeze_copy.default)
def _unsqueeze_example_inputs():
    return (torch.randn(2, 3), 1), {}


# @register_example_inputs(torch.ops.aten.var.dim)
# def _var_dim_example_inputs():
#     return (torch.randn(2, 3, 4), [1]), {}


@register_example_inputs(torch.ops.aten.view.default)
def _view_example_inputs():
    return (torch.randn(2, 3, 4), [2, 12]), {}


@register_example_inputs(torch.ops.aten.zeros.default)
def _zeros_example_inputs():
    return ([2, 3],), {}
