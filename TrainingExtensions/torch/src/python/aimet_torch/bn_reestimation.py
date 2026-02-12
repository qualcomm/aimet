# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""BatchNorm Reestimation"""

import itertools
from typing import Iterable, List, Callable, Any

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.modules.batchnorm import _BatchNorm
from aimet_torch.utils import in_eval_mode, in_train_mode
from aimet_torch.common.utils import Handle


def _get_active_bn_modules(model: torch.nn.Module) -> Iterable[_BatchNorm]:
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            bn = module
            if bn.running_mean is not None and bn.running_var is not None:
                yield bn


def _for_each_module(
    modules: Iterable[torch.nn.Module], action: Callable[[torch.nn.Module], Handle]
) -> Handle:
    """
    Apply an undoable action to each module.

    :param modules: Modules to apply the action.
    :param action: Action to be applied to the modules.
    :returns: Handle that undos the applied action.
    """

    handles: List[Handle] = []

    def cleanup():
        for handle in handles:
            handle.remove()

    try:
        for module in modules:
            handle = action(module)
            assert isinstance(handle, Handle)
            handles.append(handle)
        return Handle(cleanup)
    except Exception:
        cleanup()
        raise


def _reset_bn_stats(module: _BatchNorm) -> Handle:
    """
    Reset BN statistics to the initial values.

    :param module: BatchNorm module.
    :returns: Handle that restores the original BN statistics upon handle.remove().
    """
    orig_running_mean = module.running_mean.clone()
    orig_running_var = module.running_var.clone()
    orig_num_batches_tracked = module.num_batches_tracked.clone()

    def cleanup():
        module.running_mean.copy_(orig_running_mean)
        module.running_var.copy_(orig_running_var)
        module.num_batches_tracked.copy_(orig_num_batches_tracked)

    try:
        module.reset_running_stats()
        return Handle(cleanup)
    except Exception:
        cleanup()
        raise


def _reset_momentum(module: _BatchNorm) -> Handle:
    """
    Set BN momentum to 1.0.

    :param module: BatchNorm module.
    :returns: Handle that restores the original BN momentum upon handle.remove().
    """
    momentum = module.momentum

    def cleanup():
        module.momentum = momentum

    try:
        module.momentum = 1.0
        return Handle(cleanup)
    except Exception:
        cleanup()
        raise


DEFAULT_NUM_BATCHES = 100


def reestimate_bn_stats(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_batches: int = DEFAULT_NUM_BATCHES,
    forward_fn: Callable[[torch.nn.Module, Any], Any] = None,
) -> Handle:
    """
    Reestimate BatchNorm statistics (running mean and var).

    :param model: Model to reestimate the BN stats.
    :param dataloader: Training dataset.
    :param num_batches: The number of batches to be used for reestimation.
    :param forward_fn: Optional adapter function that performs forward pass
                       given a model and a input batch yielded from the data loader.
    :returns: Handle that undos the effect of BN reestimation upon handle.remove().
    """
    forward_fn = forward_fn or (lambda model, data: model(data))
    bn_modules = tuple(_get_active_bn_modules(model))

    # Set all the layers to eval mode except batchnorm layers
    with in_eval_mode(model), in_train_mode(bn_modules), torch.no_grad():
        with _for_each_module(bn_modules, action=_reset_momentum):
            handle = _for_each_module(bn_modules, action=_reset_bn_stats)

            try:
                # Batchnorm statistics accumulation buffer
                buffer = {
                    bn: {
                        "sum_mean": torch.zeros_like(bn.running_mean),
                        "sum_var": torch.zeros_like(bn.running_var),
                    }
                    for bn in bn_modules
                }

                num_batches = min(len(dataloader), num_batches)
                dataloader_slice = itertools.islice(dataloader, num_batches)

                for data in tqdm(
                    dataloader_slice, total=num_batches, desc="batchnorm reestimation"
                ):
                    forward_fn(model, data)

                    for bn in bn_modules:
                        buffer[bn]["sum_mean"] += bn.running_mean
                        buffer[bn]["sum_var"] += bn.running_var

                for bn in bn_modules:
                    sum_mean = buffer[bn]["sum_mean"]
                    sum_var = buffer[bn]["sum_var"]

                    # Override BN stats with the reestimated stats.
                    bn.running_mean.copy_(sum_mean / min(len(dataloader), num_batches))
                    bn.running_var.copy_(sum_var / min(len(dataloader), num_batches))

                return handle
            except Exception:
                handle.remove()
                raise
