# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Sequential MSE base"""

import functools
from abc import abstractmethod, ABC
import json
import os
import tempfile
import contextlib
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from aimet_torch.common.progress import progress_bar

from aimet_torch.common.utils import AimetLogger
from safetensors.torch import save_file, load_file

from aimet_torch.utils import (
    CachedDataset,
    get_ordered_list_of_modules,
    in_eval_mode,
    StopForwardException,
    change_tensor_device_placement,
    get_device,
)
from aimet_torch.nn import BaseQuantizationMixin
from aimet_torch._base.adaround.activation_sampler import (
    create_modulelist_for_group_modules,
    get_block_inputs,
    get_block_outputs,
)
from aimet_torch.utils import default_forward_fn

# The following modules with weights are supported
SUPPORTED_MODULES = (
    torch.nn.Linear,
    torch.nn.Conv2d,
)

# Skip running Sequential MSE if param BW is higher than supported PARAM_BW.
SUPPORTED_PARAM_BW = 4

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SeqMse)


@dataclass
class SeqMseParams:
    """
    Sequential MSE parameters

    :param num_batches: Number of batches.
    :param num_candidates: Number of candidates to perform grid search. Default 20.
    :param inp_symmetry: Input symmetry. Available options are 'asym', 'symfp' and 'symqt'. Default 'symqt'.
    :param loss_fn: Loss function. Available options are 'mse', 'l1' and 'sqnr'. Default 'mse'.
    :param forward_fn: Optional adapter function that performs forward pass given a model and inputs
     yielded from the data loader. The function expects model as first argument and inputs to model as second argument.
    """

    num_batches: Optional[int]
    num_candidates: int = 20
    inp_symmetry: str = "symqt"
    loss_fn: str = "mse"
    forward_fn: Callable = default_forward_fn

    def __post_init__(self):
        # pylint: disable=attribute-defined-outside-init
        if self.loss_fn == "mse":
            self._loss_fn = functional.mse_loss
        elif self.loss_fn == "l1":
            self._loss_fn = functional.l1_loss
        elif self.loss_fn == "sqnr":
            self._loss_fn = neg_sqnr
        else:
            raise ValueError(f"Invalid loss function: {self.loss_fn}")

    def get_loss_fn(self) -> Callable:
        """Returns loss function"""
        return self._loss_fn


class SequentialMseBase(ABC):
    """
    Sequentially minimizing activation MSE loss in layer-wise way to decide optimal param quantization encodings.
    """

    @classmethod
    def apply_seq_mse(
        cls,
        model: torch.nn.Module,
        sim,
        data_loader: DataLoader,
        params: SeqMseParams,
        modules_to_exclude: Optional[List[torch.nn.Module]] = None,
        checkpoints_config: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Sequentially minimizing activation MSE loss in layer-wise way to decide optimal param quantization encodings.

            1 Disable all input/output quantizers, param quantizers of non-supported modules
            2 Find and feeze optimal parameter encodings candidate for remaining supported modules
            3 Re-enable disabled quantizers from step 1

        Example userflow:
        model = Model().eval()
        sim = QuantizationSimModel(...)
        apply_seq_mse(...)
        sim.compute_encodings(...) [compute encodings for all activations and parameters of non-supported modules]
        sim.export(...)

        NOTE:
        1) module reference passed to modules_to_exclude should be from FP32 model.
        2) module from modules_to_exclude won't be quantized and skipped when applying sequential MSE.
        3) Except finding param encodings for supported modules, config JSON file will be respected and
        final state of sim will be unchanged.

        :param model: Original fp32 model
        :param sim: Corresponding QuantizationSimModel object
        :param data_loader: Data loader
        :param params: Sequential MSE parameters
        :param modules_to_exclude: List of supported type module(s) to exclude when applying Sequential MSE
        :param checkpoints_config: Config files to split fp32/quant model by checkpoints to speedup activations sampling
        :param cache_dir: Optional directory to cache/load optimized param encodings
        """
        # disable all input/output activation quantizers and
        # param quantizers of all the non-supported modules and from modules_to_exclude list.
        # then, re-enable disabled quantizers after running sequential mse.
        # this ensures that config JSON file will be respected and final state of sim will be unchanged.
        with (
            cls.temporarily_disable_quantizers(model, sim, modules_to_exclude),
            tempfile.TemporaryDirectory() as tempdir,
        ):
            # Initialize param encodings of modules of supported types.
            cls.compute_all_param_encodings(sim)

            dummy_input = change_tensor_device_placement(
                next(iter(data_loader)), get_device(model)
            )
            cached_dataset = CachedDataset(
                data_loader, params.num_batches, os.path.join(tempdir, "cached_dataset")
            )
            if checkpoints_config:
                cls.apply_seq_mse_using_opt_sampling(
                    checkpoints_config,
                    model,
                    sim,
                    modules_to_exclude,
                    cached_dataset,
                    params,
                    tempdir,
                    cache_dir,
                )
            else:
                fp32_modules = get_ordered_list_of_modules(
                    model,
                    dummy_input,
                    fwd_func=params.forward_fn,
                    ignore_duplicates=True,
                )
                fp32_modules = [
                    (name, module)
                    for name, module in fp32_modules
                    if isinstance(module, SUPPORTED_MODULES)
                ]

                def map_from_qt_to_fp_model(inp_module):
                    # helper function to map all modules_to_exclude from the QT object to the FP object
                    # if necessary
                    for name, module in sim.model.named_modules():
                        if module == inp_module:
                            fp_module = functools.reduce(
                                getattr, [model] + name.split(".")
                            )
                            return fp_module
                    return inp_module

                # Pre-compute the set of excluded FP32 modules
                excluded_fp_modules: Set[torch.nn.Module] = (
                    set(map(map_from_qt_to_fp_model, modules_to_exclude))
                    if modules_to_exclude
                    else set()
                )

                if cache_dir:
                    cached_module_names = cls.get_module_names_from_cache(cache_dir)
                    if excluded_fp_modules:
                        cached_module_names = [
                            name
                            for name in cached_module_names
                            if model.get_submodule(name) not in excluded_fp_modules
                        ]
                    cls.load_cached_optimized_params(
                        sim.model, cached_module_names, cache_dir
                    )
                    fp32_modules = [
                        (name, module)
                        for name, module in fp32_modules
                        if name not in cached_module_names
                    ]

                if excluded_fp_modules:
                    fp32_modules = [
                        (name, module)
                        for name, module in fp32_modules
                        if module not in excluded_fp_modules
                    ]

                # Find and freeze optimal param encodings candidate
                cls.run_seq_mse(
                    fp32_modules,
                    model,
                    sim.model,
                    params,
                    params.forward_fn,
                    cached_dataset,
                    cached_quant_dataset=None,
                    cache_dir=cache_dir,
                )

    @classmethod
    def apply_seq_mse_using_opt_sampling(
        cls,
        checkpoints_config: str,
        model: torch.nn.Module,
        sim,
        modules_to_exclude: Optional[List[torch.nn.Module]],
        cached_dataset: CachedDataset,
        params: SeqMseParams,
        tempdir: str,
        cache_dir: Optional[str],
    ):
        """
        Apply sequential MSE using optimized sampling of intermediate data. When checkpoints_config file is provided,
        intermediate activations from breakpoint are treated as model inputs for next blocks.

        NOTE: Assumption is that the outputs from the current block are fed directly to following block
        and there are no functional operations in-between.

        :param checkpoints_config: Config files to split fp32/quant model by checkpoints to speedup activations sampling
        :param model: Original fp32 model
        :param sim: Corresponding QuantizationSimModel object
        :param modules_to_exclude: List of supported type module(s) to exclude when applying Sequential MSE
        :param cached_dataset: Cached dataset
        :param params: Sequential MSE parameters
        :param tempdir: temporary working directory
        :param cache_dir: Optional directory to cache/load optimized param encodings
        """
        # pylint: disable=too-many-locals
        with open(checkpoints_config) as f:
            ckpts_file = json.load(f)

        # Validate required keys upfront with proper error handling (not assert, which can be
        # disabled with the -O interpreter flag).
        required_keys = {"grouped_modules", "include_static_inputs", "cache_on_cpu"}
        missing_keys = required_keys - ckpts_file.keys()
        if missing_keys:
            raise ValueError(
                f"Missing required keys in checkpoints config file '{checkpoints_config}': {missing_keys}"
            )

        grouped_modules = ckpts_file["grouped_modules"]
        breakpoint_module_name = ckpts_file["grouped_modules"][
            list(grouped_modules.keys())[0]
        ][0]
        include_static_inputs = ckpts_file["include_static_inputs"]
        cache_on_cpu = ckpts_file["cache_on_cpu"]
        cached_fp_dataset, cached_quant_dataset = get_block_inputs(
            model,
            sim,
            breakpoint_module_name,
            cached_dataset,
            cache_on_cpu,
            params.forward_fn,
            params.num_batches,
            tempdir,
        )
        device = get_device(model)
        model.cpu()
        sim.model.cpu()

        # Forward function for the ModuleList object
        def fwd_fn_modulelist(modulelists, x):
            for mod in modulelists:
                x = mod(*x) if isinstance(x, (tuple, list)) else mod(x)
            return x

        # helper function to map all modules_to_exclude from the QT object to the FP object
        # if necessary
        def map_from_qt_to_fp_model(inp_module):
            """Map a module from the quantized model to the corresponding FP32 module."""
            for name, module in sim.model.named_modules():
                if module == inp_module:
                    return functools.reduce(getattr, [model] + name.split("."))
            return inp_module

        excluded_fp_modules: Set[torch.nn.Module] = (
            set(map(map_from_qt_to_fp_model, modules_to_exclude))
            if modules_to_exclude
            else set()
        )

        sub_fp_models, sub_sim_models = create_modulelist_for_group_modules(
            model, sim, grouped_modules
        )
        for i, (fp_block, quant_sim_block, static_input) in enumerate(
            zip(sub_fp_models, sub_sim_models, include_static_inputs)
        ):
            args, kwargs = cached_fp_dataset[0]
            # Use explicit error raises instead of assert, which can be silently disabled
            # with the Python -O (optimize) flag.
            if kwargs:
                raise ValueError(
                    f"Keyword arguments are not supported for block inputs, got: {list(kwargs.keys())}"
                )
            if len(args) != 1:
                raise ValueError(
                    f"Expected exactly 1 positional argument for block inputs, got {len(args)}"
                )

            fp32_modules = get_ordered_list_of_modules(
                fp_block, args[0], fwd_func=fwd_fn_modulelist, ignore_duplicates=True
            )
            fp32_modules = [
                (name, module)
                for name, module in fp32_modules
                if isinstance(module, SUPPORTED_MODULES)
            ]

            # Use a block-scoped cache directory to avoid mutating the outer cache_dir variable,
            # which would corrupt subsequent iterations if an exception occurs mid-loop.
            block_cache_dir = os.path.join(cache_dir, str(i)) if cache_dir else None

            if block_cache_dir:
                cached_module_names = cls.get_module_names_from_cache(block_cache_dir)
                if excluded_fp_modules:
                    cached_module_names = [
                        name
                        for name in cached_module_names
                        if fp_block.get_submodule(name) not in excluded_fp_modules
                    ]
                cls.load_cached_optimized_params(
                    quant_sim_block, cached_module_names, block_cache_dir
                )
                fp32_modules = [
                    (name, module)
                    for name, module in fp32_modules
                    if name not in cached_module_names
                ]

            if excluded_fp_modules:
                fp32_modules = [
                    (name, module)
                    for name, module in fp32_modules
                    if module not in excluded_fp_modules
                ]

            cls.run_seq_mse(
                fp32_modules,
                fp_block,
                quant_sim_block,
                params,
                fwd_fn_modulelist,
                cached_fp_dataset,
                cached_quant_dataset=cached_quant_dataset,
                cache_dir=block_cache_dir,
            )

            # Get the outputs from the current block and assign to be the inputs for next block
            # except for the last block
            if i < len(sub_fp_models) - 1:
                get_block_outputs(
                    fp_block,
                    quant_sim_block,
                    static_input,
                    cached_fp_dataset,
                    cached_quant_dataset,
                    cache_on_cpu,
                    fwd_fn_modulelist,
                    device,
                    tempdir,
                )
        model.to(device)
        sim.model.to(device)

    @classmethod
    def run_seq_mse(
        cls,
        fp32_modules: List[Tuple[str, torch.nn.Module]],
        model: torch.nn.Module,
        quant_model: torch.nn.Module,
        params: SeqMseParams,
        forward_fn: Callable,
        cached_fp_dataset: CachedDataset,
        cached_quant_dataset: Optional[CachedDataset] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Run Sequential MSE

        :param fp32_modules: List of FP32 candidate modules in order of occurence
        :param model: FP32 model
        :param quant_model: QuantizationSimModel object
        :param params: Sequential MSE parameters
        :param forward_fn: Optional adapter function that performs forward pass given a model and inputs
        yielded from the data loader. The function expects model as first argument and inputs to model as second argument.
        :param cached_fp_dataset: Cached dataset object
        :param cached_quant_dataset: Cached dataset object
        :param cache_dir: Optional directory to save optimized param encodings
        """
        name_to_quant_module = {
            name: quant_module
            for name, quant_module in quant_model.named_modules()
            if isinstance(quant_module, BaseQuantizationMixin)
        }

        if not cached_quant_dataset:
            cached_quant_dataset = cached_fp_dataset

        def _is_optimizable(quant_module):
            return bool(
                quant_module is not None
                and quant_module.param_quantizers["weight"]
                and quant_module.param_quantizers["weight"].bitwidth
                <= SUPPORTED_PARAM_BW
            )

        # Total is the number of modules that will actually be optimized (the same
        # filter applied inside the loop), so the bar reflects real per-layer progress
        # rather than every module in the ordered list.
        num_optimizable = sum(
            _is_optimizable(name_to_quant_module.get(name)) for name, _ in fp32_modules
        )
        pbar = progress_bar(total=num_optimizable, desc="Sequential MSE", unit="module")

        for module_qualified_name, fp32_module in fp32_modules:
            quant_module = name_to_quant_module.get(module_qualified_name)

            if quant_module is None:
                _logger.warning(
                    "Module %s not found in quant model, skipping",
                    module_qualified_name,
                )
                continue

            if not quant_module.param_quantizers["weight"]:
                continue

            if quant_module.param_quantizers["weight"].bitwidth > SUPPORTED_PARAM_BW:
                continue

            _logger.info(
                "Finding and freezing optimal param encodings candidate of module: %s",
                module_qualified_name,
            )
            if params.inp_symmetry == "asym":
                fp32_inp_acts = cls.get_module_inp_acts(
                    fp32_module, model, forward_fn, cached_fp_dataset
                )
                quant_inp_acts = cls.get_module_inp_acts(
                    quant_module, quant_model, forward_fn, cached_quant_dataset
                )
                cls.optimize_module(quant_module, fp32_inp_acts, quant_inp_acts, params)
            elif params.inp_symmetry == "symfp":
                fp32_inp_acts = cls.get_module_inp_acts(
                    fp32_module, model, forward_fn, cached_fp_dataset
                )
                cls.optimize_module(quant_module, fp32_inp_acts, fp32_inp_acts, params)
            elif params.inp_symmetry == "symqt":
                quant_inp_acts = cls.get_module_inp_acts(
                    quant_module, quant_model, forward_fn, cached_quant_dataset
                )
                cls.optimize_module(
                    quant_module, quant_inp_acts, quant_inp_acts, params
                )
            else:
                raise ValueError(f"Invalid inp_symmetry: {params.inp_symmetry}")

            if cache_dir:
                cls.save_to_cache(module_qualified_name, quant_module, cache_dir)

            pbar.update(1)

        pbar.close()

    @classmethod
    def load_cached_optimized_params(
        cls, quant_model: torch.nn.Module, module_names: List[str], cache_dir: str
    ):
        """
        Load optimized quantizer min/max from cache

        :param quant_model: Quantized wrapper model
        :type quant_model: torch.nn.Module
        :param module_names: name of modules to load
        :type module_names: List[str]
        :param cache_dir: Directory containing cached param encodings
        :type cache_dir: str
        """
        # Load optimized min-max value and re-compute encoding
        for name in module_names:
            try:
                qmodule = quant_model.get_submodule(name)
            except AttributeError:
                _logger.warning(
                    "Module %s not found in model, skipping cached data", name
                )
                continue

            cached_state_dict_file = f"{cache_dir}/{name}.safetensors"
            if not os.path.exists(cached_state_dict_file):
                _logger.debug("State dict cache file not found for %s", name)
                continue

            state_dict = load_file(cached_state_dict_file)
            quantizer = qmodule.param_quantizers["weight"]
            quantizer.load_state_dict(state_dict)
            _logger.info(
                "Loaded weight param quantizer state dict of %s from %s",
                name,
                cached_state_dict_file,
            )
            cls._freeze_quantizer_encoding(quantizer)

    @classmethod
    def get_module_names_from_cache(cls, cache_dir: str) -> List[str]:
        """
        Get names of running modules from cache

        :param cache_dir: Directory containing the cache file
        :type cache_dir: str
        :return: list of running module names
        :rtype: List[str]
        """
        cache_file = f"{cache_dir}/running_module_names.txt"
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                content = f.read().strip()

            if not content:
                return []
            return content.split("\n")
        return []

    @classmethod
    def save_to_cache(cls, module_qualified_name: str, quant_module, cache_dir: str):
        """
        Save quantizer min/max to cache to reuse

        :param module_qualified_name: submodule name
        :type module_qualified_name: str
        :param quant_module: quantize wrapper module
        :type quant_module: QuantizationMixin
        :param cache_dir: Directory to save cache files
        :type cache_dir: str
        """
        if not cache_dir:
            raise ValueError("cache_dir cannot be None or empty")

        os.makedirs(cache_dir, exist_ok=True)

        # Save running module name
        cache_file = os.path.join(cache_dir, "running_module_names.txt")
        with open(cache_file, "a") as f:
            f.write(module_qualified_name)
            f.write("\n")

        # Save optimized min-max value
        saved_state_dict_file = os.path.join(
            cache_dir, f"{module_qualified_name}.safetensors"
        )
        state_dict = quant_module.param_quantizers["weight"].state_dict()
        # TODO (vinhpham): extra_state is dict not tensor must be removed before save state with safetensor
        if "extra_state" in state_dict:
            state_dict.pop("extra_state")
        if "_extra_state" in state_dict:
            state_dict.pop("_extra_state")
        _logger.info(
            "Save %s weight param quantizer state dict to %s",
            module_qualified_name,
            saved_state_dict_file,
        )
        save_file(state_dict, saved_state_dict_file)

    @staticmethod
    def get_module_inp_acts(
        module: torch.nn.Module,
        model: torch.nn.Module,
        forward_fn: Callable,
        cached_dataset: CachedDataset,
    ) -> torch.Tensor:
        """
        For given module, get inputs to the module.

        :param module: FP32/quant module
        :param model: FP32/quant model
        :param forward_fn: Optional adapter function that performs forward pass given a model and inputs
        yielded from the data loader. The function expects model as first argument and inputs to model as second argument.
        :param cached_dataset: Cached dataset
        :return: Concatenated inputs
        """
        inp_acts = []

        def hook_fn(_, inp, __):
            if isinstance(inp, tuple):
                inp_acts.append(inp[0])
            else:
                inp_acts.append(inp)
            raise StopForwardException

        handle = module.register_forward_hook(hook_fn)

        iterator = iter(cached_dataset)
        for inp in iterator:
            args, kwargs = change_tensor_device_placement(inp, get_device(model))
            try:
                with in_eval_mode(model), torch.no_grad():
                    forward_fn(model, *args, **kwargs)
            except StopForwardException:
                pass
        handle.remove()

        if not inp_acts:
            raise RuntimeError(
                f"No input activations were captured for module {module}. "
                "Ensure the dataset is non-empty and the module is reachable during the forward pass."
            )

        inp_acts = torch.stack(inp_acts)
        return inp_acts

    @staticmethod
    def _get_quantizers_to_be_disabled(
        model: torch.nn.Module,
        sim,
        modules_to_exclude: Optional[List[torch.nn.Module]],
    ):
        """
        For given quantsim model, get all quantizers to be disabled before applying sequential MSE.
        """
        # pylint: disable=protected-access
        name_to_fp32_module_dict = {
            name: fp32_module for name, fp32_module in model.named_modules()
        }

        quantizers_to_be_disabled = []
        for name, quant_wrapper in sim.quant_wrappers():
            for quantizer in quant_wrapper.input_quantizers:
                if quantizer.enabled:
                    quantizers_to_be_disabled.append(quantizer)

            for quantizer in quant_wrapper.output_quantizers:
                if quantizer.enabled:
                    quantizers_to_be_disabled.append(quantizer)

            for quantizer in quant_wrapper.param_quantizers.values():
                if (
                    not isinstance(quant_wrapper._module_to_wrap, SUPPORTED_MODULES)
                    and quantizer.enabled
                ):
                    quantizers_to_be_disabled.append(quantizer)

            # disable param quantizers from exclusion list
            if modules_to_exclude:
                with contextlib.suppress(KeyError):
                    fp32_module = name_to_fp32_module_dict[name]
                    if fp32_module in modules_to_exclude:
                        for quantizer in quant_wrapper.param_quantizers.values():
                            if quantizer.enabled:
                                quantizers_to_be_disabled.append(quantizer)

        return quantizers_to_be_disabled

    @staticmethod
    def get_candidates(
        num_candidates: int,
        per_channel_max: torch.Tensor,
        per_channel_min: Optional[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform grid search.

        :param num_candidates: Number of candidates
        :param per_channel_max: Per channel max values
        :param per_channel_min: Per channel min values
        :return: candidates
        """
        candidates = []
        if per_channel_min is not None:
            for cand in range(num_candidates):
                cand_max = torch.tensor(per_channel_max / num_candidates * (cand + 1))
                cand_min = torch.tensor(per_channel_min / num_candidates * (cand + 1))
                candidates.append((cand_max, cand_min))
        else:
            for cand in range(num_candidates):
                cand_max = torch.tensor(per_channel_max / num_candidates * (cand + 1))
                cand_min = -cand_max
                candidates.append((cand_max, cand_min))
        return candidates

    @staticmethod
    def compute_recon_loss(xqwq: torch.Tensor, xw: torch.Tensor, params: SeqMseParams):
        """
        Compute reconstruction loss and return the sum by reducing over all the dimensions except last channel dimension.

        :param xqwq: X^Q^ quantized-dequantized values
        :param xw: XW FP32 values
        :param params: Sequential MSE parameters
        :return: loss
        """
        loss_fn = params.get_loss_fn()
        channel_dim = xqwq.shape[-1]
        xqwq = xqwq.reshape(-1, channel_dim)
        xw = xw.reshape(-1, channel_dim)
        loss = loss_fn(xqwq, xw, reduction="none").sum(0)
        assert loss.size() == torch.Size([channel_dim])
        return loss

    @classmethod
    def get_per_channel_min_and_max(
        cls, quant_module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get per channel min/max values across output channel.

        :param quant_module: Quant module to be optimized
        :return:
        """
        # pylint: disable=protected-access
        module = cls._get_original_module(quant_module)

        if isinstance(module, torch.nn.Conv2d):
            channel_dim = module.out_channels
            weight = module.weight.reshape(channel_dim, -1)
        elif isinstance(module, torch.nn.Linear):
            weight = module.weight
        else:
            raise ValueError("Unsupported module: ", module)

        if cls._is_symmetric_quantizer(quant_module.param_quantizers["weight"]):
            per_channel_max = torch.max(weight.abs(), dim=1)[0].detach()
            per_channel_min = None
        else:
            per_channel_max = torch.max(weight, dim=1)[0].detach()
            per_channel_min = torch.min(weight, dim=1)[0].detach()

        return per_channel_min, per_channel_max

    @classmethod
    def compute_outputs(
        cls,
        quant_module,
        x: torch.Tensor,
        xq: torch.Tensor,
        w: torch.Tensor,
        wq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute X^W^ and XW output activations.

        :param quant_module: Wrapper module to be optimized
        :param x: Inputs from FP32 model
        :param xq: Inputs from QuantSim model
        :param w: FP32 weights
        :param wq: Quantized-dequantized weights
        :return: xqwq, xw
        """
        # pylint: disable=protected-access
        module = cls._get_original_module(quant_module)

        if isinstance(module, torch.nn.Linear):
            xqwq = functional.linear(xq, wq)
            xw = functional.linear(x, w)
        elif isinstance(module, torch.nn.Conv2d):
            xqwq = functional.conv2d(
                xq,
                wq,
                stride=module.stride,
                dilation=module.dilation,
                padding=module.padding,
                groups=module.groups,
            )
            xw = functional.conv2d(
                x,
                w,
                stride=module.stride,
                dilation=module.dilation,
                padding=module.padding,
                groups=module.groups,
            )

            # [N, C, H, W] --> [N, H, W, C], so that loss can be computed across channel dimension.
            xqwq = xqwq.permute(0, 2, 3, 1)
            xw = xw.permute(0, 2, 3, 1)
        else:
            raise ValueError("Unsupported module: ", module)
        return xqwq, xw

    @classmethod
    @abstractmethod
    def temporarily_disable_quantizers(
        cls,
        model: torch.nn.Module,
        sim,
        modules_to_exclude: Optional[List[torch.nn.Module]],
    ):
        """
        For given quantsim model, disable quantizers needed to be disabled before applying sequential MSE.

        :param model: Original fp32 model
        :param sim: QuantizationSimModel object
        :param modules_to_exclude: List of supported modules to exclude when applying Sequential MSE
        :return: List of quantizers to be disabled.
        """

    @classmethod
    @abstractmethod
    def compute_all_param_encodings(cls, sim):
        """
        Compute encodings for all parameters, needed for initializing Sequential MSE

        :param sim: Quant sim
        """

    @classmethod
    @abstractmethod
    def optimize_module(
        cls, quant_module, x: torch.Tensor, xq: torch.Tensor, params: SeqMseParams
    ):
        """
        Find and freeze optimal parameter encodings candidate for given module.

        :param quant_module: Quant module to be optimized
        :param x: Inputs to module from FP32 model
        :param xq: Inputs to module from QuantSim model
        :param params: Sequenial MSE parameters
        """

    @classmethod
    @abstractmethod
    def compute_param_encodings(
        cls, quantizer, x_min: torch.Tensor, x_max: torch.Tensor
    ):
        """
        Compute encodings for parameter quantizer using given x_min and x_max values.

        :param quantizer: Tensor quantizer
        :param x_min: min values
        :param x_max: max values
        """

    @classmethod
    @abstractmethod
    def _is_symmetric_quantizer(cls, quantizer): ...

    @classmethod
    @abstractmethod
    def _freeze_quantizer_encoding(cls, quantizer): ...

    @classmethod
    @abstractmethod
    def _get_quantized_weight(cls, quant_module): ...

    @classmethod
    @abstractmethod
    def _get_original_module(cls, quant_module): ...


def neg_sqnr(pred: torch.Tensor, target: torch.Tensor, eps=1e-10, reduction="none"):
    """
    Loss function to minimize negative SQNR which is equivalent to maximizing SQNR.

    :param pred: X^Q^ quantized-dequantized values
    :param target: XW FP32 values
    :param eps: epsilon
    :param reduction: unused arg added only to have the same signature as that of functional losses of pytorch library
    :return: Negative SQNR
    """
    # pylint: disable=unused-argument
    quant_error = target - pred
    exp_noise = torch.mean(quant_error**2, 0, keepdim=True) + eps
    exp_signal = torch.mean(target**2, 0, keepdim=True)
    sqnr = exp_signal / exp_noise
    sqnr_db = 10 * torch.log10(sqnr)
    return -sqnr_db
