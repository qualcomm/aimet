# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

"""AdaScale implementation"""

from copy import deepcopy
from dataclasses import dataclass
from types import NoneType
from typing import Callable, List, Any, Tuple, Type, Sequence, Optional, Dict
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file

from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2DecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3Model, Phi3DecoderLayer
from transformers.models.mistral.modeling_mistral import (
    MistralModel,
    MistralDecoderLayer,
)

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3DecoderLayer
except ImportError:
    Qwen3Model = Qwen3DecoderLayer = None

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLTextModel,
        Qwen2_5_VLDecoderLayer,
    )
except ImportError:
    Qwen2_5_VLTextModel = Qwen2_5_VLDecoderLayer = None

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextModel,
        Qwen3VLTextDecoderLayer,
    )
except ImportError:
    Qwen3VLTextModel = Qwen3VLTextDecoderLayer = None

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextModel,
        Qwen3_5DecoderLayer,
    )
except ImportError:
    Qwen3_5TextModel = Qwen3_5DecoderLayer = None

try:
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextModel,
        Gemma4TextDecoderLayer,
    )
except ImportError:
    Gemma4TextModel = Gemma4TextDecoderLayer = None


from aimet_torch.common.utils import AimetLogger
from aimet_torch.common.early_stopping import (
    _create_early_stopping,
)
from aimet_torch.common.progress import progress_bar
from aimet_torch import QuantizationSimModel
from aimet_torch.nn import QuantizedLinear, compute_param_encodings, QuantizedConv2d
from aimet_torch.utils import (
    default_forward_fn,
    remove_all_quantizers,
    remove_activation_quantizers,
)
from aimet_torch.quantization.affine import QuantizeDequantize
from aimet_torch.adascale.adascale_quantizer import (
    AdaScaleQuantizeDequantize,
    AdaScaleLinearQuantizeDequantize,
    AdaScaleConv2dQuantizeDequantize,
)
from aimet_torch.blockwise_sampler import (
    BlockwiseSampler,
    change_tensor_and_cache_device_placement,
    CachedIterable,
)
from aimet_torch.utils import get_device


@dataclass
class AdaScaleModelConfig:
    block_type: Type = None  # block types to use in a given model
    beta_gamma_lr: float = 1e-3  # lr for beta and gamma
    scales_lr: float = 5e-4  # lr for s2, s3, [s4]
    enable_caching_after_block: int = 0
    # for models with ops in between blocks (ex: Qwen3VL), intermediate activations cannot be cached accurately.
    # This param can be used to disable caching for initial blocks until the caching strategy can be used.


# mapping of model type and the corresponding adascale config
adascale_model_config_dict = {
    LlamaModel: AdaScaleModelConfig(
        block_type=LlamaDecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    Qwen2Model: AdaScaleModelConfig(
        block_type=Qwen2DecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    MistralModel: AdaScaleModelConfig(
        block_type=MistralDecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    Phi3Model: AdaScaleModelConfig(
        block_type=Phi3DecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
}

if Qwen2_5_VLTextModel is not None and Qwen2_5_VLDecoderLayer is not None:
    adascale_model_config_dict.update(
        {
            Qwen2_5_VLTextModel: AdaScaleModelConfig(
                block_type=Qwen2_5_VLDecoderLayer,
                beta_gamma_lr=1e-3,
                scales_lr=5e-4,
            )
        }
    )

if Qwen3Model is not None and Qwen3DecoderLayer is not None:
    adascale_model_config_dict.update(
        {
            Qwen3Model: AdaScaleModelConfig(
                block_type=Qwen3DecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
            )
        }
    )
if Qwen3VLTextModel is not None and Qwen3VLTextDecoderLayer is not None:
    adascale_model_config_dict.update(
        {
            Qwen3VLTextModel: AdaScaleModelConfig(
                block_type=Qwen3VLTextDecoderLayer,
                beta_gamma_lr=1e-3,
                scales_lr=5e-4,
                enable_caching_after_block=3,
            )
        }
    )

if Qwen3_5TextModel is not None and Qwen3_5DecoderLayer is not None:
    adascale_model_config_dict.update(
        {
            Qwen3_5TextModel: AdaScaleModelConfig(
                block_type=Qwen3_5DecoderLayer,
                beta_gamma_lr=1e-3,
                scales_lr=5e-4,
            )
        }
    )

if Gemma4TextModel is not None and Gemma4TextDecoderLayer is not None:
    adascale_model_config_dict.update(
        {
            Gemma4TextModel: AdaScaleModelConfig(
                block_type=Gemma4TextDecoderLayer,
                beta_gamma_lr=1e-3,
                scales_lr=5e-4,
                enable_caching_after_block=35,
            )
        }
    )


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)


_QT_SAMPLING_PROB = 0.5

_BlockOutput = torch.Tensor | Tuple[torch.Tensor, ...]

# Loss function contract: takes the FP and quantized block outputs and the index of the current
# calibration input (in data_loader order), and returns a scalar loss tensor.
_LossFn = Callable[[_BlockOutput, _BlockOutput, int], torch.Tensor]

# Temporary flag to enable early stopping of the per-block optimization loop.
# None/False disables it; True enables it with default parameters. To configure
# the parameters, set it to an
# aimet_torch.common.early_stopping._EarlyStoppingConfig instead.
# TODO: promote to a real config / public arg once validated.
_EARLY_STOPPING = None


def _mse_loss_fn(
    fp_out: _BlockOutput,
    qt_out: _BlockOutput,
    data_idx: Optional[int] = None,  # pylint: disable=unused-argument
    p: float = 2.0,
) -> torch.Tensor:
    """Returns the block loss between fp_out and qt_out.

    Uses FlexRound's lp_loss: the per-element error is summed over dim 1 (the
    sequence dim for the transformer decoder blocks this targets) and averaged
    over the rest, so it is not divided by sequence length S. ``data_idx`` is
    accepted to honor the loss function contract and is unused by the default loss.
    """
    if isinstance(fp_out, (tuple, list)):
        fp_out = torch.cat(fp_out)
    if isinstance(qt_out, (tuple, list)):
        qt_out = torch.cat(qt_out)

    return (fp_out - qt_out).abs().pow(p).sum(1).mean()


supported_modules: List = [QuantizedLinear, QuantizedConv2d]


class PerBlockCheckpointManager:
    """
    Manages per-block checkpoints for AdaScale optimization.
    Each block is saved separately after completion.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.checkpoint_dir / "progress.json"

    def save_completed_block(self, block: torch.nn.Module, block_idx: int):
        """
        Save a completed block's AdaScale params.
        Called after block optimization finishes.
        """
        flat_params = AdaScale.extract_adascale_params(block)

        block_checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_block_{block_idx}.safetensors"
        )
        save_file(flat_params, block_checkpoint_path)

        _logger.info(
            "Block %d checkpoint saved to: %s", block_idx, block_checkpoint_path
        )
        _logger.info(
            "Size: %.2f MB", block_checkpoint_path.stat().st_size / 1024 / 1024
        )

        # Update progress tracker
        self._update_progress(block_idx, completed=True)

    def load_completed_block(self, block: torch.nn.Module, block_idx: int):
        """
        Load and restore a completed block's AdaScale params.
        Called when resuming optimization to restore already-optimized blocks.
        """
        block_checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_block_{block_idx}.safetensors"
        )

        if not block_checkpoint_path.exists():
            return None

        flat_params = load_file(block_checkpoint_path, device="cpu")

        AdaScale.restore_adascale_params(block, flat_params)

        _logger.info("Loaded checkpoint for block %d successfully", block_idx)

    def get_progress(self) -> Dict:
        """Get current progress information"""
        if not self.progress_file.exists():
            return {
                "total_blocks": 0,
                "completed_blocks": [],
                "current_block": 0,
            }

        with open(self.progress_file, "r") as f:
            return json.load(f)

    def _update_progress(self, block_idx: int, completed: bool, **metadata):
        """Update progress tracker file"""
        progress = self.get_progress()

        if completed:
            if block_idx not in progress["completed_blocks"]:
                progress["completed_blocks"].append(block_idx)
                progress["completed_blocks"].sort()
            progress["current_block"] = block_idx + 1
        else:
            progress["current_block"] = block_idx

        # Update any additional metadata
        progress.update(metadata)

        temp_file = self.progress_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(progress, f, indent=2)
        temp_file.replace(self.progress_file)

    def is_block_completed(self, block_idx: int) -> bool:
        """Check if a block has been completed"""
        block_checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_block_{block_idx}.safetensors"
        )
        return block_checkpoint_path.exists()

    def get_resume_point(self) -> tuple[int, int]:
        """
        Get the point to resume from.
        Returns: (all_done, block_idx)
        """
        progress = self.get_progress()
        current_block = progress.get("current_block", 0)
        completed_blocks = progress["completed_blocks"]
        total_blocks = progress["total_blocks"]

        # If all blocks are completed
        if total_blocks == len(completed_blocks) and len(completed_blocks) > 0:
            return True, current_block

        # Otherwise, start from next uncompleted block
        return False, current_block


class AdaScale:
    """
    AdaScale is PTQ technique which performs Knowledge Distillation on blocks of modules by using the FP32 output as its
    reference output. Adascale is based on FlexRound: https://arxiv.org/abs/2306.00317 but integrates LWC from Omniquant.

    The optimization is performed on a block-by-block basis by comparing the quantized output of the block with its FP32
    equivalent and by training the parameters (gamma, beta, s2, s3) which are temporarily introduced in every supported
    module.

    A block is defined as a non-leaf module which takes in one activation input tensor and outputs one activation tensor
    Currently only Linear layers are supported, and all the linears in a block are optimized at the same time.

    While performing the optimization, the activation quantizers are disabled, linear modules' weight quantizers are
    changed to specialized QDQ (with learnable parameters introduced) and rest of the param's are left quantized with
    default QuantizeDequantize.


    """

    @classmethod
    def apply_adascale(
        cls,
        qsim: QuantizationSimModel,
        data_loader: DataLoader,
        forward_fn: Callable[[torch.nn.Module, Any], Any] = None,
        num_iterations: int = 1500,
        checkpoint_dir: str = None,  # For checkpointing
        *,
        loss_fn: Optional[_LossFn] = None,
    ):
        """
        :param qsim: Quantization Sim model
        :param data_loader: DataLoader object to load the input data
        :param forward_fn: forward function to run the forward pass of the model
        :param num_iterations: Number of iterations to optimize for during AdaScale BKD
        :param checkpoint_dir: Directory to save/resume per-block checkpoints from
        :param loss_fn: Loss function with signature ``loss_fn(fp_out, quant_out, data_idx)`` returning a scalar
            loss tensor. ``data_idx`` is the index of the current calibration input in ``data_loader`` order. Defaults
            to MSE loss if None.

        Note that the forward_fn should take exactly two arguments -
        1) the model
        2) The object returned from the dataloader irrespective of whether it's a tensor/tuple of tensors/dict/etc

        The forward_fn should prepare the "input sample" as needed and call the forward pass in the very end. The forward_fn
        should not be running any sort of eval, creating full dataloader inside the method, etc.

        Example usage:
            >>> model = DummyModel()
            >>> dummy_input = ...
            >>> data_set = DataSet(dummy_input)
            >>> data_loader = DataLoader(data_set, ...)
            >>> sim = QuantizationSimModel(model, dummy_input)
            >>> apply_adascale(sim, data_loader, forward_fn=forward_fn, num_iterations=1500)
            >>> sim.compute_encodings(...)
            >>> sim.export(...)

        .. note::
        1. apply_adascale modifies the weights in-place in the model
        2. compute encodings should not be called before the apply_adascale call
        3. Activation quantizers will remain uninitialized throughout the feature, and so compute encodings needs to be called by the user afterwards. This is so activation encodings will be computed with updated weights taken into account.

        Warning: This feature is currently considered experimental pending API changes
        """
        # pylint: disable=too-many-locals
        if not forward_fn:
            forward_fn = default_forward_fn

        assert num_iterations > 0, "num_iterations must be greater than 0"

        # Initialize checkpoint manager
        checkpoint_manager = None
        start_block_idx = 0

        if checkpoint_dir:
            checkpoint_manager = PerBlockCheckpointManager(checkpoint_dir)

            # Get resume point
            all_blocks_done, start_block_idx = checkpoint_manager.get_resume_point()

            if all_blocks_done:
                _logger.info(
                    "All %d blocks have been optimized for AdaScale.", start_block_idx
                )
                _logger.info("Loading checkpoints and skipping AdaScale.")
            else:
                if start_block_idx > 0:
                    _logger.info("Resuming from block %d", start_block_idx)
                else:
                    _logger.info("Starting fresh AdaScale optimization")

        compute_param_encodings(qsim.model)

        adascale_blocks = cls._get_blocks(qsim)

        # replace with adascale weight quantizer which introduces trainable params - beta, gamma, s2, s3
        device = get_device(qsim.model)
        dtype = next(qsim.model.parameters()).dtype

        # Update progress with total blocks
        if checkpoint_manager:
            metadata = {
                "total_blocks": len(adascale_blocks),
            }
            # pylint: disable=protected-access
            checkpoint_manager._update_progress(
                start_block_idx, completed=False, **metadata
            )

        def _restore_completed_block(blk: torch.nn.Module, idx: int):
            """Install AdaScale quantizers on a completed block and load its saved params."""
            temp_device = get_device(blk)
            temp_dtype = next(blk.parameters()).dtype
            cls._replace_with_adascale_weight_quantizers(blk)
            checkpoint_manager.load_completed_block(blk, idx)
            blk.to(device=temp_device, dtype=temp_dtype)

        # If every block is already checkpointed, restore all of them and skip sampling/optimization entirely.
        if checkpoint_manager and all_blocks_done:
            for idx, blk in enumerate(adascale_blocks):
                _restore_completed_block(blk, idx)
            cls.fold_adascale_quantizers(qsim.model)
            qsim.model.to(device=device, dtype=dtype)
            del checkpoint_manager
            _logger.info(
                "AdaScale restored from checkpoints; no blocks left to optimize."
            )
            return

        qsim.model.to(device=torch.device("cpu"), dtype=dtype)

        disable_caching = cls._model_specific_blocks_to_disable_caching(qsim)

        # Resume by yielding from the checkpointed prefix's end. get_resume_point guarantees a contiguous completed
        # prefix (blocks 0..start_block_idx-1). In the non-cached region the sampler simply skips sampling those
        # blocks; in the cached region it still chains through them internally (using the weights we restore below)
        # but does not re-yield them. Either way, the prefix must be restored before sampling so the resume block's
        # forward pass sees the optimized upstream weights.
        can_skip_sampling = checkpoint_manager is not None and start_block_idx > 0
        sampler_start = start_block_idx if can_skip_sampling else 0

        if can_skip_sampling:
            for idx in range(start_block_idx):
                _restore_completed_block(adascale_blocks[idx], idx)

        sampler = BlockwiseSampler(
            qsim,
            adascale_blocks,
            data_loader,
            forward_fn,
            keep_unused_blocks_on_cpu=True,
            cache_activations_on_disk=True,
            disable_caching_until_block=disable_caching,
        )

        qsim.model.requires_grad_(False)
        beta_gamma_lr, scales_lr = AdaScale._model_specific_lr(qsim)

        with remove_activation_quantizers(qsim.model):
            for offset, (block, fp_block_inputs, qt_block_inputs) in enumerate(
                sampler.sample(
                    device=device,
                    desc="AdaScale blocks processed",
                    start_block=sampler_start,
                )
            ):
                block_idx = sampler_start + offset
                # The eagerly-restored contiguous prefix is skipped by sampler_start; this still handles any
                # non-contiguous completed blocks that appear at or after the resume point.
                if checkpoint_manager and checkpoint_manager.is_block_completed(
                    block_idx
                ):
                    _logger.info(
                        "Block %d already completed, loading checkpoint...", block_idx
                    )
                    _restore_completed_block(block, block_idx)
                    continue

                fp_block_inputs = CachedIterable(fp_block_inputs)
                qt_block_inputs = _SamplingIterable(
                    CachedIterable(qt_block_inputs),
                    fp_block_inputs,
                    ratio=_QT_SAMPLING_PROB,
                    device=device,
                )

                # Optimize block
                cls.adascale_block(
                    block,
                    fp_block_inputs,
                    qt_inputs=qt_block_inputs,
                    num_iterations=num_iterations,
                    beta_gamma_lr=beta_gamma_lr,
                    scales_lr=scales_lr,
                    loss_fn=loss_fn,
                )

                # Save completed block checkpoint
                if checkpoint_manager:
                    checkpoint_manager.save_completed_block(
                        block=block,
                        block_idx=block_idx,
                    )

        del sampler, checkpoint_manager

        cls.fold_adascale_quantizers(qsim.model)

        qsim.model.to(device=device, dtype=dtype)

        _logger.info("AdaScale optimization completed!")

    @classmethod
    def adascale_block(
        cls,
        block: torch.nn.Module,
        fp_inputs: Sequence[Tuple[Tuple, Dict]],
        qt_inputs: Sequence[Tuple[Tuple, Dict]],
        *,
        num_iterations: int = 1500,
        beta_gamma_lr: float = 1e-3,
        scales_lr: float = 5e-4,
        loss_fn: Optional[_LossFn] = None,
    ):
        """
        Performs AdaScale algorithm to optimize weight quantization of input block to minimize the output MSE
        between floating point and quantized execution of the block.

        Args:
            block: Quantized torch.nn.Module object to optimize
            fp_inputs: Sequence of (args, kwargs) observed by the block during floating-point model execution
            qt_inputs: Sequence of (args, kwargs) observed by the block during quantized model execution over the same
                input set as fp_inputs.
            num_iterations: Number of iterations to optimize for during AdaScale BKD
            beta_gamma_lr: Learning rate for beta and gamma parameters
            scales_lr: Learning rate for scale parameters
            loss_fn: Loss function with signature ``loss_fn(fp_out, quant_out, data_idx)`` returning a scalar loss
                tensor, where ``data_idx`` is the index of the current input in ``fp_inputs``/``qt_inputs``.
                Defaults to MSE loss if None.

        """
        if loss_fn is None:
            loss_fn = _mse_loss_fn

        compute_param_encodings(block)
        device = get_device(block)
        dtype = next(block.parameters()).dtype
        block.requires_grad_(False)
        cls._replace_with_adascale_weight_quantizers(block)
        block.to(device=device, dtype=dtype)

        def run_forward(args, kwargs):
            args = change_tensor_and_cache_device_placement(args, device)
            kwargs = change_tensor_and_cache_device_placement(kwargs, device)
            return block(*args, **kwargs)

        if not qt_inputs:
            qt_inputs = fp_inputs

        # only set adascale params to train mode
        all_beta_gamma_parameters, all_scale_parameters = (
            cls._get_adascale_trainable_params(block)
        )
        trainable_params = [
            {"params": all_beta_gamma_parameters, "lr": beta_gamma_lr},
            {"params": all_scale_parameters, "lr": scales_lr},
        ]
        optimizer = torch.optim.Adam(trainable_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iterations, eta_min=0.0
        )
        cls._set_requires_grad(all_beta_gamma_parameters + all_scale_parameters, True)

        fp_out = []  # save fp batchwise block outputs to use across epochs
        with remove_all_quantizers(block):
            for args, kwargs in fp_inputs:
                fp_block_results = run_forward(args, kwargs)
                fp_block_results = change_tensor_and_cache_device_placement(
                    fp_block_results, "cpu"
                )
                fp_out.append(fp_block_results)
                del args, kwargs, fp_block_results

        early_stopping = _create_early_stopping(_EARLY_STOPPING)

        pbar = progress_bar(
            total=num_iterations,
            leave=False,
            position=1,
            desc="Iterations completed",
        )
        curr_iteration = 0
        with remove_activation_quantizers(block):
            while curr_iteration < num_iterations:
                if early_stopping is not None and early_stopping.early_stop:
                    break
                for batch_idx, (args, kwargs) in enumerate(qt_inputs):
                    pbar.update(1)
                    curr_iteration += 1
                    if curr_iteration > num_iterations:
                        pbar.close()
                        break
                    with torch.set_grad_enabled(True):
                        # Tolerate in-place cache updates (e.g. linear-attention
                        # recurrent state) on tensors saved for backward. Keeping off
                        # optimizer.step() from the scope.
                        with torch.autograd.graph.allow_mutation_on_saved_tensors():
                            quant_out = run_forward(args, kwargs)

                            del args, kwargs

                            batch_fp_out = change_tensor_and_cache_device_placement(
                                deepcopy(fp_out[batch_idx]), device
                            )

                            loss = loss_fn(batch_fp_out, quant_out, batch_idx)
                            loss.backward()

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        # Early stopping check
                        should_stop = early_stopping is not None and early_stopping(
                            loss.item()
                        )
                        del quant_out, batch_fp_out, loss
                        if should_stop:
                            pbar.close()
                            break

    @staticmethod
    def extract_adascale_params(block: torch.nn.Module) -> dict:
        """
        Extract only AdaScale parameters from a block.
        Returns a lightweight dict containing only beta, gamma, s2, s3, s4.
        """
        device = get_device(block)
        block.to(device="cpu")

        adascale_params = {}
        for name, module in block.named_modules():
            if isinstance(module, tuple(supported_modules)):
                weight_quantizer = module.param_quantizers["weight"]

                if isinstance(weight_quantizer, AdaScaleQuantizeDequantize):
                    # Extract AdaScale parameters
                    quantizer_state = weight_quantizer.state_dict()
                    if "extra_state" in quantizer_state:
                        quantizer_state.pop("extra_state")
                    if "_extra_state" in quantizer_state:
                        quantizer_state.pop("_extra_state")
                    adascale_params[name] = quantizer_state

        adascale_params = AdaScale.flatten_state_dict(adascale_params)

        block.to(device=device)

        return adascale_params

    @staticmethod
    def restore_adascale_params(block: torch.nn.Module, adascale_params: dict):
        """
        Restore AdaScale parameters to a block.
        Assumes block already has AdaScaleQuantizeDequantize quantizers installed.
        """
        device = get_device(block)
        block.to(device="cpu")

        adascale_params = AdaScale.unflatten_state_dict(adascale_params)

        for name, module in block.named_modules():
            if name in adascale_params:
                weight_quantizer = module.param_quantizers["weight"]

                if isinstance(weight_quantizer, AdaScaleQuantizeDequantize):
                    quantizer_state = adascale_params[name]
                    # Restore parameters
                    weight_quantizer.load_state_dict(quantizer_state)

        block.to(device=device)

    @staticmethod
    def flatten_state_dict(
        original_dict: dict[str, dict[str, torch.Tensor]],
        separator: str = "/",
    ) -> dict[str, torch.Tensor]:
        return {
            f"{outer_key}{separator}{inner_key}": value
            for outer_key, inner_dict in original_dict.items()
            for inner_key, value in inner_dict.items()
        }

    @staticmethod
    def unflatten_state_dict(
        flat_dict: dict[str, torch.Tensor],
        separator: str = "/",
    ) -> dict[str, dict[str, torch.Tensor]]:
        original_dict: dict[str, dict[str, torch.Tensor]] = {}

        for flat_key, value in flat_dict.items():
            outer_key, inner_key = flat_key.split(separator, maxsplit=1)

            if isinstance(value, torch.Tensor):
                value = value.detach().clone()

            original_dict.setdefault(outer_key, {})[inner_key] = value

        return original_dict

    @staticmethod
    def _screen_for_target_type(model: torch.nn.Module) -> Type:
        """
        Helper to get the model type to optimize
        This is needed because the target module might not be at the top level in which case we go deeper and fetch it
        """
        for module in model.modules():
            for target in adascale_model_config_dict:
                if isinstance(module, target):
                    return target
        # No targets found in provided model
        return NoneType

    @staticmethod
    def _get_blocks(qsim: QuantizationSimModel):
        """helper to get all the blocks in the model represented by adascale_model_config_dict"""

        target_type = AdaScale._screen_for_target_type(qsim.model)
        block_type = adascale_model_config_dict.get(
            target_type, AdaScaleModelConfig()
        ).block_type
        target_modules = []
        if block_type is not None:
            target_modules = [
                m for m in qsim.model.modules() if isinstance(m, block_type)
            ]
        return target_modules

    @staticmethod
    def _model_specific_blocks_to_disable_caching(qsim: QuantizationSimModel) -> int:
        """helper function to get the number of initial blocks for which caching should be disabled"""
        target_type = AdaScale._screen_for_target_type(qsim.model)
        num_blocks_to_disable_caching = adascale_model_config_dict.get(
            target_type, AdaScaleModelConfig()
        ).enable_caching_after_block
        return num_blocks_to_disable_caching

    @staticmethod
    def _model_specific_lr(qsim: QuantizationSimModel) -> tuple[float, float]:
        """
        Given the sim object, query the model type and return the custom lr to be used
        """
        target_type = AdaScale._screen_for_target_type(qsim.model)
        model_config = adascale_model_config_dict.get(
            target_type, AdaScaleModelConfig()
        )
        return model_config.beta_gamma_lr, model_config.scales_lr

    @classmethod
    def _replace_with_adascale_weight_quantizers(cls, block: torch.nn.Module):
        """Replace all the weight quantizers in supported modules with adascale quantizers"""
        for layer in block.modules():
            if isinstance(layer, tuple(supported_modules)):
                if not isinstance(layer.param_quantizers["weight"], QuantizeDequantize):
                    continue
                layer.param_quantizers["weight"] = cls._get_adascale_qdq_mapping()[
                    type(layer)
                ](layer.param_quantizers["weight"], layer.weight.shape)

    @classmethod
    @torch.no_grad()
    def fold_adascale_quantizers(cls, module: torch.nn.Module):
        for layer in module.modules():
            if isinstance(layer, tuple(supported_modules)) and isinstance(
                layer.param_quantizers["weight"], AdaScaleQuantizeDequantize
            ):
                layer.weight.copy_(
                    layer.param_quantizers["weight"].get_folded_weight(layer.weight)
                )
                layer.param_quantizers["weight"] = layer.param_quantizers[
                    "weight"
                ].get_qdq()
                layer.param_quantizers["weight"].allow_overwrite(False)
                layer.requires_grad_(False)

    @staticmethod
    def _get_adascale_trainable_params(
        non_leaf_module: torch.nn.Module,
    ) -> Tuple[List, List]:
        """Get all the adascale scale params present in the non-leaf module"""
        all_scale_parameters = []
        all_beta_gamma_parameters = []
        for module in non_leaf_module.modules():
            if isinstance(module, tuple(supported_modules)) and isinstance(
                module.param_quantizers["weight"], AdaScaleQuantizeDequantize
            ):
                beta_gamma_params, scale_parameters = module.param_quantizers[
                    "weight"
                ].get_adascale_trainable_parameters()
                all_beta_gamma_parameters.extend(beta_gamma_params)
                all_scale_parameters.extend(scale_parameters)
        return all_beta_gamma_parameters, all_scale_parameters

    @staticmethod
    def _set_requires_grad(adascale_params: list, val: bool):
        """Helper to update requires_grad to the input `val` for all the params in `adascale_params`"""
        for p in adascale_params:
            p.requires_grad = val

    @staticmethod
    def _get_adascale_qdq_mapping() -> dict:
        return {
            QuantizedLinear: AdaScaleLinearQuantizeDequantize,
            QuantizedConv2d: AdaScaleConv2dQuantizeDequantize,
        }


class _SamplingIterable:
    """
    Args
        iterable_1: First iterable to sample from
        iterable_2: Second iterable to sample from
        ratio: Ratio of values from tensors in iterable_1 to values from iterable_2 to use in the output
        device: Device to place the yielded values on
    """

    def __init__(self, iterable_1, iterable_2, ratio=0.5, device="cpu"):
        self.iterable_1 = iterable_1
        self.iterable_2 = iterable_2
        self.ratio = ratio
        self.device = device

    def __iter__(self):
        for (args_1, kwargs_1), (args_2, _) in zip(
            self.iterable_1, self.iterable_2, strict=True
        ):
            combined_args = tuple(
                self._combine_tensors(arg_1, arg_2)
                for arg_1, arg_2 in zip(
                    change_tensor_and_cache_device_placement(args_1, self.device),
                    change_tensor_and_cache_device_placement(args_2, self.device),
                    strict=True,
                )
            )

            yield (
                combined_args,
                change_tensor_and_cache_device_placement(kwargs_1, self.device),
            )

    def _combine_tensors(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor):
        return (
            torch.where(
                torch.rand_like(tensor_1, dtype=tensor_1.dtype, device=self.device)
                < self.ratio,
                tensor_1,
                tensor_2,
            )
            if isinstance(tensor_1, torch.Tensor)
            else tensor_1
        )


apply_adascale = AdaScale.apply_adascale
adascale_block = AdaScale.adascale_block
fold_adascale_quantizers = AdaScale.fold_adascale_quantizers
