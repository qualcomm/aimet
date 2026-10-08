# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""AdaScale implementation"""

import contextlib
from typing import Callable, Collection, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import tempfile
import gc
import warnings
import onnx_ir
import os

from aimet_onnx.common.utils import AimetLogger  # pylint: disable=import-error
from aimet_onnx.common.early_stopping import (  # pylint: disable=import-error
    _create_early_stopping,
)
from aimet_onnx.common.progress import progress_bar  # pylint: disable=import-error
from aimet_onnx.experimental.adascale.utils import (
    convert_to_torch,
    change_tensor_device_placement,
)
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.utils import (
    get_torch_device,
)
from aimet_onnx import ir_utils
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.experimental.llm_topology.block_boundaries import (
    get_decoder_block_boundaries,
    resolve_residual_tensor_name,
)
from aimet_onnx.experimental.llm_topology.topology_types import (
    BlockKind,
    LlmTopology,
)

from aimet_onnx.experimental.adascale.quantizer import (
    add_qlinear_layers,
    get_adascale_trainable_params,
    replace_with_adascale_quantizers,
)

from aimet_onnx.experimental.adascale.activation_sampler import ActivationSampler
from aimet_onnx.experimental.adascale.model_converter import (
    get_pt_block,
    copy_pt_weights_to_onnx,
    copy_pt_encodings_to_sim,
    required_extra_block_inputs,
    upcast_fp16_block_to_bf16,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)


_QT_SAMPLING_PROB = 1.0

# Loss function contract: takes the FP and quantized block outputs and the index of the current
# calibration input (in inputs order), and returns a scalar loss tensor.
_LossFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]

# Temporary flag to enable early stopping of the per-block optimization loop.
# None/False disables it; True enables it with default parameters. To configure
# the parameters, set it to an
# aimet_onnx.common.early_stopping._EarlyStoppingConfig instead.
# TODO: promote to a real config / public arg once validated.
_EARLY_STOPPING = None


def _mse_loss_fn(
    fp_out: torch.Tensor,
    qt_out: torch.Tensor,
    data_idx: Optional[int] = None,  # pylint: disable=unused-argument
    p: float = 2.0,
) -> torch.Tensor:
    """Returns the block loss between fp_out and qt_out.

    Uses FlexRound's lp_loss: the per-element error is summed over dim 1 (the
    sequence dim for the transformer decoder blocks this targets) and averaged
    over the rest, so it is not divided by sequence length S. ``data_idx`` is
    accepted to honor the loss function contract and is unused by the default loss.
    """
    return (fp_out - qt_out).abs().pow(p).sum(1).mean()


_DEBUG_NUM_PARTIAL_ITERATIONS = None
_DEBUG_NUM_PARTIAL_ITERATIONS_START = None
_DEBUG_NUM_PARTIAL_ITERATIONS_END = None


@dataclass
class AdaScaleModelConfig:
    model_type: str
    beta_gamma_lr: float = 1e-3  # lr for beta and gamma
    scales_lr: float = 5e-4  # lr for s2, s3, [s4]


# mapping of model type and the corresponding adascale config
adascale_model_config_dict = {
    "llama": AdaScaleModelConfig(
        model_type="llama", beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "qwen2": AdaScaleModelConfig(
        model_type="qwen2", beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "mistral": AdaScaleModelConfig(
        model_type="mistral", beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "qwen3": AdaScaleModelConfig(
        model_type="qwen3", beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "qwen3_vl": AdaScaleModelConfig(
        model_type="qwen3", beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "qwen3_5": AdaScaleModelConfig(
        model_type="qwen3_5", beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "qwen3_5_text": AdaScaleModelConfig(
        model_type="qwen3_5", beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "phi3": AdaScaleModelConfig(model_type="phi3", beta_gamma_lr=1e-3, scales_lr=5e-4),
    "qwen2_5_vl": AdaScaleModelConfig(
        model_type="qwen2_5_vl", beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "gemma3": AdaScaleModelConfig(
        model_type="gemma3", beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "gemma3_text": AdaScaleModelConfig(
        model_type="gemma3", beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "internvl_chat": AdaScaleModelConfig(
        model_type="qwen3", beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
}


def _block_boundaries_from_topology(topology: LlmTopology) -> List[Tuple[str, str]]:
    """Read the per-block residual-stream boundaries off ``topology``.

    :param topology: Topology of the model being optimized.
    :return: One ``(start_tensor, end_tensor)`` pair per decoder block, in topological order.
    :raises ValueError: If ``topology`` has no decoder blocks, or a block carries no
        residual-stream tensor names to bound it with.
    """
    if not topology.blocks:
        raise ValueError(
            "topology contains no decoder blocks, so there is nothing for AdaScale to optimize. "
            "Verify that analyze_llm_topology() was run on the model being optimized."
        )

    mamba = [
        i for i, block in enumerate(topology.blocks) if block.kind is BlockKind.MAMBA
    ]
    if mamba:
        raise ValueError(
            f"topology has Mamba blocks {mamba}, which AdaScale does not support yet."
        )

    boundaries = []
    for block_idx, block in enumerate(topology.blocks):
        if block.residual_input is None or block.residual_output is None:
            raise ValueError(
                f"topology block {block_idx} has no residual-stream boundary "
                f"(residual_input={block.residual_input}, residual_output={block.residual_output}). "
                "AdaScale needs both to slice the block out of the graph."
            )
        boundaries.append((block.residual_input, block.residual_output))
    return boundaries


def _validate_boundaries_in_graph(
    graph: onnx_ir.Graph, block_end_points: List[Tuple[str, str]]
):
    """Check every boundary tensor exists in ``graph`` before optimization starts.

    A topology built from a different (or since-modified) model names tensors the graph
    does not have. Caught here, that is one clear error; left uncaught, it surfaces much
    later as an opaque onnxruntime failure while sampling block activations.

    :param graph: Graph AdaScale is about to optimize.
    :param block_end_points: Boundaries as returned by :func:`_block_boundaries_from_topology`.
    :raises ValueError: If any boundary tensor is absent from ``graph``.
    """
    names_in_graph = onnx_ir.convenience.create_value_mapping(graph).keys()
    missing = [
        name
        for start, end in block_end_points
        for name in (start, end)
        if name not in names_in_graph
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} block boundary tensor(s) are not present in the model being "
            f"optimized: {missing[:10]}{' ...' if len(missing) > 10 else ''}. The topology was "
            "likely built from a different model than the one held by this sim."
        )


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

    # pylint: disable=unused-argument, unused-variable

    @classmethod
    def apply_adascale(
        cls,
        sim: QuantizationSimModel,
        inputs: Collection[Dict[str, np.ndarray]],
        adascale_model_config: AdaScaleModelConfig,
        num_iterations: int = 1500,
        *,
        topology: Optional[LlmTopology] = None,
        loss_fn: Optional[_LossFn] = None,
    ):
        """
        :param sim: Quantization Sim model
        :param inputs: (Collection[Dict[str, np.ndarray]]): The set of input samples to use during optimization.
        :param adascale_model_config: Adascale model config. There are pre-defined configs for
                                      Llama, Qwen2, Mistral, Qwen3, Phi3. For other models use AdaScaleModelConfig
        :param num_iterations: Number of iterations to optimize for during AdaScale
        :param topology: Decoder-stack topology of the model held by ``sim``, from
            :func:`~aimet_onnx.experimental.llm_topology.analyze_llm_topology`.
            AdaScale optimizes one decoder block at a time and takes the block boundaries from
            it — structure discovery belongs to ``llm_topology``, not to AdaScale. Analyze the
            **float** model before building the sim:
            a topology describes the model rather than the sim, so it is derived once and reused.
            Optional today: when omitted the topology is discovered internally from the sim's graph
            and a warning is raised. Passing it explicitly is recommended, and may become required
            in a future release.
        :param loss_fn: Loss function with signature ``loss_fn(fp_out, quant_out, data_idx)`` returning a scalar
            loss tensor. ``data_idx`` is the index of the current input in ``inputs`` order. Defaults to MSE loss if None.

        Example usage:
            >>> model = DummyModel()
            >>> inputs = ...
            >>> adascale_model_config = adascale_model_config['llama']
            >>> topology = analyze_llm_topology(model, "llama")   # analyze the float model
            >>> sim = QuantizationSimModel(model)
            >>> apply_adascale(sim, inputs, adascale_model_config, num_iterations=num_iterations,
            ...                topology=topology)
            >>> sim.compute_encodings(...)
            >>> sim.export(...)

        .. note::
        1. apply_adascale modifies the weights in-place in the model
        2. compute encodings should not be called before the apply_adascale call
        3. Activation quantizers will remain uninitialized throughout the feature, and so compute encodings needs to be called by the user afterwards. This is so activation encodings will be computed with updated weights taken into account.

        Warning: This feature is currently considered experimental pending API changes
        """
        # pylint: disable=protected-access
        sim._compute_param_encodings(overwrite=False)

        # TODO: if 'topology' is ever made required, delete this branch along with
        # the get_decoder_block_boundaries import.
        if topology is None:
            warnings.warn(
                "apply_adascale() was called without 'topology', so the decoder-block "
                "structure is being discovered internally from the sim. Prefer building it "
                "with aimet_onnx.experimental.llm_topology."
                "analyze_llm_topology(model, model_type) on the float model and passing "
                "topology=...; this argument may become required in a future release.",
                UserWarning,
                stacklevel=2,
            )
            blocks_end_points = get_decoder_block_boundaries(sim.model.model)
        else:
            blocks_end_points = _block_boundaries_from_topology(topology)

        cls._apply_adascale(
            sim,
            inputs,
            blocks_end_points,
            num_iterations,
            loss_fn=loss_fn,
            beta_gamma_lr=adascale_model_config.beta_gamma_lr,
            scales_lr=adascale_model_config.scales_lr,
        )

    @classmethod
    def _apply_adascale(
        cls,
        sim: QuantizationSimModel,
        inputs: Collection[Dict[str, np.ndarray]],
        block_end_points: List[Tuple[str, str]],
        num_iterations: int = 1500,
        *,
        loss_fn: Optional[_LossFn] = None,
        beta_gamma_lr: float = 1e-3,
        scales_lr: float = 5e-4,
    ):
        with cls._disable_activation_quantizers(sim):
            device = get_torch_device(sim.session)
            graph_input_names = [inp.name for inp in sim.session.get_inputs()]
            if graph_input_names != list(inputs[0].keys()):
                raise ValueError(
                    "Graph input names do not match the keys in the provided inputs."
                )

            del sim.session
            gc.collect()
            torch.cuda.empty_cache()
            with tempfile.TemporaryDirectory() as tempdir:
                unquantized_path = os.path.join(tempdir, "unquantized_model.onnx")
                sim_path = os.path.join(tempdir, "sim_model.onnx")
                # Deep copy the model to ensure original weights are maintained
                sim_model: onnx_ir.Model = onnx_ir.from_proto(sim.model.model)
                onnx_ir.passes.common.TopologicalSortPass().call(sim_model)
                unquantized_model = sim_model.clone()
                ir_utils.remove_aimet_quantizers(unquantized_model)

                _validate_boundaries_in_graph(unquantized_model.graph, block_end_points)

                # Save the unquantized model + weights once. Its
                # dtype matches the sim's float dtype (fp16 by default).
                onnx_ir.save(
                    unquantized_model,
                    unquantized_path,
                    external_data="unquantized_model.data",
                )

                for idx in range(len(block_end_points)):
                    if (
                        _DEBUG_NUM_PARTIAL_ITERATIONS is not None
                        and idx >= _DEBUG_NUM_PARTIAL_ITERATIONS
                    ):
                        break

                    if (
                        _DEBUG_NUM_PARTIAL_ITERATIONS_START is not None
                        and _DEBUG_NUM_PARTIAL_ITERATIONS_END is not None
                        and (
                            idx < _DEBUG_NUM_PARTIAL_ITERATIONS_START
                            or idx >= _DEBUG_NUM_PARTIAL_ITERATIONS_END
                        )
                    ):
                        continue

                    _logger.info("Optimizing block: %d", idx)

                    # Step back through leading Casts to land on the true
                    # cross-block residual (fp16 graphs keep the Cast).
                    start_residual = resolve_residual_tensor_name(
                        unquantized_model.graph, block_end_points[idx][0]
                    )
                    end_residual = resolve_residual_tensor_name(
                        unquantized_model.graph, block_end_points[idx][1]
                    )
                    extra_inputs = required_extra_block_inputs(
                        sim_model.graph, [start_residual], [end_residual]
                    )
                    if extra_inputs:
                        _logger.debug(
                            "Block %d adding extra inputs %s", idx, extra_inputs
                        )

                    onnx_ir.save(
                        sim_model, path=sim_path, external_data="sim_model.data"
                    )
                    qsim_sess = ActivationSampler(
                        start_residual,
                        sim_path,
                        sim.providers,
                    )

                    fp_inputs, qsim_inputs = [], []
                    for input in inputs:  # pylint: disable=redefined-builtin
                        sample = qsim_sess.sample_acts(input)
                        qsim_inputs.append(
                            [sample] + [input[name] for name in extra_inputs]
                        )

                    del qsim_sess

                    unquantized_sampler = ActivationSampler(
                        start_residual,
                        unquantized_path,
                        sim.providers,
                    )

                    for input in inputs:
                        sample = unquantized_sampler.sample_acts(input)
                        fp_inputs.append(
                            [sample] + [input[name] for name in extra_inputs]
                        )

                    del unquantized_sampler

                    block_input_output_names = (
                        [start_residual] + extra_inputs,
                        [end_residual],
                    )

                    AdaScale.optimize_adascale_block(
                        sim_model,
                        sim.qc_quantize_op_dict,
                        fp_inputs,
                        qsim_inputs,
                        block_input_output_names,
                        beta_gamma_lr,
                        scales_lr,
                        num_iterations,
                        device,
                        loss_fn=loss_fn,
                    )
                    del fp_inputs, qsim_inputs
                    gc.collect()
                    torch.cuda.empty_cache()

                sim.model.model.CopyFrom(onnx_ir.to_proto(sim_model))
                sim._rebuild_session()  # pylint: disable=protected-access

    @staticmethod
    def get_block_start_end_name(
        blocks_end_points: List[Tuple], block_idx: int, input_list_names: List[str]
    ) -> Tuple[List[str], List[str]]:
        start_tensor, end_tensor = blocks_end_points[block_idx]
        block_input_names = [start_tensor] + input_list_names
        block_output_names = [end_tensor]

        return block_input_names, block_output_names

    @staticmethod
    def optimize_adascale_block(
        sim_model: onnx_ir.Model,
        quantizer_dict: Dict[str, QcQuantizeOp],
        fp_inputs: List[List[np.ndarray]],
        quantized_inputs: List[List[np.ndarray]],
        block_input_output_names: Tuple[List[str], List[str]],
        beta_gamma_lr: float = 1e-3,
        scales_lr: float = 5e-4,
        num_iterations: int = 1500,
        device: torch.device = torch.device("cpu"),
        *,
        loss_fn: Optional[_LossFn] = None,
    ):
        """
        :param sim: QuantizationSimModel object created using the fp32 model
        :param fp_inputs: List of input tensors to the block
        :param quantized_inputs: List of quantized input tensors to the block
        :param block_input_output_names: Tuple of list of input and output tensor names to the block
        :param beta_gamma_lr: learning rate to use for beta/gamma params
        :param scales_lr: learning rate to use for scales params
        :param num_iterations: Number of iterations to optimize for during AdaScale
        :param device: torch device to use for optimization
        :param loss_fn: Loss function with signature ``loss_fn(fp_out, quant_out, data_idx)`` returning a scalar
            loss tensor, where ``data_idx`` is the index of the current input in ``fp_inputs``/``quantized_inputs``.
            Defaults to MSE loss if None.

        This API performs adascale on the block through the following steps:
            - Using the block input and output tensor names, get the onnx block
            - Convert the above onnx block to a pytroch module
            - Apply AdaScale optimization on the above block using the hyperparameters, fp inputs and quantized inputs
            passed to the method
            - Copy back the weights and encodings to the original sim object passed to the method

        Important points to note:
        - fp32 model weights should be original model weights
        - sim would be updated in place with adascaled weights

        """
        if loss_fn is None:
            loss_fn = _mse_loss_fn

        pytorch_block, pt_weights_to_onnx_initializers = get_pt_block(
            sim_model, block_input_output_names
        )
        pytorch_block.requires_grad_(False)

        torch_fp_input = [convert_to_torch(inp) for inp in fp_inputs]
        torch_quant_input = [convert_to_torch(inp) for inp in quantized_inputs]
        pytorch_block.to(device)

        pytorch_block, (torch_fp_input, torch_quant_input), _autocast_ctx = (
            upcast_fp16_block_to_bf16(
                pytorch_block, device, torch_fp_input, torch_quant_input
            )
        )

        fp_out = []
        with torch.no_grad(), _autocast_ctx():
            for input_tensor in torch_fp_input:
                input_tensor = [
                    inp_t.to(device=device) for inp_t in input_tensor
                ]  # Create a new tensor
                out = pytorch_block(*input_tensor).detach()

                out.requires_grad_(False)
                fp_out.append(change_tensor_device_placement(out, torch.device("cpu")))
        per_module_bitwidth = {
            name: quantizer_dict[onnx_name].bitwidth
            for name, onnx_name in pt_weights_to_onnx_initializers.items()
            if onnx_name in quantizer_dict
        }
        pytorch_block = add_qlinear_layers(
            pytorch_block,
            per_module_bitwidth=per_module_bitwidth,
        )
        replace_with_adascale_quantizers(pytorch_block)

        # only set adascale params to train mode
        all_beta_gamma_parameters, all_scale_parameters = get_adascale_trainable_params(
            pytorch_block
        )
        adascale_params = all_beta_gamma_parameters + all_scale_parameters
        for p in adascale_params:
            p.requires_grad = True

        trainable_params = [
            {
                "params": all_beta_gamma_parameters,
                "lr": beta_gamma_lr,
            },
            {
                "params": all_scale_parameters,
                "lr": scales_lr,
            },
        ]

        optimizer = torch.optim.Adam(trainable_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iterations, eta_min=0.0
        )

        gc.collect()
        torch.cuda.empty_cache()

        early_stopping = _create_early_stopping(_EARLY_STOPPING)

        pytorch_block.to(device)
        with torch.set_grad_enabled(True):
            for iteration in progress_bar(range(num_iterations)):
                data_idx = iteration % len(torch_fp_input)
                fp_input = torch_fp_input[data_idx]
                quant_input = torch_quant_input[iteration % len(torch_quant_input)]
                if _QT_SAMPLING_PROB == 1.0:
                    input_tensor = quant_input
                elif _QT_SAMPLING_PROB == 0.0:
                    input_tensor = fp_input
                else:
                    input_tensor = quant_input
                    input_tensor[0] = torch.where(
                        torch.rand_like(quant_input[0], dtype=quant_input[0].dtype).to(
                            device=device
                        )
                        < _QT_SAMPLING_PROB,
                        quant_input[0].to(device=device),
                        fp_input[0].to(device=device),
                    )
                if isinstance(input_tensor, torch.Tensor):
                    input_tensor = [input_tensor]
                input_tensor = [
                    inp_t.to(device=device) for inp_t in input_tensor
                ]  # Create a new tensor
                with _autocast_ctx():
                    quant_out = pytorch_block(*input_tensor)
                batch_fp_out = fp_out[data_idx].to(device)
                loss = loss_fn(
                    batch_fp_out,
                    quant_out,
                    data_idx,
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Early stopping check
                should_stop = early_stopping is not None and early_stopping(loss.item())
                del quant_out, batch_fp_out, loss, input_tensor, fp_input, quant_input
                if should_stop:
                    break

        copy_pt_weights_to_onnx(
            pytorch_block, sim_model, pt_weights_to_onnx_initializers, quantizer_dict
        )
        copy_pt_encodings_to_sim(
            pytorch_block, quantizer_dict, pt_weights_to_onnx_initializers
        )

        del (
            pytorch_block,
            torch_quant_input,
            torch_fp_input,
            optimizer,
            pt_weights_to_onnx_initializers,
            fp_out,
            fp_inputs,
            quantized_inputs,
        )

    @staticmethod
    @contextlib.contextmanager
    def _disable_activation_quantizers(qsim):
        """
        Disable activation quantizers
        :param qsim: Quantization simulator
        """

        enabled_activation_quantizers = [
            name
            for name in qsim.activation_names
            if qsim.qc_quantize_op_dict[name].enabled
        ]

        try:
            for name in enabled_activation_quantizers:
                qsim.qc_quantize_op_dict[name].enabled = False

            yield qsim

        finally:
            for name in enabled_activation_quantizers:
                qsim.qc_quantize_op_dict[name].enabled = True


apply_adascale = AdaScale.apply_adascale
apply_blocklevel_optimization = AdaScale.optimize_adascale_block
