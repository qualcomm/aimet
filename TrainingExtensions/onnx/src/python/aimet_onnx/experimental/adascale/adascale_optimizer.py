# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""AdaScale implementation"""

import contextlib
from typing import Collection, Dict, List, Tuple
import copy
from dataclasses import dataclass
import numpy as np
import torch
import tqdm
import tempfile
import gc

from aimet_onnx.common.utils import AimetLogger  # pylint: disable=import-error
from aimet_onnx.experimental.adascale.utils import (
    convert_to_torch,
    change_tensor_device_placement,
)
from aimet_onnx.utils import (
    get_torch_device,
)
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.experimental.adascale.find_blocks import (
    get_decoder_blocks_end_points,
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
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)


_QT_SAMPLING_PROB = 1.0
_LOSS_FN = torch.nn.MSELoss()
_DEBUG_NUM_PARTIAL_ITERATIONS = None


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
    "phi3": AdaScaleModelConfig(model_type="phi3", beta_gamma_lr=1e-3, scales_lr=5e-4),
}


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

    ADASCALE_PARAM_BW = 4  # TODO remove this temporary solution
    # pylint: disable=unused-argument, unused-variable

    @classmethod
    def apply_adascale(
        cls,
        sim: QuantizationSimModel,
        inputs: Collection[Dict[str, np.ndarray]],
        adascale_model_config: AdaScaleModelConfig,
        num_iterations: int = 1500,
    ):
        """
        :param sim: Quantization Sim model
        :param inputs: (Collection[Dict[str, np.ndarray]]): The set of input samples to use during optimization.
        :param adascale_model_config: Adascale model config. There are pre-defined configs for
                                      Llama, Qwen2, Mistral, Qwen3, Phi3. For other models use AdaScaleModelConfig
        :param num_iterations: Number of iterations to optimize for during AdaScale

        Example usage:
            >>> model = DummyModel()
            >>> inputs = ...
            >>> adascale_model_config = adascale_model_config['llama']
            >>> sim = QuantizationSimModel(model)
            >>> apply_adascale(sim, inputs, adascale_model_config, num_iterations=num_iterations)
            >>> sim.compute_encodings(...)
            >>> sim.export(...)

        .. note::
        1. apply_adascale modifies the weights in-place in the model
        2. compute encodings should not be called before the apply_adascale call
        3. Activation quantizers will remain uninitialized throughout the feature, and so compute encodings needs to be called by the user afterwards. This is so activation encodings will be computed with updated weights taken into account.

        Warning: This feature is currently considered experimental pending API changes
        """
        # pylint: disable=protected-access
        with cls._disable_activation_quantizers(sim):
            # Compute param encodings
            sim._compute_param_encodings(overwrite=False)

            blocks_end_points = get_decoder_blocks_end_points(
                sim, adascale_model_config.model_type
            )

            with tempfile.TemporaryDirectory() as tempdir:
                fp32_model = copy.deepcopy(sim.model.model)
                fp32_model = QuantizationSimModel.remove_quantizers(fp32_model)

                for idx in range(len(blocks_end_points)):
                    if (
                        _DEBUG_NUM_PARTIAL_ITERATIONS is not None
                        and idx >= _DEBUG_NUM_PARTIAL_ITERATIONS
                    ):
                        break

                    _logger.info("Optimizing block: %d", idx)

                    qsim_sess = ActivationSampler(
                        blocks_end_points[idx][0].inputs[0].name,
                        sim.model.model,
                        sim.providers,
                        tempdir,
                    )

                    fp_inputs, qsim_inputs = [], []
                    for input in inputs:  # pylint: disable=redefined-builtin
                        qsim_inputs.append(qsim_sess.sample_acts(input))

                    qsim_sess.restore_graph()
                    del qsim_sess

                    fp32_sampler = ActivationSampler(
                        blocks_end_points[idx][0].inputs[0].name,
                        fp32_model,
                        sim.providers,
                        tempdir,
                    )
                    for input in inputs:
                        fp_inputs.append(fp32_sampler.sample_acts(input))

                    fp32_sampler.restore_graph()
                    del fp32_sampler

                    fp_input_list = []
                    qsim_input_list = []
                    for i in range(len(fp_inputs)):
                        fp_input_list.append(
                            [
                                fp_inputs[i],
                                inputs[i]["attention_mask"],
                                inputs[i]["position_ids"],
                                inputs[i][f"past_key_{idx}_in"],
                                inputs[i][f"past_value_{idx}_in"],
                            ]
                        )

                        qsim_input_list.append(
                            [
                                qsim_inputs[i],
                                inputs[i]["attention_mask"],
                                inputs[i]["position_ids"],
                                inputs[i][f"past_key_{idx}_in"],
                                inputs[i][f"past_value_{idx}_in"],
                            ]
                        )

                    block_input_output_names = AdaScale.get_block_start_end_name(
                        blocks_end_points, idx
                    )

                    AdaScale.optimize_adascale_block(
                        sim,
                        fp_input_list,
                        qsim_input_list,
                        block_input_output_names,
                        adascale_model_config.beta_gamma_lr,
                        adascale_model_config.scales_lr,
                        num_iterations,
                    )
                    del fp_input_list, qsim_input_list, fp_inputs, qsim_inputs
                sim._rebuild_session()  # pylint: disable=protected-access

    @staticmethod
    def get_block_start_end_name(blocks_end_points, block_idx):
        block_inputs = [blocks_end_points[block_idx][0].inputs[0].name]
        common_inputs = ["attention_mask", "position_ids"]
        block_input_names = (
            block_inputs
            + common_inputs
            + [f"past_key_{block_idx}_in", f"past_value_{block_idx}_in"]
        )

        block_output_names = [blocks_end_points[block_idx][1].inputs[0].name]

        return block_input_names, block_output_names

    @staticmethod
    def optimize_adascale_block(
        sim: QuantizationSimModel,
        fp_inputs: List[np.ndarray],
        quantized_inputs: List[np.ndarray],
        block_input_output_names: Tuple[List[str], List[str]],
        beta_gamma_lr: float = 1e-3,
        scales_lr: float = 5e-4,
        num_iterations: int = 1500,
    ):
        """
        :param fp32_model_path: ONNX model path with original FP32 model weights
        :param sim: QuantizationSimModel object created using the fp32 model
        :param fp_inputs: List of input tensors to the block
        :param quantized_inputs: List of quantized input tensors to the block
        :param block_input_output_names: Tuple of list of input and output tensor names to the block
        :param beta_gamma_lr: learning rate to use for beta/gamma params
        :param scales_lr: learning rate to use for scales params
        :param num_iterations: Number of iterations to optimize for during AdaScale

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
        pytorch_block, pt_weights_to_onnx_initializers = get_pt_block(
            sim.model.model, block_input_output_names
        )
        pytorch_block.requires_grad_(False)

        torch_fp_input = convert_to_torch(fp_inputs)
        torch_quant_input = convert_to_torch(quantized_inputs)

        torch_device = get_torch_device(sim.session)
        pytorch_block.to(torch_device)
        fp_out = []
        with torch.no_grad():
            for input_tensor in torch_fp_input:
                if isinstance(input_tensor, torch.Tensor):
                    input_tensor = [input_tensor]

                for i, in_t in enumerate(input_tensor):
                    input_tensor[i] = in_t.to(device=torch_device)
                out = pytorch_block(*input_tensor).detach()

                out.requires_grad_(False)
                fp_out.append(change_tensor_device_placement(out, torch.device("cpu")))
        pytorch_block = add_qlinear_layers(
            pytorch_block, bitwidth=AdaScale.ADASCALE_PARAM_BW
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

        with torch.set_grad_enabled(True):
            for iteration in tqdm.tqdm(range(num_iterations)):
                fp_input = torch_fp_input[iteration % len(torch_fp_input)]
                quant_input = torch_quant_input[iteration % len(torch_quant_input)]
                if _QT_SAMPLING_PROB == 1.0:
                    input_tensor = quant_input
                elif _QT_SAMPLING_PROB == 0.0:
                    input_tensor = fp_input
                else:
                    input_tensor = torch.where(
                        torch.rand_like(quant_input, dtype=quant_input.dtype)
                        < _QT_SAMPLING_PROB,
                        quant_input,
                        fp_input,
                    )
                pytorch_block.to(torch_device)
                if isinstance(input_tensor, torch.Tensor):
                    input_tensor = input_tensor.to(device=torch_device)
                    quant_out = pytorch_block(input_tensor)
                else:
                    for i, in_t in enumerate(input_tensor):
                        input_tensor[i] = in_t.to(device=torch_device)
                    quant_out = pytorch_block(*input_tensor)
                batch_fp_out = fp_out[iteration % len(torch_fp_input)].to(torch_device)
                loss = _LOSS_FN(
                    quant_out,
                    batch_fp_out,
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                del quant_out, batch_fp_out, loss, input_tensor

        copy_pt_weights_to_onnx(
            pytorch_block, sim.model.model, pt_weights_to_onnx_initializers
        )
        copy_pt_encodings_to_sim(pytorch_block, sim, pt_weights_to_onnx_initializers)

        del (
            pytorch_block,
            torch_quant_input,
            torch_fp_input,
            optimizer,
            pt_weights_to_onnx_initializers,
            fp_out,
        )

        gc.collect()
        torch.cuda.empty_cache()

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
