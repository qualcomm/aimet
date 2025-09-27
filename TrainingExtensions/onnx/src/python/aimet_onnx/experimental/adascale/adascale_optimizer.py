# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""AdaScale implementation"""

import contextlib
from typing import Collection, Dict, Tuple, Optional, Sequence, Any

import copy
from dataclasses import dataclass
from typing import Type
import numpy as np
import torch
import tqdm

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from aimet_common.utils import AimetLogger  # pylint: disable=import-error
from aimet_onnx.utils import (
    add_hook_to_get_activation,
    build_session,
    remove_activation_hooks,
    get_torch_device,
)
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.experimental.adascale.find_blocks import (
    get_decoder_blocks_end_points,
    get_conv_linear_layers_decoder_block,
    get_position_embedding_names,
)

from aimet_onnx.experimental.adascale.quantizer import (
    add_qlinear_layers,
    get_adascale_trainable_params,
    replace_with_adascale_quantizers,
)

from aimet_onnx.experimental.adascale.activation_sampler import ActivationSampler
from aimet_onnx.experimental.adascale.model_converter import ModelConverter

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)


_QT_SAMPLING_PROB = 0.5
_LOSS_FN = torch.nn.MSELoss()


@dataclass
class AdaScaleModelConfig:
    block_type: Type = None  # block types to use in a given model
    beta_gamma_lr: float = 1e-3  # lr for beta and gamma
    scales_lr: float = 5e-4  # lr for s2, s3, [s4]


# mapping of model type and the corresponding adascale config
adascale_model_config_dict = {
    "LlamaModel": AdaScaleModelConfig(
        block_type=LlamaDecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "Qwen2Model": AdaScaleModelConfig(
        block_type=Qwen2DecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
    "MistralModel": AdaScaleModelConfig(
        block_type=MistralDecoderLayer, beta_gamma_lr=1e-3, scales_lr=5e-4
    ),
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

    # pylint: disable=unused-argument, unused-variable

    @classmethod
    def apply_adascale(
        cls,
        sim: QuantizationSimModel,
        inputs: Collection[Dict[str, np.ndarray]],
        adascale_model_config: AdaScaleModelConfig,
        num_iterations: int = 1500,
        providers: Optional[Sequence[str | Tuple[str, Dict[Any, Any]]]] = None,
    ):
        """
        :param sim: Quantization Sim model
        :param inputs: (Collection[Dict[str, np.ndarray]]): The set of input samples to use during optimization.
        :param adascale_model_config: Adascale model config. There are pre-defined configs for
                                      LlamaModel, Qwen2Model, MistralModel. For other models use AdaScaleModelConfig
        :param num_iterations: Number of iterations to optimize for during AdaScale

        Example usage:
            >>> model = DummyModel()
            >>> inputs = ...
            >>> adascale_model_config = adascale_model_config['LlamaModel']
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
        torch_device = get_torch_device(sim.session)
        # pylint: disable=protected-access
        # Disable all activation quantizers
        with cls._disable_activation_quantizers(sim):
            # Compute param encodings
            sim._compute_param_encodings(overwrite=False)

            adascale_blocks_end_points, modules = cls._get_blocks(sim)

            cos_name, sin_name = get_position_embedding_names(
                sim, adascale_blocks_end_points
            )

            # Read in sin and cos
            hook1 = add_hook_to_get_activation(sim.model.model, cos_name)
            hook2 = add_hook_to_get_activation(sim.model.model, sin_name)
            sess = build_session(sim.model.model, providers, None)
            cos, sin = sess.run([cos_name, sin_name], inputs[0])
            cos = np.squeeze(cos, axis=1)
            sin = np.squeeze(sin, axis=1)
            remove_activation_hooks(sim.model.model, hook1)
            remove_activation_hooks(sim.model.model, hook2)

            del sess

            fp32_model = copy.deepcopy(sim.model.model)
            fp32_model = QuantizationSimModel.remove_quantizers(fp32_model)
            converter = ModelConverter(sim, adascale_model_config)

            for idx in range(len(adascale_blocks_end_points)):
                _logger.info("Optimizing decoder block: %d", idx)

                qsim_sess = ActivationSampler(
                    adascale_blocks_end_points[idx][0].inputs[0].name,
                    sim.model.model,
                    sim.providers,
                )

                fp_inputs, qsim_inputs = [], []
                for input in inputs:  # pylint: disable=redefined-builtin
                    qsim_inputs.append(qsim_sess.sample_acts(input))

                qsim_sess.restore_graph()
                # Cleanup qsim session
                del qsim_sess

                fp32_sess = ActivationSampler(
                    adascale_blocks_end_points[idx][0].inputs[0].name,
                    fp32_model,
                    sim.providers,
                )
                for input in inputs:
                    fp_inputs.append(fp32_sess.sample_acts(input))
                fp32_sess.restore_graph()
                # Cleanup fp32 session
                del fp32_sess

                torch_fp_input = [torch.from_numpy(arr).float() for arr in fp_inputs]
                torch_quant_input = [
                    torch.from_numpy(arr).float() for arr in qsim_inputs
                ]

                pytorch_block = converter._copy_weights_onnx_to_pt(idx)
                pytorch_block.requires_grad_(False)

                fp_out = []
                for i, input in enumerate(torch_fp_input):
                    out = pytorch_block(
                        input,
                        position_embeddings=(
                            torch.from_numpy(cos).float(),
                            torch.from_numpy(sin).float(),
                        ),
                        attention_mask=torch.from_numpy(
                            inputs[i]["attention_mask"]
                        ).float(),
                    )[0].detach()
                    out.requires_grad_(False)
                    fp_out.append(out)

                pytorch_block = add_qlinear_layers(pytorch_block)
                replace_with_adascale_quantizers(pytorch_block)

                # only set adascale params to train mode
                all_beta_gamma_parameters, all_scale_parameters = (
                    get_adascale_trainable_params(pytorch_block)
                )
                adascale_params = all_beta_gamma_parameters + all_scale_parameters
                for p in adascale_params:
                    p.requires_grad = True

                trainable_params = [
                    {
                        "params": all_beta_gamma_parameters,
                        "lr": adascale_model_config.beta_gamma_lr,
                    },
                    {
                        "params": all_scale_parameters,
                        "lr": adascale_model_config.scales_lr,
                    },
                ]

                optimizer = torch.optim.Adam(trainable_params)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_iterations, eta_min=0.0
                )

                pytorch_block.to(torch_device)
                with torch.set_grad_enabled(True):
                    for iteration in tqdm.tqdm(range(num_iterations)):
                        input = torch_quant_input[iteration % len(torch_quant_input)]

                        # todo: use probabilitistic sampling of qt and fp input
                        quant_out = pytorch_block(
                            input.to(torch_device),
                            position_embeddings=(
                                torch.from_numpy(cos).float().to(torch_device),
                                torch.from_numpy(sin).float().to(torch_device),
                            ),
                            attention_mask=torch.from_numpy(inputs[0]["attention_mask"])
                            .float()
                            .to(torch_device),
                        )[0]

                        loss = _LOSS_FN(
                            quant_out,
                            fp_out[iteration % len(torch_quant_input)].to(torch_device),
                        )

                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                converter._copy_weights_encodings_pt_to_onnx(pytorch_block)

                # Rebuild the session
                sim._rebuild_session()

                # Cleanup
                del torch_fp_input
                del torch_quant_input

    @staticmethod
    def _get_blocks(qsim):
        """helper to get all the blocks in the model represented by adascale_model_config_dict"""
        end_points = get_decoder_blocks_end_points(qsim)
        conv_linear_modules = get_conv_linear_layers_decoder_block(qsim, end_points)
        return end_points, conv_linear_modules

    @staticmethod
    def _sample_block_inputs(qsim_sess, fp32_sess, inputs):
        qsim_input = qsim_sess.sample_acts(inputs)
        fp_input = fp32_sess.sample_acts(inputs)
        return fp_input, qsim_input

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
