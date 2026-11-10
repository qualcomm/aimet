# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: skip-file

"""Top level API for Adaptive Rounding - Post-Training Quantization (PTQ)"""

import tempfile
from typing import Dict, List, Collection, Optional
from aimet_onnx.common.progress import progress_bar  # pylint: disable=import-error
import numpy as np

# Import AIMET specific modules
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.adaround.adaround_tensor_quantizer import AdaroundTensorQuantizer
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.meta.utils import get_module_act_func_pair
from aimet_onnx import utils
from aimet_onnx.utils import find_shared_param_names
from aimet_onnx.adaround.adaround_optimizer import AdaroundOptimizer
from aimet_onnx.adaround.utils import (
    ModelData,
    ModuleInfo,
    read_attributes_for_op,
    AdaroundSupportedModules,
)

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class Adaround:
    """
    Weight-rounding mechanism for Post Training Quantization (PTQ)
    """

    @classmethod
    def apply_adaround(
        cls,
        sim: QuantizationSimModel,
        inputs: Collection[Dict[str, np.ndarray]],
        num_iterations: int = 10000,
        nodes_to_include: Optional[List[str]] = None,
        *,
        node_names_to_optimize: Optional[List[str]] = None,
    ):
        """
        Optimizes the rounding direction of weights in the QuantizationSimModel to reduce quantization error.

        After applying AdaRound to a QuantizationSimModel object, the quantization encodings will be frozen
        for optimized weights and the sim model will contain updated weight tensors.

        Args:
            sim (QuantizationSimModel): QuantizationSimModel instance to optimize
            inputs (Collection[Dict[str, np.ndarray]]): The set of input samples to use during optimization.
            num_iterations (int): Number of optimization steps to take for each layer. Recommended value is
                10K for weight bitwidths >= 8-bits, 15K for weight bitwidths < 8 bits.
            nodes_to_include: List of node names to optimize. If None, all the nodes(under supported types) will be optimized

        """
        if node_names_to_optimize is not None:
            logger.warning(
                "The argument 'node_names_to_optimize' is deprecated and will be removed in future releases. Use 'nodes_to_include' instead."
            )
            if nodes_to_include is not None:
                raise ValueError(
                    "Both 'nodes_to_include' and 'node_names_to_optimize' parameters cannot be set at the same time. Use only 'nodes_to_include'."
                )
            nodes_to_include = node_names_to_optimize

        # pylint: disable=too-many-locals, protected-access
        sim._compute_param_encodings(overwrite=False)

        module_act_func_pair = get_module_act_func_pair(sim.connected_graph)

        fp32_model = QuantizationSimModel.remove_quantizers(sim.model.model)

        with (
            utils.disable_quantizers(sim, set(sim.activation_names))
            and tempfile.TemporaryDirectory() as tmp_dir
        ):
            # Cache model input data to temporary directory
            cached_dataset = utils.CachedDataset(inputs, len(inputs), tmp_dir)

            param_to_tensor_quantizer_dict = (
                Adaround._create_param_to_tensor_quantizer_dict(sim)
            )
            model_data = ModelData(sim)
            quantized_layer_to_input_tensor_name = (
                Adaround._get_quantized_layer_input_tensor_name(sim)
            )
            shared_param_names = find_shared_param_names(
                sim.connected_graph, param_types=("weight",)
            )

            # AdaRound must be applied to modules in the order of occurrence
            for module in progress_bar(sim.connected_graph.ordered_ops):
                name = module.name
                module_info = model_data.module_to_info[name]

                if nodes_to_include and module.name not in nodes_to_include:
                    continue

                if shared_param_names.intersection(
                    p.name for p, _ in module.parameters.values()
                ):
                    logger.info(
                        "Skipping AdaRound for module with shared parameters: %s",
                        name,
                    )
                    continue

                if (
                    cls._is_supported_layer_type(module_info)
                    and param_to_tensor_quantizer_dict[
                        module_info.params["weight"].name
                    ]._enabled
                ):
                    # Get module's next following activation function
                    act_func = module_act_func_pair.get(name)
                    quantized_input_name = quantized_layer_to_input_tensor_name[name]
                    logger.info(
                        "Started Optimizing weight rounding of module: %s", name
                    )
                    AdaroundOptimizer.adaround_module(
                        module_info,
                        quantized_input_name,
                        sim,
                        fp32_model,
                        act_func,
                        cached_dataset,
                        num_iterations,
                        param_to_tensor_quantizer_dict,
                    )
                    # Freeze quantizer's encodings
                    weight_name = module_info.params["weight"].name
                    sim.qc_quantize_op_dict[weight_name].freeze_encodings()

        # Re-build session since weights have been updated
        sim._rebuild_session()

    @classmethod
    def _is_supported_layer_type(cls, module_info: ModuleInfo):
        if not module_info.type in AdaroundSupportedModules:
            return False

        if not "weight" in module_info.params:
            return False

        if (
            module_info.type in ("Conv", "ConvTranspose")
            and len(module_info.params["weight"].shape) != 4
        ):
            # Only 2d conv/convtranspose is supported
            return False

        attributes = read_attributes_for_op(module_info)
        if "pads" in attributes:
            if len(attributes["pads"]) > 4:
                logger.info(
                    "Skipping the Convolution layer because padding size greater than 4 is not supported for optimization"
                )
                return False

        return True

    @staticmethod
    def _create_param_to_tensor_quantizer_dict(
        quant_sim: QuantizationSimModel,
    ) -> Dict[str, AdaroundTensorQuantizer]:
        """
        Create Adaround tensor quantizers for weight tensor

        :param quant_sim: Quant sim
        :return: Dict of param name to AdaroundTensorQuantizer
        """
        param_to_tq_dict = {}
        for param_name in quant_sim.param_names:
            quantizer = quant_sim.qc_quantize_op_dict[param_name]
            ch_axis = -1
            if quantizer.quant_info.usePerChannelMode:
                ch_axis = quantizer.quant_info.channelAxis
            adaround_quantizer = AdaroundTensorQuantizer(
                quantizer.bitwidth,
                quantizer.enabled,
                quantizer.encodings,
                ch_axis,
            )
            param_to_tq_dict[param_name] = adaround_quantizer

        return param_to_tq_dict

    @staticmethod
    def _get_quantized_layer_input_tensor_name(sim):
        quantized_layer_to_input_tensor_name = {}
        for node in sim.model.model.graph.node:
            if node.op_type in AdaroundSupportedModules:
                quantized_layer_to_input_tensor_name[node.name] = node.input[0]
        return quantized_layer_to_input_tensor_name


apply_adaround = Adaround.apply_adaround
