# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Phi-3.5 ONNX model class"""

import torch

from aimet_onnx import quantsim
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.phi3 import Phi_3
from GenAITests.shared.models.generator import Generator
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache

from GenAITests.onnx.models.utils.torch_onnx_export_utils import (
    get_onnx_model,
    get_model_checkpoint_path,
)
from GenAITests.onnx.models.utils.quantsim_utils import (
    _set_tensors_to_output_8b_sym,
    _tie_quantizers_for_kv_cache,
    _set_lm_head_to_8b,
    get_ort_providers,
    AttributePatch,
)


@YAMLConfigParser.register_model
class Phi_3_ONNX(Phi_3):
    @classmethod
    def instantiate_quantsim(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int,
        small_model: bool = False,
    ):
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        model = cls.instantiate_model(model_id, small_model)

        exportable_model = ONNXExportableModuleWithCache(model)

        dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int)
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        assembled_dummy_inputs = Generator.prepare_inputs(
            model,
            dummy_input_ids,
            dummy_attention_mask,
            [],
            sequence_length,
            context_length,
        )

        onnx_model = get_onnx_model(
            get_model_checkpoint_path(model_id),
            exportable_model,
            context_length,
            assembled_dummy_inputs,
            Generator.get_input_names(model.config.num_hidden_layers),
            Generator.get_output_names(model.config.num_hidden_layers),
        )

        with (
            AttributePatch(quantsim, "op_types_to_tie_qtzrs", ["Concat"]),
            AttributePatch(quantsim, "_tie_qtzrs", True),
            AttributePatch(
                quantsim,
                "op_outputs_to_ignore",
                quantsim.op_outputs_to_ignore + ["Slice", "Constant"],
            ),
        ):
            quant_sim = QuantSimOnnx(
                model=onnx_model,
                quant_scheme="min_max",
                default_activation_bw=16,
                default_param_bw=4,
                config_file=cls.get_quantsim_config(),
                providers=get_ort_providers(
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                ),
            )

        # Setting kv_cache and some other layers to 8-bit
        _set_tensors_to_output_8b_sym(quant_sim)
        # Setting the LM head weights to 8-bit.
        _set_lm_head_to_8b(quant_sim)
        # Tie kv_cache
        _tie_quantizers_for_kv_cache(quant_sim)

        return quant_sim
