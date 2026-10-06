# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Model transforms (adaptations) for GenAI testing."""

from GenAILab.qai_hub_lm.transforms.sha import (
    LlamaSHAAdaptation,
    Qwen3SHAAdaptation,
)
from GenAILab.qai_hub_lm.transforms.sha_conv import (
    LlamaSHAConvAdaptation,
    Qwen3SHAConvAdaptation,
)
from GenAILab.qai_hub_lm.transforms.fast_exportable import (
    Qwen2VLFastExportableAdaptation,
    Qwen3VLFastExportableAdaptation,
)
from GenAILab.qai_hub_lm.transforms.exportable_moe import (
    ExportableMoEAdaptation,
    PredicatedExperts,
    forced_expert_activation,
    set_onnx_force_all,
)
from GenAILab.qai_hub_lm.transforms.split_fused_layers import (
    Phi3SplitFusedLayersAdaptation,
)
from GenAILab.qai_hub_lm.transforms.attention_mask_scale import (
    AttentionMaskScaleAdaptation,
)

# Qwen 3.5 linear-attention export support and expert subselection require a
# transformers version that ships the qwen3_5 and qwen3_5_moe models. Guard the
# imports so older environments still load the rest of the transforms package.
try:
    from GenAILab.qai_hub_lm.transforms.exportable_linear_attention import (
        Qwen3_5ExportableLinearAttentionAdaptation,
    )
    from GenAILab.qai_hub_lm.transforms.expert_subselection import (
        ExpertSubselectionAdaptation,
    )
except ImportError:
    import warnings

    warnings.warn(
        "Qwen 3.5 ExportableLinearAttention and ExpertSubselection adaptations "
        "are not available. "
        "Please upgrade to a later version of transformers to use this model."
    )
