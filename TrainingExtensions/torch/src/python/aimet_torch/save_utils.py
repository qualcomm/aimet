# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utilities to save a models and related parameters"""

from aimet_torch.nn import BaseQuantizationMixin


class SaveUtils:
    """Utility class to save a models and related parameters"""

    @staticmethod
    def remove_quantization_wrappers(module):
        """
        Removes quantization wrappers from model (in place)
        :param module: Model
        """
        for module_name, module_ref in module.named_children():
            if isinstance(module_ref, BaseQuantizationMixin):
                setattr(module, module_name, module_ref.get_original_module())
            # recursively call children modules
            else:
                SaveUtils.remove_quantization_wrappers(module_ref)
