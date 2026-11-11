# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
import torch

from aimet_torch.common.connected_graph.product import Product as _Product
from ..nn import BaseQuantizationMixin
from .operation import Op

_V2 = True


class Product(_Product):
    def __init__(self, name, shape, dtype: torch.dtype | None):
        super().__init__(name, shape)
        self.constant_value = None
        self.dtype = dtype

    def is_quantized(self):
        producer: Optional[Op] = self._producer

        if self.is_parm:
            # Parameters are always quantized with param quantizers
            return True

        if self.is_model_input or self.is_const or not producer:
            # Model inputs and non-param constants are not quantized yet
            return False

        producer_module = producer.get_module()
        if producer_module:
            if _V2:
                # If the producer is a quantized layer, assume the output
                # must have been quantized by the previous layer.
                # This doesn't cover all cases, but it's a reasonable assumption
                # for a short-term fix.
                return isinstance(producer_module, BaseQuantizationMixin) or (
                    producer.is_grid_preserving_op()
                    and (
                        producer.inputs[0].is_quantized()
                        or producer.inputs[0].is_model_input
                        or producer.inputs[0].is_const
                    )
                )

            # Producer is nn.Module. This product will have been quantized by
            # the output quantizer of producer
            return True

        if producer.is_grid_preserving_op():
            # Producer is a functional data movement op, such as torch.reshape.
            # Check if the producer's input were already quantized
            return producer.inputs[0].is_quantized()

        return True
