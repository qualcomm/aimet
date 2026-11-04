# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Pytorch Operation class and utilities"""

from typing import Union
import torch

from aimet_torch.common.connected_graph.operation import Op as _Op


class Op(_Op):
    """Subclass Op inherited from aimet_torch.common.connected_graph.operation.Op"""

    def __init__(
        self,
        name: str,
        dotted_name: str,
        output_shape,
        is_anonymous: bool,
        op_type: str,
        residing_module: Union[torch.nn.Module, None],
    ):
        """
        Initializer for Op
        :param name: name of the operation
        :param dotted_name: dotted name of the operation
        :param output_shape: shape of the output product of the operation
        :param is_anonymous: whether this is an anonymous operation
        :param op_type: type of the operation
        :param residing_module: Torch module in which the op is situated
        """
        super().__init__(name, dotted_name, output_shape, is_anonymous, op_type)
        self.residing_module = residing_module

    def is_grid_preserving_op(self) -> bool:
        """
        Returns True if self is a grid-preserving op.

        Unary function `f(x1, x2, ..., xn)` is grid-preserving
        if and only if `q(f(x)) == f(q(x))` for arbitrary quantization function `q`.

        Note that grid-preserving ops is a subset of grid-equivariant ops
        defined in `_is_grid_equivariant_op`
        """
        return self._is_grid_equivariant_op(unary_only=True)

    def is_grid_equivariant_op(self) -> bool:
        """
        Returns True if self is a grid-equivariant op.

        N-ary function `f(x1, x2, ..., xn)` is grid-equivariant
        if and only if `q(f(x1, x2, ..., xn)) == f(q(x1), q(x2), ..., q(xn))`
        for arbitrary quantization function `q`.
        """
        return self._is_grid_equivariant_op(unary_only=False)

    def _is_grid_equivariant_op(self, unary_only: bool) -> bool:
        from .connectedgraph import ConnectedGraph
        from aimet_torch.common.onnx._utils import (
            _is_grid_preserving_op,
            _is_grid_equivariant_op,
        )
        from ..onnx_utils import map_torch_types_to_onnx
        from ..nn import QuantizationMixin

        module = self.get_module()

        if not module:
            return (
                self.type == "CG_Split"
                or self.type in ConnectedGraph.math_invariant_types
            )

        if isinstance(module, QuantizationMixin):
            module = module.get_original_module()

        module_cls = type(module)
        onnx_op_types = map_torch_types_to_onnx.get(module_cls)

        if not onnx_op_types:
            # ONNX op type unknown.
            # To be safe, we should assume non-grid-preserving op in this case.
            return False

        if unary_only:
            _filter = _is_grid_preserving_op
        else:
            _filter = _is_grid_equivariant_op

        return all(_filter(op_type) for op_type in onnx_op_types)
