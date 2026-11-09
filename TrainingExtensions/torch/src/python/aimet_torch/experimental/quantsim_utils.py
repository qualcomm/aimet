# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Experimental quantsim utilities"""

from __future__ import annotations
from typing import overload, Callable, Sequence, Type, TYPE_CHECKING
import torch
import copy

from aimet_torch.common.utils import AimetLogger
from aimet_torch._base.nn.modules import custom

if TYPE_CHECKING:
    from aimet_torch.common.connected_graph.product import Product
    from aimet_torch.meta.connectedgraph import Op
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.quantization.affine.quantizer import AffineQuantizerBase


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


@overload
def propagate_output_encodings(
    sim: QuantizationSimModel,
    module_type: Type[torch.nn.Module],
    safe: bool = True,
):
    """Propagate output encodings of the given module type"""


@overload
def propagate_output_encodings(
    sim: QuantizationSimModel, qmodule: torch.nn.Module, safe: bool = True
):
    """Propagate output encodings of qmodule"""


@overload
def propagate_output_encodings(
    sim: QuantizationSimModel,
    condition: Callable[[torch.nn.Module], bool],
    safe: bool = True,
):
    """Propagate output encodings of all the modules that satisfies the given condition."""


def propagate_output_encodings(sim: QuantizationSimModel, arg, safe: bool = True):
    """
    Propagate output encodings of all the modules that satisfy the given condition.

    :param sim: Quantization simulation model
    :param arg: Module type, qmodule, or condition selecting modules to propagate through
    :param safe: If True, do not propagate through tensors consumed by multiple
        distinct ops, avoiding encoding changes for unrelated consumers. Defaults to True
    """

    if isinstance(arg, type) and issubclass(arg, torch.nn.Module):
        module_type = arg
        condition = lambda module: isinstance(module, module_type)
    elif isinstance(arg, torch.nn.Module):
        qmodule = arg
        condition = lambda module: module is qmodule
    else:
        condition = arg

    if not sim.connected_graph:
        msg = (
            f"Couldn't find a traced graph from {type(sim).__qualname__}. "
            "propagate_output_encodings is only supported when traced graph is present "
            "as part of quantsim"
        )
        raise RuntimeError(msg)

    sim.connected_graph.assert_safe()
    _propagate_output_encodings(sim, condition, safe)


def _propagate_output_encodings(
    sim: QuantizationSimModel,
    condition: Callable[[torch.nn.Module], bool],
    safe: bool = True,
):
    """Propagate output encodings of all the modules that satisfies the given condition."""
    # pylint: disable=redefined-builtin, protected-access
    cg = sim.connected_graph

    def _set_src_qtzr(x: Product, consumer: Op, qtzr: AffineQuantizerBase):
        has_multiple_consumers = len(set(x.consumers)) > 1
        if has_multiple_consumers:
            if safe or not all(qtzr._is_overwrite_allowed.values()):
                # Do not propagate through tensors with multiple consumers in safe mode,
                # or if qtzr has a fixed output range, to avoid catastrophic accuracy drop.
                #
                # For example:
                #
                #       +-> Softmax -> ...
                # x ----+-> ReLU --> QDQ (`qtzr`)
                #                   ∈ [0,?]
                return

            logger.warning(
                "Propagating output encoding through input %d of op %s even though "
                "it is consumed by multiple ops because safe=False. This may reduce "
                "accuracy for other consumers.",
                consumer.inputs.index(x),
                consumer.name,
            )

        producer = x.producer

        if not producer:
            if x.shape is None:
                # ``x`` is a non-tensor root input
                return

            # ``x`` is a root input (i.e. has no producer).
            # In this case, set the input quantizer of the consumer to ``qtzr``
            i = consumer.inputs.index(x)
            qmodule = sim._get_qmodule(consumer)  # pylint: disable=protected-access

            if not qmodule:
                return

            if i < len(qmodule.input_quantizers):
                qmodule.input_quantizers[i] = qtzr
            return

        qmodule = sim._get_qmodule(producer)  # pylint: disable=protected-access

        if qmodule:
            # There exists a qmodule associated with the graph node ``producer``
            # In this case, set the output quantizer of the producer to ``qtzr``
            outputs = getattr(producer, "output_products", [producer.outputs[0]])
            if isinstance(qmodule, custom.Split):
                # torch.split is an output-variadic operation whose number of outputs
                # can't be predicted statically.
                # As a workaround, AIMET qsplit module has only one output quantizer
                # that gets applied to all output tensors
                i = 0
            elif x in outputs:
                i = outputs.index(x)
            else:
                # ``x`` isn't among the recorded outputs of its own producer, which means
                # the graph is internally inconsistent. Skip this edge instead of crashing
                # so that quantsim construction can still succeed.
                logger.warning(
                    "Product %s is not a recorded output of its producer %s. "
                    "Skipping encoding propagation through this edge.",
                    x.name,
                    producer.name,
                )
                return

            if i < len(qmodule.output_quantizers) and qmodule.output_quantizers[i]:
                qmodule.output_quantizers[i] = qtzr

        # Check if the producer is a grid-preserving op (reshape, permute, etc)
        # and propagate the encoding further to the ancestors of the producer
        #
        # For example, in the following graph, even though reshape doesn't have output quantizer,
        # we can continue to search if reshape's producer has output quantizer to tie with `qtzr`
        #
        # --> reshape -----------> relu -------> QDQ
        #   (`producer`)        (`consumer`)   (`qtzr`)
        #
        if producer.is_grid_equivariant_op() or isinstance(qmodule, custom.Cast):
            for input in producer.inputs:
                _set_src_qtzr(input, consumer=producer, qtzr=qtzr)

    for op in reversed(cg.ordered_ops):
        qmodule = sim._get_qmodule(op)  # pylint: disable=protected-access

        if not qmodule:
            continue

        if not condition(qmodule):
            continue

        if len(qmodule.output_quantizers) != 1:
            msg = (
                "Encoding propagation is only supported for qmodules with exactly "
                f"1 output quantizer, but found {len(qmodule.output_quantizers)} "
                "output quantizers"
            )
            raise RuntimeError(msg)

        (qtzr,) = qmodule.output_quantizers

        if qtzr is None:
            msg = (
                "Encoding propagation is only supported for qmodules with exactly "
                "1 output quantizer, but found qmodule.output_quantizers[0] == None"
            )
            raise RuntimeError(msg)

        for input in op.inputs:
            _set_src_qtzr(input, consumer=op, qtzr=qtzr)


def clip_weights_to_7f7f(sim: "QuantizationSimModel"):
    """
    Clip sim model weights which are 16 bit symmetric to have a max of 0x7f7f when quantized.

    :param sim: Quantsim model to clip weights for
    """
    from aimet_torch.quantization.affine import AffineQuantizerBase

    affected_layers = []
    for name, quant_layer in sim.named_qmodules():
        # pylint: disable=too-many-boolean-expressions
        if (
            "weight" in quant_layer.param_quantizers
            and quant_layer.param_quantizers["weight"] is not None
            and quant_layer.param_quantizers["weight"].bitwidth == 16
            and isinstance(quant_layer.param_quantizers["weight"], AffineQuantizerBase)
            and quant_layer.param_quantizers["weight"].symmetric
            and quant_layer.param_quantizers["weight"].is_initialized()
        ):
            clipped_weight = torch.minimum(
                quant_layer.weight,
                quant_layer.param_quantizers["weight"].get_scale() * 0x7F7F,
            )
            with torch.no_grad():
                quant_layer.weight.copy_(clipped_weight)

            affected_layers.append(name)
    logger_str = f"Clipping weights of the following layers to 0x7f7f max quantized value: {affected_layers}"
    logger.debug(logger_str)


def set_matmul_second_input_producer_to_8bit_symmetric(sim: "QuantizationSimModel"):
    """
    set matmul second input producer for 8 bit symmetric encodings.
    :param sim: Quantsim model to apply matmul exception
    """
    from aimet_torch.nn import BaseQuantizationMixin

    model_name = sim.connected_graph._model_name  # pylint: disable=protected-access
    quant_modules = {
        name: module
        for name, module in sim.model.named_modules()
        if isinstance(module, BaseQuantizationMixin)
    }

    def get_connected_graph_op(connected_graph, model_name, name):
        # pylint: disable=protected-access
        original_module = connected_graph._name_to_module[f"{model_name}.{name}"]
        return connected_graph._module_to_op_dict[original_module]

    def get_closest_producer(op: Op):
        prefix = f"{model_name}."
        dotted_name = op.dotted_name
        if dotted_name.startswith(prefix):
            dotted_name = dotted_name[len(prefix) :]
        quant_module = quant_modules.get(dotted_name, None)
        if quant_module:
            if quant_module.output_quantizers[0]:
                return quant_module

            if len(op.input_ops) == 1:
                return get_closest_producer(op.input_ops[0])

            logger.warning(
                "A wrapper of %s with output quantization disabled has no input or more than one input exists. "
                "It's ambiguous to find the nearest producer in this case",
                str(op.dotted_name),
            )
            return None

        if not op.input_ops:
            logger.warning("No input exists for navigation for traversal, aborting..")
            return None

        if len(op.input_ops) > 1:
            logger.warning(
                "Multiple input ops exist, traversal to find closest producer is performed based on the first input"
            )

        return get_closest_producer(op.input_ops[0])

    for name, module in quant_modules.items():
        if isinstance(module, custom.MatMul):
            _, target_quantizer = module.input_quantizers
            matmul_op = get_connected_graph_op(sim.connected_graph, model_name, name)
            if not target_quantizer:
                input_op = matmul_op.inputs[1].producer
                if input_op:
                    closest_producer_wrapper = get_closest_producer(input_op)
                    if closest_producer_wrapper:
                        target_quantizer = closest_producer_wrapper.output_quantizers[0]
                    else:
                        logger.warning(
                            "The closest wrapper could not be found. MatMul exception rule does not apply. "
                            "If you haven't used model preparer, consider using it."
                        )

            if target_quantizer:
                target_quantizer.qmin = -128
                target_quantizer.qmax = 127
                target_quantizer.symmetric = True


class QuantizedMaskAdd(torch.nn.Module):
    # pylint: disable=missing-class-docstring
    def __init__(self):
        from aimet_torch.nn import QuantizationMixin

        super().__init__()
        self.nullrequant = QuantizationMixin.from_module(custom.NullRequant())
        self.add = QuantizationMixin.from_module(custom.Add())

    # pylint: disable=redefined-builtin
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Get shape from input to avoid graph optimization in
        `torch.onnx.utils._optimize_graph` for multiple Reshape ops when sim.export()
        """
        bsz, _, seq_len, ctx_len = input.shape
        return self.add(input, self.nullrequant(mask, [bsz, -1, seq_len, ctx_len]))


@overload
def apply_requant_mask(
    sim: QuantizationSimModel, module_list: Sequence[torch.nn.Module]
):
    """
    Apply adaptive quantized attention mask for given module list.
    Args:
      sim: QuantizationSimModel
      module_list: A Sequence of module that should be considered
                   a MaskAdd operator
    """


@overload
def apply_requant_mask(
    sim: QuantizationSimModel, condition: Callable[[torch.nn.Module], bool]
):
    """
    Apply adaptive quantized attention mask for submodule of sim whose condition(submodule) is True.
    Args:
      sim: QuantizationSimModel
      condition: A function that takes each submodule of sim.model,
                 and return True/False to indicate if the submodule should be
                 considered a MaskAdd operator
    """


def apply_requant_mask(sim: QuantizationSimModel, arg):
    """
    Apply adaptive quantized attention mask to sim model of LLMs.
    """
    # pylint: disable=protected-access
    from aimet_torch.nn.modules.custom import QuantizedAdd

    if isinstance(arg, Sequence):
        module_list = arg
        condition = lambda module: module in module_list
    else:
        condition = arg

    mask_adds, mask_add_act_mins, mask_maxs = [], [], []

    def replace_mask_add_in_model(module: torch.nn.Module):
        for name, child in module.named_children():
            if condition(child):
                if not isinstance(child, QuantizedAdd):
                    msg = (
                        f"apply_requant_mask can only handle {QuantizedAdd}, "
                        f"but got {type(child)}"
                    )
                    raise RuntimeError(msg)

                if all(qtzr is None for qtzr in child.input_quantizers):
                    msg = (
                        "apply_requant_mask expects at least one of the input quantizers "
                        f"to exist, but got {child}"
                    )
                    raise RuntimeError(msg)

                mask_index = 1
                if (
                    child.input_quantizers[0] is not None
                    and child.input_quantizers[0].is_initialized()
                    and torch.all(child.input_quantizers[0].max == 0)
                ):
                    mask_index = 0

                q_mask_add = QuantizedMaskAdd()
                q_mask_add.nullrequant.input_quantizers[0] = copy.deepcopy(
                    child.input_quantizers[mask_index]
                )
                q_mask_add.nullrequant.output_quantizers[0] = copy.deepcopy(
                    child.input_quantizers[mask_index]
                )
                q_mask_add.add.output_quantizers[0] = child.output_quantizers[0]
                setattr(module, name, q_mask_add)
                if (
                    child.input_quantizers[mask_index].is_initialized()
                    and child.output_quantizers[0].is_initialized()
                ):
                    mask_adds.append(q_mask_add)
                    mask_add_act_mins.append(
                        child.output_quantizers[0].min - child.output_quantizers[0].max
                    )
                    mask_maxs.append(child.input_quantizers[mask_index].max)
                else:
                    logger.warning(
                        "The quantizers for %s may remain uninitialized "
                        "only if sim model is about to use `load_encodings`",
                        str(name),
                    )

    sim.model.apply(replace_mask_add_in_model)

    if mask_adds:
        mask_add_act_global_min = min(mask_add_act_mins)
        for mask_add, mask_add_act_min, mask_max in zip(
            mask_adds, mask_add_act_mins, mask_maxs
        ):
            mask_add.nullrequant.input_quantizers[0].set_range(
                mask_add_act_global_min, mask_max
            )
            mask_add.nullrequant.output_quantizers[0].set_range(
                mask_add_act_min, mask_max
            )
