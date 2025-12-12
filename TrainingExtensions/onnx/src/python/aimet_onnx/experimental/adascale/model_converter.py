# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.experimental.adascale.quantizer import QuantizedLinear, QuantizedConv2d

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)
import onnx
from onnx import numpy_helper
from onnx.utils import Extractor
from onnx2torch import convert
from aimet_onnx.experimental.adascale.onnx2torch_ext import *  # pylint: disable=wildcard-import, unused-wildcard-import
from aimet_onnx.utils import add_value_info
from aimet_onnx.quantsim import QuantizationSimModel
from onnx2torch.onnx_graph import OnnxGraph
from typing import Tuple, List, Dict, Collection
from aimet_onnx.common.quantsim import calculate_delta_offset

filter_op = ["MatMul", "Conv"]


def _get_onnx_subgraph(
    extractor: Extractor,
    block_input_output_names: Tuple[List[str], List[str]],
):
    """
    Given a onnx block end points get onnx subgraph
    """
    block_input_names, block_output_names = block_input_output_names
    try:
        block_fp32_model = extractor.extract_model(
            block_input_names,
            block_output_names,
        )
        return block_fp32_model
    except Exception:
        raise RuntimeError(  # pylint: disable=raise-missing-from
            f"Unable to extract onnx subgraph for given block input/output {block_input_output_names}"
        )


def _get_onnx_block_info(onnx_subgraph: onnx.ModelProto):
    """
    For an onnx subgraph get onnx param name from initializer list map
    """
    graph = onnx_subgraph.graph
    name_to_node_filtered = {n.name: n for n in graph.node if n.op_type in filter_op}
    initializer_name_to_index_map = {
        init.name: idx for idx, init in enumerate(graph.initializer)
    }
    node_name_to_onnx_param = {}
    for node in name_to_node_filtered.values():
        # TODO remove using "bias" word search and add op specific logic instead
        if node.op_type == "Conv":
            node_name_to_onnx_param[OnnxGraph.generate_node_name(node)] = node.input[1]
        else:
            for edge in node.input:
                if edge in initializer_name_to_index_map and "bias" not in edge:
                    # Bias will not be updated so we donot need to keep track of bias
                    node_name_to_onnx_param[OnnxGraph.generate_node_name(node)] = edge
    return node_name_to_onnx_param


def get_pt_block(
    model: onnx.ModelProto, block_input_output_names: Tuple[List[str], List[str]]
):
    """
    Given a onnx block end points get a pytorch block
    :param model: onnx.ModelProto
    :param block_input_output_names: input/output names for block end points
    """
    # As of onnx 1.18, value info must be populated prior to instantiating Extractor
    with add_value_info(model):
        extractor = Extractor(model)
        onnx_block = _get_onnx_subgraph(extractor, block_input_output_names)
        onnx_block = QuantizationSimModel.remove_quantizers(onnx_block)
        param_map = _get_onnx_block_info(onnx_block)
        return convert(onnx_block), param_map


def copy_pt_weights_to_onnx(
    pt_block: torch.fx.GraphModule,
    onnx_model: onnx.ModelProto,
    param_map: Collection[Dict[str, str]],
):
    """
    Given a pt_block with adascale params computed, copy the params to onnx model
    :param pt_block: pytorch block with adascale weight quantizers
    :param onnx_model: onnx model before adascale
    :param pt_weights_to_onnx_initializers: Mapping between PT weight names to ONNX initializers
    """
    initializer_name_to_index_map = {
        init.name: idx for idx, init in enumerate(onnx_model.graph.initializer)
    }

    for name, module in pt_block.named_modules():
        if param_map.get(name) is None:
            continue
        if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
            pytorch_weight = (
                module.param_quantizers["weight"]
                .get_folded_weight(module.weight)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            pytorch_weight = module.weight.detach().cpu().numpy()

        if isinstance(module, torch.nn.Linear):
            pytorch_weight = pytorch_weight.T

        onnx_tensor_name = param_map[name]
        onnx_param_tensor = numpy_helper.to_array(
            onnx_model.graph.initializer[
                initializer_name_to_index_map[onnx_tensor_name]
            ]
        )
        if pytorch_weight.shape != onnx_param_tensor.shape:
            raise ValueError(
                f"pt param shape {pytorch_weight.shape} did not match onnx shape {onnx_param_tensor.shape}"
            )
        if not (pytorch_weight == onnx_param_tensor).all():
            onnx_model.graph.initializer[
                initializer_name_to_index_map[onnx_tensor_name]
            ].CopyFrom(numpy_helper.from_array(pytorch_weight, onnx_tensor_name))
            _logger.info(
                "Copy from PyTorch to ONNX: torch : %s  onnx param : %s",
                name,
                onnx_tensor_name,
            )


def copy_pt_encodings_to_sim(
    pt_block: torch.fx.GraphModule,
    sim: QuantizationSimModel,
    pt_weights_to_onnx_initializers: Collection[Dict[str, str]],
):
    """
    Given the PT block with adascale params computed, copy the encodings to sim
    :param pt_block: pytorch block with adascale weight quantizers
    :param sim: QuantizationSimModel instance
    :param pt_weights_to_onnx_initializers: Mapping between PT weight names to ONNX initializers
    """
    for name, module in pt_block.named_modules():
        if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
            onnx_param_name = pt_weights_to_onnx_initializers[name]

            # copy encodings over to onnx quantizers
            new_min = module.param_quantizers["weight"].get_min().detach().cpu().numpy()
            new_max = module.param_quantizers["weight"].get_max().detach().cpu().numpy()

            enc = sim.qc_quantize_op_dict[onnx_param_name].get_encodings()

            if len(new_min) != len(enc) or len(new_max) != len(enc):
                raise RuntimeError(
                    "Encodings of the onnx quantizer and adascale quantizer have different lengths"
                )

            for i, encoding in enumerate(enc):
                delta, offset = calculate_delta_offset(
                    min_val=new_min[i],
                    max_val=new_max[i],
                    bitwidth=module.param_quantizers["weight"].bitwidth,
                    use_symmetric_encodings=True,
                    use_strict_symmetric=False,
                )
                encoding.delta = delta
                encoding.offset = offset
                encoding.min = new_min[i]
                encoding.max = new_max[i]
            sim.qc_quantize_op_dict[onnx_param_name].load_encodings(enc)
            sim.qc_quantize_op_dict[onnx_param_name].freeze_encodings()
