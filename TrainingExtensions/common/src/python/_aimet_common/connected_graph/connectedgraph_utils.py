# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Utilities for ConnectedGraph"""

import json
import os
from typing import List, Dict, Tuple, Set
from ..connected_graph.connectedgraph import ConnectedGraph, get_ordered_ops
from ..connected_graph.operation import Op

__all__ = [
    "CG_SPLIT",
    "_get_all_input_and_consumer",
    "export_connected_graph",
    "get_all_input_ops",
    "get_all_ops_with_constant_inputs",
    "get_all_output_ops",
    "get_ordered_ops",
]

# Op type of ConnectedGraph Split to distinguish from native split operation
CG_SPLIT = "CG_Split"


def get_all_input_ops(conn_graph: ConnectedGraph) -> List[Op]:
    """
    Return a list of all operations with model inputs as inputs.

    :param conn_graph: Connected graph to search for input ops in
    :return: List of all operations with no inputs
    """
    memo = set()
    ret = []
    for _, op in _get_all_input_and_consumer(conn_graph):
        if op not in memo:
            memo.add(op)
            ret.append(op)
    return ret


def _get_all_input_and_consumer(conn_graph: ConnectedGraph):
    all_ops = conn_graph.get_all_ops().values()
    for op in all_ops:
        for item in op.inputs:
            if not item.producer and item.is_model_input:
                yield item, op


def get_all_ops_with_constant_inputs(conn_graph: ConnectedGraph) -> Set[Op]:
    """
    Return a set of all operations with constant inputs.

    :param conn_graph: Connected graph to search for constant input ops in
    :return: Set of all operations with constant inputs
    """

    constant_input_ops = set()
    for op in conn_graph.get_all_ops().values():
        for product in op.inputs:
            if product.is_const:
                constant_input_ops.add(op)

    return constant_input_ops


def get_all_output_ops(conn_graph: ConnectedGraph) -> List[Op]:
    """
    Return a list of all operations with no outputs
    :param conn_graph: Connected graph to search for output ops in
    :return: List of all operations with no outputs
    """
    all_ops = conn_graph.get_all_ops().values()
    output_ops = [op for op in all_ops if not op.output_ops]
    return output_ops


def export_connected_graph(conn_graph: ConnectedGraph, path: str, filename_prefix: str):
    """
    Serialize and export the connected graph as a json file
    :param conn_graph: Connected graph to serialize and export
    :param path: Folder path to save exported connected graph. Does not include filename.
    :param filename_prefix: Filename to save exported connected graph. Do not include '.json', which will be added
        automatically.
    """
    ops_list = _serialize_ops(conn_graph)
    activation_products_list, param_products_list = _serialize_products(conn_graph)
    connected_graph_export_dict = {
        "ops": ops_list,
        "products": {
            "activations": activation_products_list,
            "parameters": param_products_list,
        },
    }

    connected_graph_export_path = os.path.join(path, filename_prefix + ".json")
    with open(connected_graph_export_path, "w") as encoding_fp_json:
        json.dump(
            connected_graph_export_dict, encoding_fp_json, sort_keys=True, indent=4
        )


def _serialize_ops(conn_graph: ConnectedGraph) -> List[Dict[str, str]]:
    """
    Get a list of ops serialized as dictionary objects with name, type, inputs, and outputs information
    :param conn_graph: Connected graph containing ops to serialize
    :return: List of ops serialized
    """
    ops_list = []
    input_ops = get_all_input_ops(conn_graph)
    for op in get_ordered_ops(input_ops):
        ops_list.append(
            {
                "name": op.dotted_name,
                "type": op.type,
                "inputs": [op.dotted_name for op in op.input_ops],
                "outputs": [op.dotted_name for op in op.output_ops],
                "is_functional": op.get_module() is None,
            }
        )
    return ops_list


def _serialize_products(
    conn_graph: ConnectedGraph,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Get lists of products serialized as dictionary objects with name and op for parameter products, and name, producer,
    and consumers for activation products.
    :param conn_graph: Connected graph containing products to serialize
    :return: Tuple of activation products and parameter products serialized
    """
    param_products_list = []
    activation_products_list = []
    for product in conn_graph.get_all_products().values():
        if product.is_parm:
            param_products_list.append(
                {"name": product.name, "op": product.consumers[0].dotted_name}
            )
        else:
            producer_name = None
            if product.producer:
                producer_name = product.producer.dotted_name
            consumer_names = []
            for consumer in product.consumers:
                consumer_names.append(consumer.dotted_name)
            activation_products_list.append(
                {"producer": producer_name, "consumers": consumer_names}
            )
    return activation_products_list, param_products_list
