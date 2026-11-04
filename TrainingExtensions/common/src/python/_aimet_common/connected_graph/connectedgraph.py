# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Connected graph abstract class and utilities"""

from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Optional
from ..connected_graph.operation import Op
from ..connected_graph.product import Product


class ConnectedGraph(ABC):
    """ConnectedGraph abstract class"""

    def __init__(self):
        self._ops = {}
        self._products = {}

    @abstractmethod
    def get_op_from_module_name(self, name: str):
        """Given the name of a operation/module, return the corresponding op in ops dict"""

    def get_all_ops(self):
        """Returns the ops dictionary"""
        return self._ops

    def get_all_products(self):
        """Returns the products dictionary"""
        return self._products

    def get_product(self, name: str) -> Product:
        """
        Returns the product with the name passed in the argument
        :param name: Product name
        """
        return self._products.get(name)


def get_ordered_ops(
    list_of_starting_ops: List[Op],
    consumers_of: Optional[Callable[[Op], Iterable[Op]]] = None,
) -> List[Op]:
    """
    Function to get all the ops in connected graph based on occurrence by Depth First Traversal
    :param list_of_starting_ops: List of starting ops of the graph
    :param consumers_of: If given, returns the ops to follow from an op, in place of its output_ops
    :return: List of connected graph ops in order of occurrence
    """
    if consumers_of is None:
        consumers_of = lambda op: op.output_ops  # pylint: disable=unnecessary-lambda-assignment

    #  Set of all ops that have been visited
    visited_ops_set = set()

    # List of all ops in order of occurrence
    ordered_ops_list = []

    op_stack = list_of_starting_ops[::-1]

    while op_stack:
        current_op = op_stack[-1]
        unvisited_consumers = [
            consumer
            for consumer in consumers_of(current_op)
            if consumer not in visited_ops_set
        ]
        if unvisited_consumers:
            op_stack.extend(reversed(unvisited_consumers))
            continue

        op_stack.pop()
        if current_op in visited_ops_set:
            continue

        visited_ops_set.add(current_op)
        ordered_ops_list.append(current_op)

    ordered_ops_list.reverse()
    return ordered_ops_list
