# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" Connected graph abstract class and utilities """

from abc import ABC, abstractmethod
from typing import List
from aimet_common.connected_graph.operation import Op
from aimet_common.connected_graph.product import Product


class ConnectedGraph(ABC):
    """ ConnectedGraph abstract class """

    def __init__(self):
        self._ops = dict()
        self._products = dict()

    @abstractmethod
    def get_op_from_module_name(self, name: str):
        """ Given the name of a operation/module, return the corresponding op in ops dict """

    def get_all_ops(self):
        """ Returns the ops dictionary """
        return self._ops

    def get_all_products(self):
        """ Returns the products dictionary """
        return self._products

    def get_product(self, name: str) -> Product:
        """
        Returns the product with the name passed in the argument
        :param name: Product name
        """
        return self._products.get(name)


def get_ordered_ops(list_of_starting_ops: List[Op]) -> List[Op]:
    """
    Function to get all the ops in connected graph based on occurrence by Depth First Traversal
    :param list_of_starting_ops: List of starting ops of the graph
    :return: List of connected graph ops in order of occurrence
    """
    #  Set of all ops that have been visited
    visited_ops_set = set()

    # List of all ops in order of occurrence
    ordered_ops_list = []

    op_stack = list_of_starting_ops[::-1]

    while op_stack:
        current_op = op_stack[-1]
        unvisited_consumers = [consumer for consumer in current_op.output_ops if consumer not in visited_ops_set]
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
