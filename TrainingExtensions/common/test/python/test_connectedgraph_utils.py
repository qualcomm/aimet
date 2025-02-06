# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" This file contains unit tests for testing connected graph utils. """

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from aimet_common.connected_graph.connectedgraph import ConnectedGraph
from aimet_common.connected_graph.operation import Op
from aimet_common.connected_graph.product import Product
from aimet_common.connected_graph import connectedgraph_utils
from aimet_common.model_module import ModelModule

@patch("aimet_common.connected_graph.connectedgraph.ConnectedGraph.__abstractmethods__", set())
def test_serialize_ops():
    conn_graph = get_dummy_connected_graph()
    ops_list = connectedgraph_utils._serialize_ops(conn_graph)
    assert len(ops_list) == 5

    expected_ops = [
        {
            'name': 'op1_dotted_name',
            'type': 'op1_type',
            'inputs': [],
            'outputs': ['op3_dotted_name'],
            'is_functional': True
        },
        {
            'name': 'op2_dotted_name',
            'type': 'op2_type',
            'inputs': [],
            'outputs': ['op3_dotted_name'],
            'is_functional': False
        },
        {
            'name': 'op3_dotted_name',
            'type': 'op3_type',
            'inputs': ['op1_dotted_name', 'op2_dotted_name'],
            'outputs': ['op4_dotted_name', 'op5_dotted_name'],
            'is_functional': True
        },
        {
            'name': 'op4_dotted_name',
            'type': 'op4_type',
            'inputs': ['op3_dotted_name'],
            'outputs': [],
            'is_functional': True
        },
        {
            'name': 'op5_dotted_name',
            'type': 'op5_type',
            'inputs': ['op3_dotted_name'],
            'outputs': [],
            'is_functional': True
        }
    ]

    for op in expected_ops:
        assert op in ops_list

@patch("aimet_common.connected_graph.connectedgraph.ConnectedGraph.__abstractmethods__", set())
def test_serialize_products():
    conn_graph = get_dummy_connected_graph()
    activations, params = connectedgraph_utils._serialize_products(conn_graph)

    # 2 inputs and 3 interop connections
    assert len(activations) == 5

    # 1 param for op4 and 2 params for op5
    assert len(params) == 3

    expected_activations = [
        {
            'producer': None,
            'consumers': ['op1_dotted_name']
        },
        {
            'producer': None,
            'consumers': ['op2_dotted_name']
        },
        {
            'producer': 'op1_dotted_name',
            'consumers': ['op3_dotted_name']
        },
        {
            'producer': 'op2_dotted_name',
            'consumers': ['op3_dotted_name']
        },
        {
            'producer': 'op3_dotted_name',
            'consumers': ['op4_dotted_name', 'op5_dotted_name']
        }
    ]

    expected_params = [
        {
            'name': 'op4.param',
            'op': 'op4_dotted_name'
        },
        {
            'name': 'op5.param1',
            'op': 'op5_dotted_name'
        },
        {
            'name': 'op5.param1',
            'op': 'op5_dotted_name'
        }
    ]

    for product in expected_activations:
        assert product in activations

    for product in expected_params:
        assert product in params

@patch("aimet_common.connected_graph.connectedgraph.ConnectedGraph.__abstractmethods__", set())
def test_export_connected_graph():
    with tempfile.TemporaryDirectory() as tmp_dir:
        conn_graph = get_dummy_connected_graph()
        connectedgraph_utils.export_connected_graph(conn_graph, tmp_dir, 'dummy_cg_export')

        with open(Path(tmp_dir, "dummy_cg_export.json"), 'r') as cg_export_file:
            cg_export = json.load(cg_export_file)

        assert 'ops' in cg_export
        assert 'products' in cg_export
        assert 'activations' in cg_export['products']
        assert 'parameters' in cg_export['products']
        assert len(cg_export['ops']) == 5
        assert len(cg_export['products']['activations']) == 5
        assert len(cg_export['products']['parameters']) == 3

def get_dummy_connected_graph():
    """
    Create a dummy connected graph with 5 ops; 2 inputs, 1 intermediate, and 2 outputs.
    Ops are connected in the following manner:
     1   2
      \ /
       3
      / \
     4   5
     Ops 1 and 2 are model inputs. Op 4 has one parameter product, and Op 5 has two parameter products.
    """
    conn_graph = ConnectedGraph()
    op1 = Op('op1', 'op1_dotted_name', None, False, 'op1_type')
    op2 = Op('op2', 'op2_dotted_name', None, False, 'op2_type')
    op2.model_module = ModelModule('module')
    op3 = Op('op3', 'op3_dotted_name', None, False, 'op3_type')
    op4 = Op('op4', 'op4_dotted_name', None, False, 'op4_type')
    op5 = Op('op5', 'op5_dotted_name', None, False, 'op5_type')

    prod_inp_1 = Product('input1_to_op1', None)
    prod_inp_1.is_model_input = True
    prod_inp_1.add_consumer(op1)
    op1.add_input(prod_inp_1)

    prod_inp_2 = Product('input2_to_op2', None)
    prod_inp_2.is_model_input = True
    prod_inp_2.add_consumer(op2)
    op2.add_input(prod_inp_2)

    prod_1_3 = Product('op1_to_op3', None)
    prod_1_3.producer = op1
    prod_1_3.add_consumer(op3)
    op1.outputs = [prod_1_3]
    op3.add_input(prod_1_3)

    prod_2_3 = Product('op2_to_op3', None)
    prod_2_3.producer = op2
    prod_2_3.add_consumer(op3)
    op2.outputs = [prod_2_3]
    op3.add_input(prod_2_3)

    prod_3_out = Product('op3_to_multiple_ops', None)
    prod_3_out.producer = op3
    prod_3_out.add_consumer(op4)
    prod_3_out.add_consumer(op5)
    op3.outputs = [prod_3_out]
    op4.add_input(prod_3_out)
    op5.add_input(prod_3_out)

    prod_4_param = Product('op4.param', None)
    prod_4_param.is_parm = True
    prod_4_param.add_consumer(op4)
    op4.add_input(prod_4_param)

    prod_5_param_1 = Product('op5.param1', None)
    prod_5_param_1.is_parm = True
    prod_5_param_1.add_consumer(op5)
    op5.add_input(prod_5_param_1)

    prod_5_param_2 = Product('op5.param2', None)
    prod_5_param_2.is_parm = True
    prod_5_param_2.add_consumer(op5)
    op5.add_input(prod_5_param_2)

    conn_graph._ops[op1.name] = op1
    conn_graph._ops[op2.name] = op2
    conn_graph._ops[op3.name] = op3
    conn_graph._ops[op4.name] = op4
    conn_graph._ops[op5.name] = op5

    conn_graph._products[prod_inp_1.name] = prod_inp_1
    conn_graph._products[prod_inp_2.name] = prod_inp_2
    conn_graph._products[prod_1_3.name] = prod_1_3
    conn_graph._products[prod_2_3.name] = prod_2_3
    conn_graph._products[prod_3_out.name] = prod_3_out
    conn_graph._products[prod_4_param.name] = prod_4_param
    conn_graph._products[prod_5_param_1.name] = prod_5_param_1
    conn_graph._products[prod_5_param_2.name] = prod_5_param_2

    return conn_graph
