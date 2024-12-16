# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
import json
import os
import unittest
import copy

import onnx
import pytest
import torch
import tempfile
from torch.nn.utils.rnn import pack_padded_sequence

import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_common.utils import AimetLogger
from aimet_torch.v1.qc_quantize_op import QcQuantizeOpMode
from aimet_torch.v1.qc_quantize_recurrent import QcQuantizeRecurrent
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.v1.tensor_quantizer import LearnedGridTensorQuantizer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class TestCase():
    def __init__(self, test_name:str, model, input_shape, valid_hx=False, sequence_lens=None, device='cpu'):
        self.test_name = test_name
        self.model = model.to(device)
        self.input_shape = input_shape
        self.valid_hx = valid_hx
        self.device = device
        self.sequence_lens = sequence_lens


# pylint: disable=too-many-public-methods
class TestQcQuantizeRecurrentOp(unittest.TestCase):
    """
    Test cases for Quantize Recurrent Custom Op
    """
    testcases = [
        TestCase(test_name="rnn_single_layer_no_bias",
                 model=torch.nn.RNN(input_size=4, hidden_size=5, num_layers=1, bias=False),
                 input_shape=(5, 3, 4)),
        TestCase(test_name="rnn_single_layer",
                 model=torch.nn.RNN(input_size=4, hidden_size=5, num_layers=1),
                 input_shape=(5, 3, 4)),

        TestCase(test_name="rnn_single_layer_valid_hx",
                 model=torch.nn.RNN(input_size=4, hidden_size=5, num_layers=1),
                 input_shape=(5, 3, 4),
                 valid_hx=True),

        TestCase(test_name="rnn_single_layer_batch_first",
                 model=torch.nn.RNN(input_size=4, hidden_size=5, num_layers=1, batch_first=True),
                 input_shape=(3, 5, 4)),

        TestCase(test_name="rnn_single_layer_packed_sequence",
                 model=torch.nn.RNN(input_size=4, hidden_size=5, num_layers=1),
                 input_shape=(5, 3, 4),
                 sequence_lens=[1, 5, 3]),

        TestCase(test_name="rnn_multilayer",
                 model=torch.nn.RNN(input_size=4, hidden_size=5, num_layers=2),
                 input_shape=(5, 3, 4)),

        TestCase(test_name="rnn_multilayer_hx_input",
                 model=torch.nn.RNN(input_size=4, hidden_size=5, num_layers=2),
                 input_shape=(5, 3, 4),
                 valid_hx=True),

        TestCase(test_name="rnn_multilayer_batch_first",
                 model=torch.nn.RNN(input_size=4, hidden_size=5, num_layers=3, batch_first=True),
                 input_shape=(3, 5, 4)),

        TestCase(test_name="rnn_bidirectional",
                 model=torch.nn.RNN(input_size=4, hidden_size=5, num_layers=1, bidirectional=True),
                 input_shape=(5, 3, 4)),

        TestCase(test_name="rnn_bidirectional_batch_first",
                 model=torch.nn.RNN(input_size=4, hidden_size=5, num_layers=1, bidirectional=True, batch_first=True),
                 input_shape=(3, 5, 4)),

        TestCase(test_name="rnn_multilayer_bidrectional",
                 model=torch.nn.RNN(input_size=4, hidden_size=5, num_layers=2, bidirectional=True),
                 input_shape=(3, 5, 4)),

        TestCase(test_name="rnn_multilayer_bidrectional_batch_first",
                 model=torch.nn.RNN(input_size=4, hidden_size=5, num_layers=2, bidirectional=True, batch_first=False),
                 input_shape=(3, 5, 4)),

        TestCase(test_name="lstm_single_layer",
                 model=torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=1),
                 input_shape=(5, 3, 4)),

        TestCase(test_name="lstm_single_layer_valid_hx",
                 model=torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=1),
                 input_shape=(5, 3, 4),
                 valid_hx=True),

        TestCase(test_name="lstm_single_layer_batch_first",
                 model=torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=1, batch_first=True),
                 input_shape=(3, 5, 4)),

        TestCase(test_name="lstm_bidirectional",
                 model=torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=1, bidirectional=True),
                 input_shape=(5, 3, 4),
                 sequence_lens=[1, 5, 3]),

        TestCase(test_name="lstm_multilayer",
                 model=torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=2),
                 input_shape=(5, 3, 4)),

        TestCase(test_name="lstm_multilayer_valid_hx",
                 model=torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=2),
                 input_shape=(5, 3, 4),
                 valid_hx=True),

        TestCase(test_name="lstm_multilayer_batch_first",
                 model=torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=3, batch_first=True),
                 input_shape=(3, 5, 4)),

        # @TODO: Following testcase failing w/ torch 2.1.2 version
        # TestCase(test_name="lstm_multilayer_bidirectional_large_dimension",
        #          model=torch.nn.LSTM(input_size=10, hidden_size=20, num_layers=3, bidirectional=True, batch_first=True),
        #          input_shape=(25, 500, 10),
        #          sequence_lens=([480,  31, 210, 9, 411, 498, 298, 345, 241, 403, 479, 347,  42,
        #                          95, 380, 454, 470,  57, 293, 457, 194,  45, 366, 458, 172])),

        TestCase(test_name="gru_single_layer",
                 model=torch.nn.GRU(input_size=4, hidden_size=5, num_layers=1),
                 input_shape=(5, 3, 4)),

        TestCase(test_name="gru_single_layer_batch_first",
                 model=torch.nn.GRU(input_size=4, hidden_size=5, num_layers=1, batch_first=True),
                 input_shape=(3, 5, 4)),

        TestCase(test_name="gru_bidirectional",
                 model=torch.nn.GRU(input_size=4, hidden_size=5, num_layers=1, bidirectional=True),
                 input_shape=(5, 3, 4)),

        TestCase(test_name="gru_multilayer",
                 model=torch.nn.GRU(input_size=4, hidden_size=5, num_layers=2),
                 input_shape=(5, 3, 4)),

        TestCase(test_name="gru_multilayer_batch_first",
                 model=torch.nn.GRU(input_size=4, hidden_size=5, num_layers=3, batch_first=True),
                 input_shape=(3, 5, 4))
    ]

    def verify_custom_op(self, tc: TestCase):
        """
       helper method to run quant RNN test
        """
        quant_op = QcQuantizeRecurrent(module_to_quantize=tc.model, weight_bw=32, activation_bw=32, is_symmetric=False,
                                       quant_scheme=QuantScheme.post_training_tf_enhanced, round_mode='nearest',
                                       data_type=QuantizationDataType.int)

        for input_quantizer in quant_op.input_quantizers.values():
            input_quantizer.enabled = False
        for name, param in quant_op.named_parameters(recurse=False):
            quant_op.param_quantizers[name].enabled = False

        x = torch.rand(tc.input_shape).to(tc.device)
        h = None
        if tc.valid_hx:
            o_rnn, h_rnn = tc.model(input=x, hx=None)
            if isinstance(h_rnn, tuple):
                h = (h_rnn[0], h_rnn[1])
            else:
                h = torch.stack([h_rnn])
                h = h[0]
        o_rnn, h_rnn = tc.model(input=x, hx=h)
        o_qc_rnn, h_qc_rnn = quant_op(x, hx=h)

        if not isinstance(h_qc_rnn, tuple):
            h_qc_rnn = [h_qc_rnn]
            h_rnn = [h_rnn]
        for h, h_qc in zip(h_rnn, h_qc_rnn):
            self.assertTrue(torch.allclose(h, h_qc, atol=1e-05),
                            msg="h/c mismatched, Failed TestCase:{}".format(tc.test_name))
        self.assertTrue(torch.allclose(o_rnn, o_qc_rnn, atol=1e-05),
                        msg="output mismatched, Failed TestCase:{}".format(tc.test_name))

    def validate_backward_pass(self, tc: TestCase):
        original_model = copy.deepcopy(tc.model)
        quant_op = QcQuantizeRecurrent(module_to_quantize=tc.model, weight_bw=8, activation_bw=8, is_symmetric=False,
                                       quant_scheme=QuantScheme.post_training_tf_enhanced, round_mode='nearest',
                                       data_type=QuantizationDataType.int)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 3
        encodings.min = -2
        encodings.delta = 1
        encodings.offset = 0.2
        for input_quantizer in quant_op.input_quantizers.values():
            input_quantizer.enabled = True
            input_quantizer.encoding = encodings
        for name, param in quant_op.named_parameters(recurse=False):
            quant_op.param_quantizers[name].enabled = True
            quant_op.param_quantizers[name].encoding = encodings
        for output_quantizer in quant_op.output_quantizers.values():
            output_quantizer.encoding = encodings
        # Checking if param are matched
        for name, param in original_model.named_parameters():
            self.assertTrue(torch.allclose(param.data, getattr(quant_op, name).data, atol=1e-05))
        inp = torch.rand(tc.input_shape, requires_grad=True).to(tc.device)
        o_qc_rnn, _ = quant_op(inp, hx=None)
        # Checking if param are matched
        for name, param in original_model.named_parameters():
            self.assertTrue(torch.allclose(param.data, getattr(quant_op, name).data, atol=1e-05))
        optimizer = torch.optim.SGD(quant_op.parameters(), lr=0.05, momentum=0.5)
        # creating a fake loss function with sum of output
        loss = o_qc_rnn.flatten().sum()
        loss.backward()
        for name, param in quant_op.module_to_quantize.named_parameters():
            self.assertTrue(param.grad is None)
            quant_param = getattr(quant_op, name)
            self.assertTrue(quant_param.grad is not None)
            self.assertTrue(torch.allclose(param.data, quant_param.data))
        optimizer.step()
        for name, param in original_model.named_parameters():
            # check if custom param have been updated
            quant_param = getattr(quant_op, name)
            self.assertFalse(torch.allclose(param.data, quant_param.data, atol=1e-05))

            # check if 'replaced' module param are still the same
            module_to_quantize_param = getattr(quant_op.module_to_quantize, name)
            self.assertTrue(torch.allclose(param.data, module_to_quantize_param.data, atol=1e-05))
        # updated the 'replaced' module and check for the reverse
        quant_op.update_params()
        for name, param in quant_op.module_to_quantize.named_parameters():
            # check if 'replaced' module param have been updated
            orig_param = getattr(original_model, name)
            self.assertFalse(torch.allclose(param.data, orig_param.data, atol=1e-05))

            # check if 'replaced' module param are matching the Custom Op
            quant_param = getattr(quant_op, name)
            self.assertTrue(torch.allclose(param.data, quant_param.data, atol=1e-05))

    def compare_quantizer(self, quantizer, loaded_quantizer):
        """
        helper function to compare two quantizers
        """
        self.assertEqual(quantizer.round_mode, loaded_quantizer.round_mode)
        self.assertEqual(quantizer.quant_scheme, loaded_quantizer.quant_scheme)
        self.assertEqual(quantizer.bitwidth, loaded_quantizer.bitwidth)
        self.assertEqual(quantizer.encoding.max, loaded_quantizer.encoding.max)
        self.assertEqual(quantizer.encoding.min, loaded_quantizer.encoding.min)
        self.assertEqual(quantizer.encoding.delta, loaded_quantizer.encoding.delta)
        self.assertEqual(quantizer.encoding.offset, loaded_quantizer.encoding.offset)

    def validate_serialize_deserialize(self, tc: TestCase):
        """
       helper method to run quant RNN test
        """
        original_model = copy.deepcopy(tc.model)
        quant_op = QcQuantizeRecurrent(module_to_quantize=tc.model, weight_bw=8, activation_bw=8, is_symmetric=False,
                                       quant_scheme=QuantScheme.post_training_tf_enhanced, round_mode='nearest',
                                       data_type=QuantizationDataType.int)
        quant_op.eval()
        inp = torch.rand(tc.input_shape, requires_grad=True).to(tc.device)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 3
        encodings.min = -2
        encodings.delta = 1
        encodings.offset = 0.2
        for input_quantizer in quant_op.input_quantizers.values():
            input_quantizer.enabled = True
            input_quantizer.encoding = encodings
        for name, param in quant_op.named_parameters(recurse=False):
            quant_op.param_quantizers[name].enabled = True
            quant_op.param_quantizers[name].encoding = encodings
        for output_quantizer in quant_op.output_quantizers.values():
            output_quantizer.encoding = encodings
        o_qc_rnn, _ = quant_op(inp, hx=None)
        optimizer = torch.optim.SGD(quant_op.parameters(), lr=0.05, momentum=0.5)
        # creating a fake loss function with sum of output
        loss = o_qc_rnn.flatten().sum()
        loss.backward()
        optimizer.step()
        # Generate Quantize encodings
        quant_op.compute_encoding()
        quant_op.compute_weight_encodings()
        o_pre, h_pre = quant_op(inp, hx=None)
        # Save and loaded a quantized model
        with tempfile.NamedTemporaryFile() as f:
            torch.save(quant_op, f)
            f.seek(0)
            loaded_model = torch.load(f)
            loaded_model.eval()
        # compare the parameters
        for name, param in quant_op.named_parameters(recurse=False):
            loaded_param = getattr(loaded_model, name)
            self.assertTrue(torch.equal(param, loaded_param),
                            msg="param mismatched recurrent op param mis-matched, TestCase:{}".format(tc.test_name))
        for name, param in quant_op.module_to_quantize.named_parameters():
            loaded_param = getattr(loaded_model.module_to_quantize, name)
            self.assertTrue(torch.equal(param, loaded_param),
                            msg="original module mismatched, TestCase:{}".format(tc.test_name))
        # compare the quantizers
        for name, output_quantizer in quant_op.output_quantizers.items():
            if output_quantizer.enabled:
                self.compare_quantizer(output_quantizer, loaded_model.output_quantizers[name])
        for name, quantizer in quant_op.param_quantizers.items():
            if quantizer.enabled:
                self.compare_quantizer(quantizer, loaded_model.param_quantizers[name])
        # check if the loaded module generates the same output
        o_post, h_post = loaded_model(inp, hx=None)
        self.assertTrue(torch.equal(o_pre, o_post),
                        msg="output mismatched, Failed TestCase:{}".format(tc.test_name))
        if isinstance(h_pre, tuple):
            for pre, post in zip(h_pre, h_post):
                self.assertTrue(torch.equal(pre, post),
                                msg="h or c mismatched, Failed TestCase:{}".format(tc.test_name))
        else:
            self.assertTrue(torch.equal(h_pre, h_post),
                            msg="h mis-matched, Failed TestCase:{}".format(tc.test_name))

    def verify_packed_sequence_inputs(self, tc: TestCase):
        """
       helper method to run quant RNN test
        """
        quant_op = QcQuantizeRecurrent(module_to_quantize=tc.model, weight_bw=32, activation_bw=32, is_symmetric=False,
                                       quant_scheme=QuantScheme.post_training_tf_enhanced, round_mode='nearest',
                                       data_type=QuantizationDataType.int)

        for input_quantizer in quant_op.input_quantizers.values():
            input_quantizer.enabled = False
        for name, param in quant_op.named_parameters(recurse=False):
            quant_op.param_quantizers[name].enabled = False

        x = torch.rand(tc.input_shape).to(tc.device)
        h = None
        seq_len_for_packing = []
        if tc.sequence_lens is None:
            if tc.model.batch_first:
                num_batches = tc.input_shape[0]
                seq_len = tc.input_shape[1]
            else:
                num_batches = tc.input_shape[1]
                seq_len = tc.input_shape[0]
            for i in range(num_batches):
                seq_len_for_packing.append(max(1, seq_len - i))
        else:
            seq_len_for_packing = tc.sequence_lens

        x = pack_padded_sequence(x, seq_len_for_packing, batch_first=tc.model.batch_first, enforce_sorted=False)

        if tc.valid_hx:
            o_rnn, h_rnn = tc.model(input=x, hx=None)
            if isinstance(h_rnn, tuple):
                h = (h_rnn[0], h_rnn[1])
            else:
                h = torch.stack([h_rnn])
                h = h[0]
        o_rnn, h_rnn = tc.model(input=x, hx=h)
        o_qc_rnn, h_qc_rnn = quant_op(x, hx=h)

        if not isinstance(h_qc_rnn, tuple):
            h_qc_rnn = [h_qc_rnn]
            h_rnn = [h_rnn]
        for h, h_qc in zip(h_rnn, h_qc_rnn):
            self.assertTrue(torch.allclose(h, h_qc, atol=1e-05),
                            msg="h/c mismatched, Failed TestCase:{}".format(tc.test_name))
        self.assertTrue(torch.allclose(o_rnn.data, o_qc_rnn.data, atol=1e-05),
                        msg="output data mismatched, Failed TestCase:{}".format(tc.test_name))
        self.assertTrue(torch.equal(o_rnn.batch_sizes, o_qc_rnn.batch_sizes),
                        msg="output batch_sizes mismatched, Failed TestCase:{}".format(tc.test_name))
        if o_rnn.unsorted_indices is not None:
            self.assertTrue(torch.equal(o_rnn.unsorted_indices, o_qc_rnn.unsorted_indices),
                            msg="output unsorted_indices mismatched, Failed TestCase:{}".format(tc.test_name))
        if o_rnn.sorted_indices is not None:
            self.assertTrue(torch.equal(o_rnn.sorted_indices, o_qc_rnn.sorted_indices),
                            msg="output sorted_indices mismatched, Failed TestCase:{}".format(tc.test_name))

    def test_qc_rnn_equivalence(self):
        """
        Unit test to validate Quantize Recurrent Op equivalence with single layer RNN with no bias params
        """

        for tc in TestQcQuantizeRecurrentOp.testcases:
            self.verify_custom_op(tc)

    def test_qc_recurrent_backward(self):
        """
        Unit test to validate Quantize Recurrent Op backward pass
        """
        torch.manual_seed(0)
        for tc in TestQcQuantizeRecurrentOp.testcases:
            self.validate_backward_pass(tc)

    def test_save_and_load_rnn(self):
        """
        Unit test to validate Recurrent Op serialize/des-serialize functionality
        """
        torch.manual_seed(0)
        for tc in TestQcQuantizeRecurrentOp.testcases:
            self.validate_serialize_deserialize(tc)

    def test_qc_rnn_default_lstm_quantizer_configuration(self):
        """
        Unit test to validate Quantize Recurrent Op default(eAI) configuration for LSTM
        """
        model = torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=2, bidirectional=True, bias=True)

        quant_op = QcQuantizeRecurrent(module_to_quantize=model, weight_bw=8, activation_bw=8, is_symmetric=False,
                                       quant_scheme=QuantScheme.post_training_tf_enhanced, round_mode='nearest',
                                       data_type=QuantizationDataType.int)

        self.assertEqual(8, quant_op.input_quantizers['input_l0'].bitwidth)
        self.assertEqual(8, quant_op.input_quantizers['initial_h_l0'].bitwidth)
        self.assertFalse(quant_op.input_quantizers['initial_c_l0'].enabled)
        self.assertEqual(8, quant_op.input_quantizers['input_l1'].bitwidth)
        self.assertEqual(8, quant_op.input_quantizers['initial_h_l1'].bitwidth)
        self.assertFalse(quant_op.input_quantizers['initial_c_l1'].enabled)

        self.assertEqual(8, quant_op.output_quantizers['h_l0'].bitwidth)
        self.assertFalse(quant_op.output_quantizers['c_l0'].enabled)
        self.assertEqual(8, quant_op.output_quantizers['h_l1'].bitwidth)
        self.assertFalse(quant_op.output_quantizers['c_l1'].enabled)

        for name, quantizer in quant_op.param_quantizers.items():
            if 'weight' in name:
                self.assertEqual(8, quantizer.bitwidth, msg="weight quantizer={}".format(name))
            else:
                self.assertFalse(quantizer.enabled)

        group_cfg = {
            # layer 0
            'hidden_l0': ['initial_h_l0', 'h_l0'],
            'bias_l0': ['bias_ih_l0', 'bias_hh_l0', 'bias_ih_l0_reverse', 'bias_hh_l0_reverse'],
            'W_l0':  ['weight_ih_l0', 'weight_ih_l0_reverse'],
            'R_l0': ['weight_hh_l0', 'weight_hh_l0_reverse'],

            # layer 1
            'hidden_l1': ['initial_h_l1', 'h_l1'],
            'bias_l1': ['bias_ih_l1', 'bias_hh_l1', 'bias_ih_l1_reverse', 'bias_hh_l1_reverse'],
            'W_l1':  ['weight_ih_l1', 'weight_ih_l1_reverse'],
            'R_l1': ['weight_hh_l1', 'weight_hh_l1_reverse']}

        for group_name, tensor_names in group_cfg.items():
            group_quantizer = quant_op.grouped_quantizers[group_name]
            if 'bias' not in group_name:
                self.assertTrue(group_quantizer.enabled)
            else:
                self.assertFalse(group_quantizer.enabled)
            for tensor_name in tensor_names:
                if tensor_name in quant_op.input_quantizers:
                    self.assertEqual(group_quantizer, quant_op.input_quantizers[tensor_name])
                elif tensor_name in quant_op.output_quantizers:
                    self.assertEqual(group_quantizer, quant_op.output_quantizers[tensor_name])
                else:
                    self.assertIn(tensor_name, quant_op.param_quantizers)
                    self.assertEqual(group_quantizer, quant_op.param_quantizers[tensor_name])

    def test_packed_sequence_inputs_equivalence(self):
        """
        Unit test to validate Quantize Recurrent Op equivalence with packed sequence inputs
        """

        for tc in TestQcQuantizeRecurrentOp.testcases:
            self.verify_packed_sequence_inputs(tc)

class GruModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = torch.nn.GRU(input_size=4, hidden_size=4, num_layers=1, bias=True)
        self.gru2 = torch.nn.GRU(input_size=4, hidden_size=4, num_layers=1, bias=True)

    def forward(self, *x):
        output, h = self.gru1(*x)
        return self.gru2(output, h)

@pytest.mark.parametrize('learned_grid', [False, True])
def test_qc_rnn_learned_grid_mode(tmp_path, learned_grid):
    """
    Unit test to validate Quantize Recurrent Op default(eAI) configuration for LSTM
    """
    torch.manual_seed(0)
    model = GruModel()

    input_shape = (4, 3, 4)
    dummy_input = (torch.rand(input_shape, requires_grad=True).to('cpu'),
                   torch.rand((1,3,4), requires_grad=True).to('cpu'))
    quant_scheme = QuantScheme.training_range_learning_with_tf_enhanced_init if learned_grid \
        else QuantScheme.post_training_tf_enhanced


    sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=quant_scheme)
    sim.model.train()
    for module in [ sim.model.gru1, sim.model.gru2]:
        for input_quantizer in module.input_quantizers.values():
            input_quantizer.enabled = True
        for name, param in module.named_parameters(recurse=False):
            module.param_quantizers[name].enabled = True
        for output_quantizer in module.output_quantizers.values():
            output_quantizer.enabled = True

    def forward_pass(model, args):
        model.eval()
        with torch.no_grad():
            output = model(*dummy_input)
        return output

    print(sim)
    # Generate Quantize encodings
    sim.compute_encodings(forward_pass, None)
    print(sim)
    if learned_grid:
        assert sim.model.gru1._mode == QcQuantizeOpMode.LEARN_ENCODINGS
        assert sim.model.gru2._mode == QcQuantizeOpMode.LEARN_ENCODINGS
        assert all(isinstance(q, LearnedGridTensorQuantizer)
                   for module in [sim.model.gru1, sim.model.gru2]
                   for quantizers in [ module.input_quantizers.values(),
                                       module.param_quantizers.values(),
                                       module.output_quantizers.values()]
                   for q in quantizers)

    # for name, _ in sim.model.gru.input_quantizers(): check for _encoding min/max params

    params = {f'{i}.{name}': param.clone().detach() for i, module in enumerate([sim.model.gru1, sim.model.gru2])
              for name, param in module.named_parameters(recurse=False)}

    optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.05, momentum=0.5)

    # train the model w/ few random inputs.
    for i in range(10):
        dummy_input = (torch.rand(input_shape, requires_grad=True).to('cpu'),
                       torch.rand((1, 3, 4), requires_grad=True).to('cpu'))
        o_qc_rnn, h_qc_rnn = sim.model(*dummy_input)
        # creating a fake loss function with sum of output
        loss = o_qc_rnn.flatten().sum() + h_qc_rnn.flatten().sum()
        loss.backward()
        optimizer.step()
    learned_params = {f'{i}.{name}': param.clone().detach() for i, module in enumerate([sim.model.gru1, sim.model.gru2])
                      for name, param in module.named_parameters(recurse=False)}

    for name in params:
        assert not torch.equal(params[name], learned_params[name])

    sim.export(tmp_path, "gru_learned", dummy_input)
    onnx_model = onnx.load(os.path.join(tmp_path,'gru_learned.onnx'))
    for node in onnx_model.graph.node:
        if node.op_type == 'GRU':
            with open(os.path.join(tmp_path, "gru_learned.encodings"), "r") as encodings_file:
                encodings = json.load(encodings_file)
                encoding_tensors = set([*encodings['activation_encodings'].keys(), *encodings['param_encodings']])
                assert set([*node.input, *node.output]) - encoding_tensors  == {''} # ignore the sequence len tensor


