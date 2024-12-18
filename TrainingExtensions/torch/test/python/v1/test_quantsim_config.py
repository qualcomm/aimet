# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Module for testing quantsim config feature """
import tempfile

import pytest
import json
import os
import torch
import numpy as np
from aimet_common.connected_graph.connectedgraph_utils import CG_SPLIT
from aimet_common.defs import QuantScheme, QuantizationDataType, QuantDtypeBwInfo, SupportedKernelsAction
from ..models.test_models import SingleResidual, QuantSimTinyModel, MultiInput, SingleResidualWithModuleAdd, \
    SingleResidualWithAvgPool
from aimet_torch.v1.quantsim import QuantizationSimModel
import aimet_torch._base.quantsim
from aimet_torch.quantsim_config import quantsim_config as qsim_config
from aimet_torch.quantsim_config.quantsim_config import get_all_ops_in_neighborhood
from aimet_torch.v1.qc_quantize_op import QcQuantizeWrapper
from aimet_torch import utils
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.v1.tensor_quantizer import StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer
from aimet_torch._base.nn.modules.custom import Add
from ..models import test_models

class ModelWithBertCustomLayerNormGelu(torch.nn.Module):
    """ Model with PyTorch LayerNorm and gelu """

    def __init__(self):
        super(ModelWithBertCustomLayerNormGelu, self).__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        # default attribute -
        # eps = 1e-05 and elementwise_affine = True
        # parameters : weight and bias
        self.customln1 = torch.nn.LayerNorm(4)
        self.gelu1 = torch.nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.customln1(x)
        x = self.gelu1(x)
        return x


# pylint: disable=protected-access
class TestQuantsimConfig:
    """ Class containing unit tests for quantsim config feature """

    def test_parse_config_file_defaults(self):
        """ Test that default quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                },
                "per_channel_quantization": "True",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32), in_place=True)
            for name, module in sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    # Output of add op is input quantized
                    if name == 'relu3':
                        assert module.input_quantizers[0].enabled
                    else:
                        assert not module.input_quantizers[0].enabled
                    if name in ["conv1", "conv2"]:
                        # Output quantizers of conv1 and conv2 are
                        # disabled due to the subsequent batchnorm
                        assert not module.output_quantizers[0].enabled
                    else:
                        assert module.output_quantizers[0].enabled
                    assert not module.input_quantizers[0].use_symmetric_encodings
                    assert not module.input_quantizers[0].use_strict_symmetric
                    assert not module.input_quantizers[0].use_unsigned_symmetric
                    assert not module.output_quantizers[0].use_symmetric_encodings
                    assert not module.output_quantizers[0].use_strict_symmetric
                    assert not module.output_quantizers[0].use_unsigned_symmetric


                    for _, param_quantizer in module.param_quantizers.items():
                        assert not param_quantizer.enabled
                        assert param_quantizer.use_symmetric_encodings
                        assert not param_quantizer.use_strict_symmetric
                        assert not param_quantizer.use_unsigned_symmetric
                        assert len(param_quantizer._cppOp) > 1

    def test_parse_config_file_params(self):
        """ Test that param specific quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                }
            },
            "params": {
                "weight": {
                    "is_quantized": "True",
                    "is_symmetric": "False"
                }
            },
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32))
            for _, module in sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    for param_name, param_quantizer in module.param_quantizers.items():
                        if param_name == 'weight':
                            if module in (sim.model.bn1, sim.model.bn2):
                                assert not param_quantizer.enabled
                            else:
                                assert param_quantizer.enabled
                            assert not param_quantizer.use_symmetric_encodings
                        else:
                            assert not param_quantizer.enabled
                            assert param_quantizer.use_symmetric_encodings

    def test_parse_config_file_default_supported_kernels(self):
        """
        Test that the supported_kernels in the defaults section is parsed correctly and its values are added
        in the dict _supported_kernels
        """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                },
                "supported_kernels":[
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 8,
                            "dtype": "int"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "float"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "float"
                        }
                    }
                ]
            },
            "params": {
                "weight": {
                    "is_quantized": "True",
                    "is_symmetric": "False"
                }
            },
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

        expected_supported_kernels = [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": QuantizationDataType.int
                        },
                        "param": {
                            "bitwidth": 8,
                            "dtype": QuantizationDataType.int
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": QuantizationDataType.float
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": QuantizationDataType.float
                        }
                    }
                ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32))

            supported_kernels_in_defaults = sim.get_supported_kernels()["defaults"]
            assert len(supported_kernels_in_defaults) == 2
            assert supported_kernels_in_defaults == expected_supported_kernels

    def test_parse_config_file_op_type(self):
        """ Test that op specific quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                }
            },
            "params": {},
            "op_type": {
                "Conv": {
                    "is_input_quantized": "True",
                    "is_symmetric": "False",
                    "params": {
                        "bias": {
                            "is_quantized": "True",
                            "is_symmetric": "False"
                        }
                    },
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32))
            for name, module in sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    if isinstance(module._module_to_wrap, torch.nn.Conv2d):
                        assert module.input_quantizers[0].enabled
                        if name in ["conv1", "conv2"]:
                            assert not module.output_quantizers[0].enabled
                        else:
                            assert module.output_quantizers[0].enabled
                    else:
                        # Output of add op is input quantized
                        if name == 'relu3':
                            assert module.input_quantizers[0].enabled
                        else:
                            assert not module.input_quantizers[0].enabled
                        assert module.output_quantizers[0].enabled

                    assert not module.input_quantizers[0].use_symmetric_encodings
                    assert not module.output_quantizers[0].use_symmetric_encodings

                    for param_name, param_quantizer in module.param_quantizers.items():
                        if isinstance(module._module_to_wrap, torch.nn.Conv2d) and param_name == 'bias':
                            assert param_quantizer.enabled
                            assert not param_quantizer.use_symmetric_encodings
                        else:
                            assert not param_quantizer.enabled
                            assert param_quantizer.use_symmetric_encodings

    def _test_parse_config_file_op_type_per_channel_helper(self, per_channel_fields):
        """ helper function to test out per_channel_quantization"""
        for k in ['defaults', 'Conv', 'Gemm']:
            assert k in per_channel_fields.keys()

        model = MultiInput()
        model.eval()
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                },
                "per_channel_quantization": per_channel_fields["defaults"],
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                },
            },
            "op_type": {
                "Conv": {
                    "per_channel_quantization": per_channel_fields["Conv"]
                },
                "Gemm": {
                    "per_channel_quantization": per_channel_fields["Gemm"]
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20)))

            return sim

    def test_parse_config_file_op_type_per_channel(self):
        """ Test that op specific quantization parameters are set correctly when using json config file """

        # test 1: expect all to be StaticGridPerChannelQuantizer
        per_channel_fields = {"defaults": "True", "Conv": "True", "Gemm": "True"}
        sim = self._test_parse_config_file_op_type_per_channel_helper(per_channel_fields)

        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if 'weight' in module.param_quantizers:
                    assert isinstance(module.param_quantizers['weight'], StaticGridPerChannelQuantizer)


        # test 2: expect all but Conv to be StaticGridPerChannelQuantizer
        per_channel_fields = {"defaults": "True", "Conv": "False", "Gemm": "True"}
        sim = self._test_parse_config_file_op_type_per_channel_helper(per_channel_fields)

        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if 'weight' in module.param_quantizers:
                    if isinstance(module._module_to_wrap, torch.nn.Conv2d):
                        assert isinstance(module.param_quantizers['weight'], StaticGridPerTensorQuantizer)
                    else:
                        assert isinstance(module.param_quantizers['weight'], StaticGridPerChannelQuantizer)


        # test 3: expect all but Conv and Gemm to be StaticGridPerChannelQuantizer
        per_channel_fields = {"defaults": "True", "Conv": "False", "Gemm": "False"}
        sim = self._test_parse_config_file_op_type_per_channel_helper(per_channel_fields)

        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if 'weight' in module.param_quantizers:
                    if isinstance(module._module_to_wrap, torch.nn.Conv2d) or isinstance(module._module_to_wrap,
                                                                                         torch.nn.Linear):
                        assert isinstance(module.param_quantizers['weight'], StaticGridPerTensorQuantizer)
                    else:
                        assert isinstance(module.param_quantizers['weight'], StaticGridPerChannelQuantizer)


        # test 4: expect all in StaticGridPerTensorQuantizer except Conv which will be in StaticGridPerChannelQuantizer
        per_channel_fields = {"defaults": "False", "Conv": "True", "Gemm": "False"}
        sim = self._test_parse_config_file_op_type_per_channel_helper(per_channel_fields)

        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if 'weight' in module.param_quantizers:
                    if isinstance(module._module_to_wrap, torch.nn.Conv2d):
                        assert isinstance(module.param_quantizers['weight'], StaticGridPerChannelQuantizer)
                    else:
                        assert isinstance(module.param_quantizers['weight'], StaticGridPerTensorQuantizer)

        random_input = (torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*random_input)


        # test 5: test to make sure only Conv has param encodings of length greater than 1(per-channel),
        # others have only one entry
        sim.compute_encodings(forward_pass, None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(tmp_dir, 'test_parse_config_file_op_type_per_channel',
                       dummy_input=(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20)))

            with open(os.path.join(tmp_dir, "test_parse_config_file_op_type_per_channel.encodings"), "r") as encodings_file:
                encodings = json.load(encodings_file)
                assert len(encodings["param_encodings"]["bn1.weight"]) == 1
                assert len(encodings["param_encodings"]["fc.weight"]) == 1
                assert len(encodings["param_encodings"]["conv1.weight"]) == 16
                assert len(encodings["param_encodings"]["conv2.weight"]) == 8
                assert len(encodings["param_encodings"]["conv3.weight"]) == 8

    def test_hw_version(self):
        """
        test the harwdware version option
        """
        model = SingleResidual()
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                },
                "hw_version": "V01"
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = os.path.join(tmp_dir, 'quantsim_config.json')
            with open(config_file, 'w') as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=config_file,
                                       dummy_input=torch.rand(1, 3, 32, 32))

            version = sim.configure_quantization_ops(config_file, 8, 8, QuantizationDataType.int).\
                          _get_hw_version()
            assert version == "V01"

            quantsim_config = {
                "defaults": {
                    "ops": {
                        "is_output_quantized": "True",
                        "is_symmetric": "False"
                    },
                    "params": {
                        "is_quantized": "False",
                        "is_symmetric": "True"
                    },
                },
                "params": {},
                "op_type": {},
                "supergroups": [],
                "model_input": {},
                "model_output": {}
            }

            with open(config_file, 'w') as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=config_file,
                                       dummy_input=torch.rand(1, 3, 32, 32))

            version = sim.configure_quantization_ops(config_file, 8, 8, QuantizationDataType.int).\
                          _get_hw_version()
            assert version == "default"

    def test_op_instance_config_1(self):
        """
        Tests the generated supported_kernels and pcq fields for all the ops
        """
        for model in [SingleResidual(), SingleResidualWithAvgPool(), SingleResidualWithModuleAdd()]:
            quantsim_config = {
                "defaults": {
                    "ops": {
                        "is_output_quantized": "True",
                        "is_symmetric": "False"
                    },
                    "params": {
                        "is_quantized": "False",
                        "is_symmetric": "True"
                    },
                    "hw_version": "V01",
                    "supported_kernels": [
                        {
                            "activation": {
                                "bitwidth": 16,
                                "dtype": "float"
                            },
                            "param": {
                                "bitwidth": 16,
                                "dtype": "float"
                            }
                        }
                    ],
                    "per_channel_quantization": "True",
                },
                "params": {},
                "op_type": {
                    'Conv':{
                        "supported_kernels": [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "int"
                                },
                                "param": {
                                    "bitwidth": 8,
                                    "dtype": "int"
                                }
                            }
                        ],
                        "per_channel_quantization": "False",
                    }
                },
                "supergroups": [],
                "model_input": {},
                "model_output": {}
            }

            with tempfile.TemporaryDirectory() as tmp_dir:
                with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                    json.dump(quantsim_config, f)
                sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                           config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                           dummy_input=torch.rand(1, 3, 32, 32))
                for _, module in sim.model.named_modules():
                    if isinstance(module, QcQuantizeWrapper):
                        assert len(module.supported_kernels) == 1
                        if module._module_to_wrap._get_name() == 'Conv2d':
                            assert isinstance(module.param_quantizers['weight'], StaticGridPerTensorQuantizer)
                            assert module.supported_kernels == [((16, QuantizationDataType.int), (8, QuantizationDataType.int))]
                        else:
                            if module.param_quantizers:
                                assert isinstance(module.param_quantizers['weight'], StaticGridPerChannelQuantizer)
                            if module.supported_kernels:
                                assert module.supported_kernels == [((16, QuantizationDataType.float), (16, QuantizationDataType.float))]
                del sim

    def test_parse_config_file_op_type_supported_kernels(self):
        """
        Test that the supported_kernels in the op_type section is parsed correctly and its values are added
        in the dict _supported_kernels
        """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "weight": {
                    "is_quantized": "True",
                    "is_symmetric": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "supported_kernels": [
                        {
                            "activation": {
                                "bitwidth": 16,
                                "dtype": "int"
                            },
                            "param": {
                                "bitwidth": 8,
                                "dtype": "int"
                            }
                        }
                    ]
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

        expected_supported_kernels = [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": QuantizationDataType.int
                        },
                        "param": {
                            "bitwidth": 8,
                            "dtype": QuantizationDataType.int
                        }
                    }
                ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32))

            supported_kernels_in_defaults = sim.get_supported_kernels()["Conv"]
            assert len(supported_kernels_in_defaults) == 1
            assert supported_kernels_in_defaults == expected_supported_kernels

    def test_parse_config_file_supported_kernels_1(self):
        """
        Only the default section has supported_kernels, make sure all the wrappers have the same default supported_kernels
        """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "weight": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                }
            },
            "op_type": {
                "Conv": {
                    "supported_kernels": [
                        {
                            "activation": {
                                "bitwidth": 16,
                                "dtype": "int"
                            },
                            "param": {
                                "bitwidth": 8,
                                "dtype": "int"
                            }
                        }
                    ]
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32), default_param_bw=16, default_output_bw=16,
                                       default_data_type=QuantizationDataType.int)

            for _, module in sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    if module._module_to_wrap._get_name() == 'Conv2d':
                        assert module.supported_kernels == [((16, QuantizationDataType.int), (8, QuantizationDataType.int))]
                    else:
                        assert module.supported_kernels == [((16, QuantizationDataType.int),(16, QuantizationDataType.int))]


    def test_parse_config_file_supported_kernels_2(self):
        """
        Check if error is raised with incorrect config
        """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "weight": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                }
            },
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            try:
                aimet_torch._base.quantsim.SUPPORTED_KERNELS_ACTION = SupportedKernelsAction.assert_on_error
                with pytest.raises(RuntimeError):
                    QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                           config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                           dummy_input=torch.rand(1, 3, 32, 32), default_param_bw=16, default_output_bw=8,
                                           default_data_type=QuantizationDataType.int)
            finally:
                aimet_torch._base.quantsim.SUPPORTED_KERNELS_ACTION = SupportedKernelsAction.warn_on_error

    def test_parse_config_file_supergroups(self):
        """ Test that supergroup quantization parameters are set correctly when using json config file """
        model = QuantSimTinyModel()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "False"
                }
            },
            "params": {},
            "op_type": {},
            "supergroups": [
                {
                    "op_list": ["Conv", "Relu"]
                },
                {
                    "op_list": ["Relu", "MaxPool"]
                },
                {
                    "op_list": ["Conv", "Relu", "AveragePool"]
                },
                {
                    "op_list": ["Conv", "Clip"]
                },
            ],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)
            # Use in_place=True here for easy access to modules through model instance variables
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       in_place=True, dummy_input=torch.rand(1, 3, 32, 32))

            # Expected supergroups: (square bracket indicates a supergroup)
            # in -> [conv1->bn1->relu1->maxpool] -> [conv2->bn2->relu2] -> [conv3->relu3->avgpool] -> [conv4] -> [fc] -> out

            for _, module in sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    # All input quantizers are disabled by config
                    assert not module.input_quantizers[0].enabled

            # First supergroup
            assert not sim.model.conv1.output_quantizers[0].enabled
            assert not sim.model.bn1.output_quantizers[0].enabled
            assert not sim.model.relu1.output_quantizers[0].enabled
            assert sim.model.maxpool.output_quantizers[0].enabled

            # Second supergroup
            assert not sim.model.conv2.output_quantizers[0].enabled
            assert not sim.model.bn2.output_quantizers[0].enabled
            assert sim.model.relu2.output_quantizers[0].enabled

            # Third supergroup
            assert not model.conv3.output_quantizers[0].enabled
            assert not sim.model.relu3.output_quantizers[0].enabled
            assert sim.model.avgpool.output_quantizers[0].enabled

            # Supergroups with only one operation
            assert model.conv4.output_quantizers[0].enabled
            assert model.fc.output_quantizers[0].enabled

    def test_parse_config_file_elementwise_ops(self):
        """ Test that elementwise op quantizers are set as expected """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {}
            },
            "params": {},
            "op_type": {
                "Add": {
                    "is_input_quantized": "True"
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32))
            for name, module in sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    if name in ['conv3', 'ada']:
                        # model.conv3 and model.ada are inputs to add
                        assert module.output_quantizers[0].enabled
                    else:
                        assert not module.output_quantizers[0].enabled
                    assert not module.input_quantizers[0].enabled

    def test_parse_config_file_model_inputs(self):
        """ Test that model input quantization parameters are set correctly when using json config file """

        class MultiInputWithConstant(torch.nn.Module):
            """ A model with multiple inputs.
                Use this model for unit testing purposes. """

            def __init__(self, num_classes=3):
                super(MultiInputWithConstant, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=3, bias=False)
                self.conv2 = torch.nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=2)
                self.conv3 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=2)
                self.add1 = Add()
                self.add2 = Add()

            def forward(self, *inputs):
                x1 = self.conv1(inputs[0])
                x1 = self.conv2(x1)
                x2 = self.conv3(inputs[1])
                x = self.add1(x1, x2)
                x = self.add2(x, torch.tensor(2.0))
                return x

        model = MultiInputWithConstant()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20)), in_place=True)
            for name, module in sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    if name in ('conv1', 'conv3'):
                        assert module.input_quantizers[0].enabled
                    elif name == 'add2':
                        assert not module.input_quantizers[0].enabled
                        assert module.input_quantizers[1].enabled
                        assert module.input_quantizers[1].is_const
                    else:
                        assert not module.input_quantizers[0].enabled
                    assert not module.output_quantizers[0].enabled
                    assert not module.input_quantizers[0].use_symmetric_encodings
                    assert not module.output_quantizers[0].use_symmetric_encodings

    def test_parse_config_file_model_outputs(self):
        """ Test that model output quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32))
            for name, module in sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    if name == 'fc':
                        # model.conv3 and model.ada are inputs to add
                        assert module.output_quantizers[0].enabled
                    else:
                        assert not module.output_quantizers[0].enabled
                    assert not module.input_quantizers[0].enabled

    def test_supergroups_with_functional_add(self):
        """ Test supergroup with functional add """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [
                {
                    "op_list": ["Add", "Relu"]
                }
            ],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)
            # Use in_place=True here for easy access to modules through model instance variables
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       in_place=True, dummy_input=torch.rand(1, 3, 32, 32))

            # Expected supergroups: (square bracket indicates a supergroup)
            # in -> [conv1] -> [bn1]-> ... -> [conv3] -----> [(+)->relu3->avgpool] -> [fc] -> out
            #                           |                      ^
            #                           +--> [conv4] -> [ada] -+

            # If add were not part of the supergroup, relu's input quantizer would be enabled
            assert not sim.model.relu3.input_quantizers[0].enabled

    def test_supergroups_with_module_add(self):
        """ Test supergroup with add module """
        model = SingleResidualWithModuleAdd()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [
                {
                    "op_list": ["Add", "Relu"]
                }
            ],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)
            # Use in_place=True here for easy access to modules through model instance variables
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       in_place=True, dummy_input=torch.rand(1, 3, 32, 32))
            for _, module in sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    # Check configs for starts of supergroups
                    if module in [model.add, model.conv1, model.conv2]:
                        # If add were not part of the supergroup, relu's input quantizer would be enabled
                        assert not module.output_quantizers[0].enabled
                    else:
                        assert module.output_quantizers[0].enabled
                    assert not module.input_quantizers[0].enabled

    def test_parse_config_file_symmetric_modes(self):
        """ Test that model output quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults":
            {
                "ops": {},
                "params":
                {
                    "is_symmetric": "True"
                },
                "per_channel_quantization": "True",
                "strict_symmetric": "True",
                "unsigned_symmetric": "False"
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32))
            for name, module in sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    for q in module.input_quantizers:
                        assert q.use_strict_symmetric
                        assert not q.use_unsigned_symmetric
                    for q in module.output_quantizers:
                        assert q.use_strict_symmetric
                        assert not q.use_unsigned_symmetric
                    for q in module.param_quantizers.values():
                        assert q.use_strict_symmetric
                        assert not q.use_unsigned_symmetric

    def test_get_all_ops_in_neighborhood(self):
        """ Test that default quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()
        input_shapes = (1, 3, 32, 32)

        random_inputs = utils.create_rand_tensors_given_shapes(input_shapes, utils.get_device(model))
        conn_graph = ConnectedGraph(model, random_inputs)
        starting_op = conn_graph.get_op_from_module_name('SingleResidual.conv3')
        add_op = [op for op in conn_graph.get_all_ops().values() if op.type == 'Add'][0]
        neighborhood = get_all_ops_in_neighborhood(starting_op, 'output', split_type=CG_SPLIT)
        assert len(neighborhood) == 2
        assert starting_op in neighborhood
        assert add_op in neighborhood

    @pytest.mark.cuda
    def test_parse_config_file_defaults_gpu(self):
        """ Test that default quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()
        model.cuda()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                }
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32).cuda(), in_place=True)
            for name, module in sim.model.named_modules():
                if isinstance(module, QcQuantizeWrapper):
                    # Output of add op is input quantized
                    if name == 'relu3':
                        assert module.input_quantizers[0].enabled
                    else:
                        assert not module.input_quantizers[0].enabled
                    if name in ["conv1", "conv2"]:
                        # Output quantizers of conv1 and conv2 are
                        # disabled due to the subsequent batchnorm
                        assert not module.output_quantizers[0].enabled
                    else:
                        assert module.output_quantizers[0].enabled
                    assert not module.input_quantizers[0].use_symmetric_encodings
                    assert not module.output_quantizers[0].use_symmetric_encodings

                    for _, param_quantizer in module.param_quantizers.items():
                        assert not param_quantizer.enabled
                        assert param_quantizer.use_symmetric_encodings
                        assert len(param_quantizer._cppOp) == 1

    def test_gelu_layernorm_quantsim_config(self):
        """
        Create a network with LayerNorm and GELU
        Override quantization config and check config is applied
        This is a validation for actual entry in the map_torch_types_to_onnx in onnx_utils
        This is used by connected graph to apply op level specific quantsim config.
        :return:
        """

        import json
        import aimet_common.libpymo as libpymo
        from aimet_common.defs import QuantScheme
        from aimet_torch.v1.quantsim import QuantizationSimModel

        class ModelWithGeluLayerNorm(torch.nn.Module):
            def __init__(self):
                super(ModelWithGeluLayerNorm, self).__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                # default attribute -
                # eps = 1e-05 and elementwise_affine = True
                # parameters : weight and bias
                self.ln1 = torch.nn.LayerNorm(4)
                self.gelu1 = torch.nn.GELU()

            def forward(self, x):
                x = self.linear1(x)
                x = self.ln1(x)
                x = self.gelu1(x)
                return x

        # create custom config to override LayerNorm and GELU
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                }
            },
            "params": {},
            "op_type": {
                "LayerNorm": {
                    "is_input_quantized": "True",
                    "params": {
                        "bias": {
                            "is_quantized": "True"
                        },
                    },
                },
                "GELU": {
                    "is_input_quantized": "True",
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            model = ModelWithGeluLayerNorm()
            model.eval()
            random_input = torch.rand(1, 4, 4)

            def forward_pass(model, args):
                model.eval()
                with torch.no_grad():
                    model(*random_input)

            # QuantSim for model
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=random_input)

            sim.compute_encodings(forward_pass, None)

            #  check quantizer added to parameters of LayerNorm
            from aimet_torch.v1.qc_quantize_op import StaticGridPerTensorQuantizer
            assert(isinstance(sim.model.ln1.param_quantizers['weight'], StaticGridPerTensorQuantizer))
            assert(isinstance(sim.model.ln1.param_quantizers['bias'], StaticGridPerTensorQuantizer))


            # LayerNorm input quantization is disabled by default
            # override with custom config file, this needs appropriate entry in onnx node name mapping
            assert(isinstance(sim.model.ln1.input_quantizers[0], StaticGridPerTensorQuantizer))
            assert(sim.model.ln1.input_quantizers[0].encoding)
            in_quantizer = sim.model.ln1.input_quantizers[0]
            assert(in_quantizer.enabled)  # disabled by default, override with config file
            assert(in_quantizer.round_mode == libpymo.RoundingMode.ROUND_NEAREST)
            assert(in_quantizer.quant_scheme == QuantScheme.post_training_tf)
            assert(in_quantizer.bitwidth == 8)


            # GELU input quantization is disabled by default
            # override with custom config file, this needs appropriate entry in onnx node name mapping
            assert(isinstance(sim.model.gelu1.input_quantizers[0], StaticGridPerTensorQuantizer))
            assert(sim.model.gelu1.input_quantizers[0].encoding)
            in_quantizer = sim.model.gelu1.input_quantizers[0]
            assert(in_quantizer.enabled)  # disabled by default, override with config file
            assert(in_quantizer.round_mode == libpymo.RoundingMode.ROUND_NEAREST)
            assert(in_quantizer.quant_scheme == QuantScheme.post_training_tf)
            assert(in_quantizer.bitwidth == 8)

    def test_default_quantsim_config_not_in_default_config_file_enforce_false(self):
        """
        Tests application of override config rule for default bitwidth and dtype for params and act.
        In this test, default supported kernel list (fp 16) in the config file DOES NOT SUPPORT
        default quantsim config (int 8) + ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG is also set to True.
        Tests application of default config rule for op level bitwidth and dtype for params
        :return:
        """
        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "float"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "float"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    }
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            INPUT_SHAPE = (1, 3, 32, 32)
            def forward_fn(model, _):
                torch.manual_seed(10)
                model.eval()
                with torch.no_grad():
                    _ = model(torch.randn(INPUT_SHAPE))

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32), in_place=True,
                                       default_param_bw=8, default_output_bw=8, default_data_type=QuantizationDataType.int)
            sim.compute_encodings(forward_fn, forward_pass_callback_args=None)

            # all quantizers should be quantsim default quantsim dtype and bw  (int 8)
            assert(sim.model.conv1.param_quantizers['weight'].enabled == True)
            assert(sim.model.conv1.param_quantizers['weight'].bitwidth == 8)
            assert(sim.model.conv1.param_quantizers['weight'].data_type == QuantizationDataType.int)

            assert(sim.model.conv1.output_quantizers[0].bitwidth == 8)
            assert(sim.model.conv1.output_quantizers[0].data_type == QuantizationDataType.int)

            # all quantizers should be quantsim default quantsim dtype and bw  (int 8)
            # that is  QUANTSIM DEFAULT bw / dtype (int 8).
            assert(sim.model.fc.param_quantizers['weight'].enabled)
            assert(sim.model.fc.param_quantizers['bias'].enabled == False)
            assert(sim.model.fc.param_quantizers['weight'].bitwidth == 8)
            assert(sim.model.fc.param_quantizers['weight'].data_type == QuantizationDataType.int)
            assert(sim.model.fc.param_quantizers['bias'].bitwidth == 8)
            assert(sim.model.fc.param_quantizers['bias'].data_type == QuantizationDataType.int)
            assert(sim.model.fc.output_quantizers[0].bitwidth == 8)
            assert(sim.model.fc.output_quantizers[0].data_type == QuantizationDataType.int)
            assert(sim.model.relu1.output_quantizers[0].bitwidth == 8)
            assert(sim.model.relu1.output_quantizers[0].data_type == QuantizationDataType.int)

            # remove test config created
            qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False

    def test_default_quantsim_config_in_default_config_file_enforce_true(self):
        """
        Tests application of override config rule for default bitwidth and dtype for params and act.
        In this test, default supported kernel list (int 8, fp 16) in the config file CONTAINS
        default quantsim config (int 8) + ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG is also set to True.
        Tests application of default config rule for op level bitwidth and dtype for params
        :return:
        """

        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True
        model = SingleResidual()
        model.eval()
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "float"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "float"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 8,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 8,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                            {
                                "activation": {
                                    "bitwidth": 8,
                                    "dtype": "int"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "int"
                                }
                            },
                        ],
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    }
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            INPUT_SHAPE = (1, 3, 32, 32)
            def forward_fn(model, _):
                torch.manual_seed(10)
                model.eval()
                with torch.no_grad():
                    _ = model(torch.randn(INPUT_SHAPE))

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32), in_place=True,
                                       default_data_type=QuantizationDataType.int, default_output_bw=8, default_param_bw=8)
            sim.compute_encodings(forward_fn, forward_pass_callback_args=None)

            # enforce is true, however default quantsim bw / dtype (fp16) is not the config file supported kernels override at index 0.
            # apply override 0 # activation : bw = 16, float # param : bw = 16, float
            assert(sim.model.conv1.param_quantizers['weight'].enabled == True)
            assert(sim.model.conv1.param_quantizers['weight'].bitwidth == 16)
            assert(sim.model.conv1.param_quantizers['weight'].data_type == QuantizationDataType.float)

            assert(sim.model.conv1.output_quantizers[0].bitwidth == 16)
            assert(sim.model.conv1.output_quantizers[0].data_type == QuantizationDataType.float)

            assert(sim.model.fc.param_quantizers['weight'].enabled)
            assert(sim.model.fc.param_quantizers['bias'].enabled == False)
            assert(sim.model.fc.param_quantizers['weight'].bitwidth == 16)
            assert(sim.model.fc.param_quantizers['weight'].data_type == QuantizationDataType.float)
            assert(sim.model.fc.param_quantizers['bias'].bitwidth == 16)
            assert(sim.model.fc.param_quantizers['bias'].data_type == QuantizationDataType.float)
            assert(sim.model.fc.output_quantizers[0].bitwidth == 16)
            assert(sim.model.fc.output_quantizers[0].data_type == QuantizationDataType.float)
            assert(sim.model.relu1.output_quantizers[0].bitwidth == 16)
            assert(sim.model.relu1.output_quantizers[0].data_type == QuantizationDataType.float)

            # remove test config created
            qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False

    def test_default_quantsim_config_not_in_default_config_file_enforce_true(self):
        """
        Tests application of override config rule for default bitwidth and dtype for params and act.
        In this test, default supported kernel list (fp 16) in the config file DOES NOT SUPPORT
        default quantsim config (int 8) + ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG is also set to True.
        Tests application of default config rule for op level bitwidth and dtype for params
        :return:
        """

        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True
        model = SingleResidual()
        model.eval()
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "float"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "float"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    }
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            INPUT_SHAPE = (1, 3, 32, 32)
            def forward_fn(model, _):
                torch.manual_seed(10)
                model.eval()
                with torch.no_grad():
                    _ = model(torch.randn(INPUT_SHAPE))

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32), in_place=True,
                                       default_data_type=QuantizationDataType.int, default_output_bw=8, default_param_bw=8)
            sim.compute_encodings(forward_fn, forward_pass_callback_args=None)

            # enforce is true, however default quantsim bw / dtype (int 8) is NOT IN the config file supported kernels
            # should be configured with config file default supported kernel [0]
            # activation : bw = 16 , float
            # param : bw = 16, float
            assert(sim.model.conv1.param_quantizers['weight'].enabled == True)
            assert(sim.model.conv1.param_quantizers['weight'].bitwidth == 16)
            assert(sim.model.conv1.param_quantizers['weight'].data_type == QuantizationDataType.float)

            assert(sim.model.conv1.output_quantizers[0].bitwidth == 16)
            assert(sim.model.conv1.output_quantizers[0].data_type == QuantizationDataType.float)

            assert(sim.model.fc.param_quantizers['weight'].enabled)
            assert(sim.model.fc.param_quantizers['bias'].enabled == False)
            assert(sim.model.fc.param_quantizers['weight'].bitwidth == 16)
            assert(sim.model.fc.param_quantizers['weight'].data_type == QuantizationDataType.float)
            assert(sim.model.fc.param_quantizers['bias'].bitwidth == 16)
            assert(sim.model.fc.param_quantizers['bias'].data_type == QuantizationDataType.float)
            assert(sim.model.fc.output_quantizers[0].bitwidth == 16)
            assert(sim.model.fc.output_quantizers[0].data_type == QuantizationDataType.float)
            assert(sim.model.relu1.output_quantizers[0].bitwidth == 16)
            assert(sim.model.relu1.output_quantizers[0].data_type == QuantizationDataType.float)

            # remove test config created
            qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False

    def test_check_correctness_of_dtype_bw_rules_valid_case(self):
        """
        Test to check api check_correctness_of_dtype_bw_rules, valid config case
        :return:
        """

        model = SingleResidual()
        model.eval()
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 8,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                        ],
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    }
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = os.path.join(tmp_dir, 'quantsim_config.json')
            with open(config_file, 'w') as f:
                json.dump(quantsim_config, f)

            INPUT_SHAPE = (1, 3, 32, 32)
            def forward_fn(model, _):
                torch.manual_seed(10)
                model.eval()
                with torch.no_grad():
                    _ = model(torch.randn(INPUT_SHAPE))
            from aimet_torch.quantsim_config.quantsim_config import QuantSimConfigurator
            dummy_input = torch.randn(INPUT_SHAPE)
            connected_graph = ConnectedGraph(model, dummy_input)

            qsim_config = QuantSimConfigurator(model, connected_graph, config_file,
                                               quantsim_output_bw=8, quantsim_param_bw=8,
                                               quantsim_data_type=QuantizationDataType.int)

            qsim_dtype_bw = QuantDtypeBwInfo(act_dtype=QuantizationDataType.int, act_bw=8,
                                             param_dtype=QuantizationDataType.int, param_bw=8)

            assert qsim_config.check_correctness_of_dtype_bw_rules(qsim_dtype_bw)

    def test_target_rule_enforced_apply_default_and_op_level_overrides_valid_case(self):
        """
        validates Config overrides provided are valid combination and application of both default level as well as
        op level kernel overrides for dtype and bitiwdth.
        Quantsim created with (int4, int4) defaults
        Default supported kernels override (at index 0 ) is (int 8, int 8) --> applied.
        Default at op level override at index 0 of supported_kernels for Conv type is
        (fp 16, fp16) --> applied to weight param.
        :return:
        """

        model = SingleResidual()
        model.eval()

        # quantsim config has default kernel overrides as well as op level kernel override
        # we begin with quantsim default config (int 4, int 4), during isntantiation of quantsim object.
        # Then, using config file we apply two levels of overrides.
        # 1) default supported_kernels at index 0 , is used to override default act/param bw dtype with int8 / int8
        # 2) After this, at op level, specifically for Conv types, there is a override provided as fp16/ fp16
        # So, param quantizers of conv shall be updated to FP16 as a override, while retaining output at int 8 as
        # configured by default level supported_kernels.

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 8,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 8,
                            "dtype": "int"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 4,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 4,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    },
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                        ]
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            INPUT_SHAPE = (1, 3, 32, 32)
            def forward_fn(model, _):
                torch.manual_seed(10)
                model.eval()
                with torch.no_grad():
                    _ = model(torch.randn(INPUT_SHAPE))

            # set enforce to true for this test
            qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32), in_place=True,
                                       default_data_type=QuantizationDataType.int, default_output_bw=4, default_param_bw=4)
            sim.compute_encodings(forward_fn, forward_pass_callback_args=None)

            # enforce is set to true
            # default supported kernels at index DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX (=0 in this case)
            # is not same as default quantsim bw and dtype(int 4/int4), apply default overrides (int8/ int8).

            assert(sim.model.fc.param_quantizers['weight'].enabled)
            assert(sim.model.fc.param_quantizers['bias'].enabled == False)
            assert(sim.model.fc.param_quantizers['weight'].bitwidth == 8)
            assert(sim.model.fc.param_quantizers['weight'].data_type == QuantizationDataType.int)
            assert(sim.model.fc.param_quantizers['bias'].bitwidth == 8)
            assert(sim.model.fc.param_quantizers['bias'].data_type == QuantizationDataType.int)
            assert(sim.model.fc.output_quantizers[0].bitwidth == 8)
            assert(sim.model.fc.output_quantizers[0].data_type == QuantizationDataType.int)
            assert(sim.model.relu1.output_quantizers[0].bitwidth == 8)
            assert(sim.model.relu1.output_quantizers[0].data_type == QuantizationDataType.int)

            # at op level (for Conv) check param quantizers are updated to fp16 while output is still retained at int8
            assert(sim.model.conv1.param_quantizers['weight'].enabled == True)
            assert(sim.model.conv1.param_quantizers['weight'].bitwidth == 16)
            assert(sim.model.conv1.param_quantizers['weight'].data_type == QuantizationDataType.float)

            assert(sim.model.conv1.output_quantizers[0].bitwidth == 8)
            assert(sim.model.conv1.output_quantizers[0].data_type == QuantizationDataType.int)

            # remove test config created
            qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False

    def test_target_rule_enforced_apply_op_level_overrides_fp16(self):
        """
        validates Config overrides provided are valid combination and application of aic100 specific rules.
        No dfeualt supported kernel override and op level FP16 support for LayerNorm and GeLU.
        Quantsim created with (int8, int8) defaults
        a) Default supported kernels override not provided.
        b) op level override at index 0 of supported_kernels for LayerNorm/GeLU type is
        (fp 16, fp16) --> applied to params.
        For GeLu, nothing is applied, as it has not params. output is retained at int 8.
        :return:
        """

        # aic100, no default supported kernels, op level for layernorm and gelu
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                }
            },
            "params": {},
            "op_type": {
                "LayerNorm": {
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                        ]
                },
                "GELU": {
                    "is_output_quantized": "True",
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                        ]
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            torch.manual_seed(10)
            model = ModelWithBertCustomLayerNormGelu()
            model.eval()

            random_input = torch.rand(1, 4, 4)

            def forward_pass(model, args):
                model.eval()
                with torch.no_grad():
                    model(*random_input)

            qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       dummy_input=random_input, default_data_type=QuantizationDataType.int,
                                       default_output_bw=8, default_param_bw=8,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'))


            # enforce is set to true
            # LayerNorm params should be set to FP 16, and activations as well since it is back to back with GeLU
            assert(sim.model.customln1.param_quantizers['weight'].data_type == QuantizationDataType.float)
            assert(sim.model.customln1.param_quantizers['weight'].bitwidth == 16)
            assert(sim.model.customln1.output_quantizers[0].data_type == QuantizationDataType.float)
            assert(sim.model.customln1.output_quantizers[0].bitwidth == 16)

            # gelu output should be set to fp16 as it has no output ops following it
            assert(sim.model.gelu1.output_quantizers[0].data_type == QuantizationDataType.float)
            assert(sim.model.gelu1.output_quantizers[0].bitwidth == 16)

            # remove test config created
            qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False

    def test_QuantDtypeBwInfo_data_class_1(self):
        """
        Test QuantDtypeBwInfo __eq__
        """
        q1 = QuantDtypeBwInfo(QuantizationDataType.int, 8, QuantizationDataType.float, 16)
        q2 = QuantDtypeBwInfo(QuantizationDataType.int, 16, QuantizationDataType.float, 16)
        q3 = QuantDtypeBwInfo(QuantizationDataType.int, 8, QuantizationDataType.float, 16)

        assert q1 != q2
        assert q1 == q3

    def test_fp16_back_to_back_overrides(self):
        """
        Test that activation tensors are set to fp16 as expected in case of standalone vs back to back.
        """
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                }
            },
            "params": {},
            "op_type": {
                "PRelu": {
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                        ]
                },
                "Add": {
                    "is_output_quantized": "True",
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                        ]
                }
            },
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            class ModelForFP16Override(torch.nn.Module):
                """
                Model for testing fp16 back to back overrides
                """

                def __init__(self):
                    super(ModelForFP16Override, self).__init__()
                    self.prelu1 = torch.nn.PReLU()
                    self.prelu2 = torch.nn.PReLU()
                    self.relu1 = torch.nn.ReLU()
                    self.add1 = Add()
                    self.prelu3 = torch.nn.PReLU()
                    self.prelu4 = torch.nn.PReLU()
                    self.add2 = Add()

                def forward(self, x1, x2, x3):
                    x1 = self.prelu1(x1)
                    x1 = self.prelu2(x1)
                    x1 = self.relu1(x1)
                    x1 = self.add1(x1, x2)
                    x1 = self.prelu3(x1)
                    x3 = self.prelu4(x3)
                    x1 = self.add2(x1, x3)
                    return x1

            # Model to test structured as follows:
            #   x1     x2        x3
            #   |       |        |
            # prelu1    |        |
            #   |       |        |
            # prelu2    |        |
            #   |       |        |
            # relu1     |        |
            #     \    /         |
            #      add1          |
            #        |           |
            #      prelu3     prelu4
            #            \   /
            #            add2
            #              |
            #
            # Since all modules except relu are in fp16 mode, we expect all quantizers to be set to fp16 except for relu1
            # output and add1 inputs

            model = ModelForFP16Override()
            model.eval()

            random_input = (torch.rand(1, 2), torch.rand(1, 2), torch.rand(1, 2))

            qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       dummy_input=random_input, default_data_type=QuantizationDataType.int,
                                       default_output_bw=8, default_param_bw=8,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'))
            for name, module in sim.quant_wrappers():
                if name == 'prelu2':
                    assert module.input_quantizers[0].bitwidth == 16
                    assert module.input_quantizers[0].data_type == QuantizationDataType.float
                    assert module.output_quantizers[0].bitwidth == 8
                    assert module.output_quantizers[0].data_type == QuantizationDataType.int
                elif name == 'relu1':
                    assert module.input_quantizers[0].bitwidth == 8
                    assert module.input_quantizers[0].data_type == QuantizationDataType.int
                    assert module.output_quantizers[0].bitwidth == 8
                    assert module.output_quantizers[0].data_type == QuantizationDataType.int
                elif name == 'add1':
                    assert module.input_quantizers[0].bitwidth == 8
                    assert module.input_quantizers[0].data_type == QuantizationDataType.int
                    assert module.input_quantizers[1].bitwidth == 8
                    assert module.input_quantizers[1].data_type == QuantizationDataType.int
                    assert module.output_quantizers[0].bitwidth == 16
                    assert module.output_quantizers[0].data_type == QuantizationDataType.float
                else:
                    for input_q in module.input_quantizers:
                        assert input_q.bitwidth == 16
                        assert input_q.data_type == QuantizationDataType.float
                    for output_q in module.output_quantizers:
                        assert output_q.bitwidth == 16
                        assert output_q.data_type == QuantizationDataType.float
                for param_q in module.param_quantizers.values():
                    assert param_q.bitwidth == 16
                    assert param_q.data_type == QuantizationDataType.float

            # remove test config created
            qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False

    def test_encoding_constraints(self):
        """ Test encoding constraints setting """

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                }
            },
            "params": {},
            "op_type": {
                "Softmax": {
                    "encoding_constraints": {
                        "min": -5.0,
                        "max": 5.0
                    },
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            class SoftmaxModel(torch.nn.Module):
                def __init__(self):
                    super(SoftmaxModel, self).__init__()
                    self.softmax = torch.nn.Softmax()

                def forward(self, inp):
                    x = self.softmax(inp)
                    return x

            model = SoftmaxModel()
            dummy_input = torch.tensor([0.5, 0.5])
            for config_file in [None, os.path.join(tmp_dir, 'quantsim_config.json')]:
                qsim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                            config_file=config_file)
                qsim.compute_encodings(lambda m, _: m(dummy_input), None)
                if not config_file:
                    assert qsim.model.softmax.output_quantizers[0].quant_scheme == QuantScheme.post_training_tf_enhanced
                    assert qsim.model.softmax.output_quantizers[0].encoding.min == 0.0
                    assert np.allclose(qsim.model.softmax.output_quantizers[0].encoding.max, 0.5, atol=1e-2)
                else:
                    assert qsim.model.softmax.output_quantizers[0].quant_scheme == QuantScheme.post_training_tf
                    assert np.allclose(qsim.model.softmax.output_quantizers[0].encoding.min, -5.0, atol=1e-1)
                    assert np.allclose(qsim.model.softmax.output_quantizers[0].encoding.max, 5.0, atol=1e-1)

    @pytest.mark.parametrize('sample_enc', [{"min": -4, "max": 4, "bitwidth": 8, "dtype": "int", "is_symmetric": "True"},
                                            {"min": -4, "max": 4, "bitwidth": 8, "dtype": "int", "is_symmetric": "False"},
                                            {"scale": 0.03, "offset": 8, "bitwidth": 8,
                                             "is_symmetric": "False", "dtype": "int"},
                                            {"scale": 0.03, "offset": 8, "bitwidth": 8,
                                             "is_symmetric": "True", "dtype": "int"}])
    def test_load_and_freeze_with_partial_encodings(self, sample_enc):
        """ Test load_and_freeze encoding API with partial_encodings """
        model = test_models.TinyModelWithNoMathInvariantOps()
        dummy_input = torch.randn([1, 3, 24, 24])

        sample_partial_encodings_list = [{"activation_encodings": {"conv1": {"input": {"0": sample_enc}},
                                                                   "mul1": {"output": {"0": sample_enc}}},
                                          "param_encodings": {}},
                                         {"activation_encodings": {"add1": {"output": {"0": sample_enc}}},
                                          "param_encodings": {}},
                                         {"activation_encodings": {"mul1": {"output": {"0": sample_enc}}},
                                          "param_encodings": {"conv1.weight": [sample_enc]}}]

        with tempfile.TemporaryDirectory() as tmp_dir:
            for partial_encodings in sample_partial_encodings_list:
                with open(os.path.join(tmp_dir, 'partial_encoding.json'), 'w') as f:
                    json.dump(partial_encodings, f)

                sim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf)

                sim.load_and_freeze_encodings(os.path.join(tmp_dir, 'partial_encoding.json'),
                                              ignore_when_quantizer_disabled=True)

                # all output quantizers and model input quantizer need to enabled (no op specific config)
                assert sim.model.mul1.output_quantizers[0].enabled
                assert sim.model.mul2.output_quantizers[0].enabled
                assert sim.model.add1.output_quantizers[0].enabled
                assert sim.model.conv1.output_quantizers[0].enabled
                assert sim.model.conv1.input_quantizers[0].enabled

                # conv param quantizer needs to be enabled
                assert sim.model.conv1.param_quantizers['weight'].enabled
                if partial_encodings['param_encodings']:
                    if sample_enc['is_symmetric'] == 'True':
                        assert sim.model.conv1.param_quantizers['weight'].use_symmetric_encodings
                    else:
                        assert not sim.model.conv1.param_quantizers['weight'].use_symmetric_encodings

    def test_load_encodings_to_allow_modifying_quantizer_type(self):
        """ Test load encodings API to allow modifying quantizer type based on encoding """
        model = test_models.TinyModelWithNoMathInvariantOps()
        dummy_input = torch.randn([1, 3, 24, 24])

        sample_act_enc = {"min": -4, "max": 4, "bitwidth": 8, "dtype": "int", "is_symmetric": "False"}
        sample_param_enc = {"min": -4, "max": 4, "bitwidth": 8, "dtype": "int", "is_symmetric": "True"}

        encodings = {"activation_encodings": {"conv1": {"input": {"0": sample_act_enc}},
                                              "mul1": {"output": {"0": sample_act_enc}}},
                     "param_encodings": {}}

        pcq_config = {
            "defaults":{
                "ops":{
                    "is_output_quantized": "True"
                },
                "params":{
                    "is_quantized": "True",
                    "is_symmetric": "True"
                },
                "strict_symmetric": "False",
                "per_channel_quantization": "True"
            },
            "params": {},
            "op_type": {},
            "model_input":{
                "is_input_quantized": "True"
            },
            "supergroups": [],
            "model_output": {}
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            pcq_config_file = os.path.join(tmp_dir, 'pcq_quantsim_config.json')
            with open(pcq_config_file, 'w') as f:
                json.dump(pcq_config, f)

            for config_file in [None, pcq_config_file]:
                if config_file is None:
                    # PTQ to PCQ case, initial quantizer is PTQ, but the encodings are of PCQ
                    encodings['param_encodings']['conv1.weight'] = [sample_param_enc for i in range(16)]
                else:
                    # PCQ to PTQ case, initial quantizer is PCQ, but the encodings are of PTQ
                    encodings['param_encodings']['conv1.weight'] = [sample_param_enc]

                with tempfile.TemporaryDirectory() as tmp_dir:
                    with open(os.path.join(tmp_dir, 'replace_quantizer_with_enc.json'), 'w') as f:
                        json.dump(encodings, f)

                    sim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf, config_file=config_file)

                    # Checking Quantizer type before loading encodings to Quantsim
                    if config_file is None:
                        assert isinstance(sim.model.conv1.param_quantizers['weight'], StaticGridPerTensorQuantizer)
                    else:
                        assert isinstance(sim.model.conv1.param_quantizers['weight'], StaticGridPerChannelQuantizer)

                    sim.load_and_freeze_encodings(os.path.join(tmp_dir, 'replace_quantizer_with_enc.json'),
                                                  ignore_when_quantizer_disabled=True)

                    sim.compute_encodings(lambda m, _: m(dummy_input), None)

                    # Checking whether the quantizer is modifed to required type after laoding encodings
                    if config_file is None:
                        assert isinstance(sim.model.conv1.param_quantizers['weight'], StaticGridPerChannelQuantizer)
                    else:
                        assert isinstance(sim.model.conv1.param_quantizers['weight'], StaticGridPerTensorQuantizer)

    def test_input_order_exception_for_multi_input_ops_in_connected_graph(self):
        model = test_models.ModelWithMultiInputOps()
        model.eval()
        dummy_input = (torch.randn(1, 3, 24, 24), torch.randn(3, 8, 2, 2))

        sim = QuantizationSimModel(model, dummy_input)
        # input order for the following ops must be preserved within
        # connected graph for the quantizers to be properly enabled/disabled
        # aimet_modules.Matmul
        assert (not sim.model.matmul.input_quantizers[0].enabled
                and sim.model.matmul.input_quantizers[1].enabled)
        # aimet_modules.Minimum
        assert (sim.model.min.input_quantizers[0].enabled
                and not sim.model.min.input_quantizers[1].enabled)
        # aimet_modules.Concat
        assert (not sim.model.concat.input_quantizers[0].enabled
                and sim.model.concat.input_quantizers[1].enabled
                and not sim.model.concat.input_quantizers[2].enabled)
        # aimet_modules.DynamicConv2d
        assert (not sim.model.dynamic_conv.input_quantizers[0].enabled
                and sim.model.dynamic_conv.input_quantizers[1].enabled
                and sim.model.dynamic_conv.input_quantizers[2].enabled)
