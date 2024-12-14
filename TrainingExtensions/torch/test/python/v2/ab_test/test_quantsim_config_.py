# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import itertools
import torch

from aimet_common.connected_graph.connectedgraph_utils import CG_SPLIT
from aimet_common.defs import QuantScheme, QuantizationDataType, QuantDtypeBwInfo, SupportedKernelsAction

import aimet_torch._base.quantsim
import aimet_torch.v1.quantsim
from aimet_torch.quantsim_config import quantsim_config as qsim_config
from aimet_torch.quantsim_config.quantsim_config import get_all_ops_in_neighborhood
from aimet_torch import utils
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch._base.nn.modules.custom import Add

from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize

from ..models_.models_to_test import SingleResidual, QuantSimTinyModel, MultiInput, SingleResidualWithModuleAdd, \
    SingleResidualWithAvgPool, ModelWithBertCustomLayerNormGelu


TORCH_INT_DTYPES = (torch.int, torch.int8, torch.int16, torch.int32, torch.int64)
TORCH_FLOAT_DTYPES = (torch.float, torch.float16, torch.float32, torch.float64, torch.bfloat16)



@pytest.fixture
def enforce_target_dtype_bitwidth_config():
    enforce_target_dtype_bitwidth_config = qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG
    try:
        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True
        yield
    finally:
        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = enforce_target_dtype_bitwidth_config


# pylint: disable=protected-access
# From https://github.com/quic/aimet/blob/b9cb122b57f591b8e62bb2bf48bb178151148011/TrainingExtensions/torch/test/python/test_quantsim_config.py#L76
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

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32), in_place=True)
            for name, module in sim.model.named_children():
                assert isinstance(module, BaseQuantizationMixin)
                # Output of add op is input quantized
                if name == 'relu3':
                    assert module.input_quantizers[0] is not None
                else:
                    assert module.input_quantizers[0] is None
                if name in ["conv1", "conv2"]:
                    # Output quantizers of conv1 and conv2 are
                    # disabled due to the subsequent batchnorm
                    assert module.output_quantizers[0] is None
                else:
                    assert module.output_quantizers[0] is not None
                if module.input_quantizers[0] is not None:
                    assert not module.input_quantizers[0].symmetric
                if module.output_quantizers[0] is not None:
                    assert not module.output_quantizers[0].symmetric


                for _, param_quantizer in module.param_quantizers.items():
                    assert param_quantizer is None

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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32))
            for _, module in sim.model.named_children():
                assert isinstance(module, BaseQuantizationMixin)
                for param_name, param_quantizer in module.param_quantizers.items():
                    if param_name == 'weight':
                        if module in (sim.model.bn1, sim.model.bn2):
                            assert param_quantizer is None
                        else:
                            assert param_quantizer is not None
                        if param_quantizer is not None:
                            assert not param_quantizer.symmetric
                    else:
                        assert param_quantizer is None

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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32))
            for name, module in sim.model.named_children():
                assert isinstance(module, BaseQuantizationMixin)
                if isinstance(module, torch.nn.Conv2d):
                    assert module.input_quantizers[0] is not None
                    if name in ["conv1", "conv2"]:
                        assert module.output_quantizers[0] is None
                    else:
                        assert module.output_quantizers[0] is not None
                else:
                    # Output of add op is input quantized
                    if name == 'relu3':
                        assert module.input_quantizers[0] is not None
                    else:
                        assert module.input_quantizers[0] is None
                    assert module.output_quantizers[0] is not None

                if module.input_quantizers[0] is not None:
                    assert not module.input_quantizers[0].symmetric
                if module.output_quantizers[0] is not None:
                    assert not module.output_quantizers[0].symmetric

                for param_name, param_quantizer in module.param_quantizers.items():
                    if isinstance(module, torch.nn.Conv2d) and param_name == 'bias':
                        assert param_quantizer is not None
                        assert not param_quantizer.symmetric
                    else:
                        assert param_quantizer is None

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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20)))

            return sim

    def test_parse_config_file_op_type_per_channel(self):
        """ Test that op specific quantization parameters are set correctly when using json config file """

        # test 1: expect all to be StaticGridPerChannelQuantizer
        per_channel_fields = {"defaults": "True", "Conv": "True", "Gemm": "True"}
        sim = self._test_parse_config_file_op_type_per_channel_helper(per_channel_fields)

        for name, module in sim.model.named_children():
            assert isinstance(module, BaseQuantizationMixin)
            if 'weight' in module.param_quantizers:
                assert any(dim > 1 for dim in module.param_quantizers['weight'].shape)


        # test 2: expect all but Conv to be StaticGridPerChannelQuantizer
        per_channel_fields = {"defaults": "True", "Conv": "False", "Gemm": "True"}
        sim = self._test_parse_config_file_op_type_per_channel_helper(per_channel_fields)

        for name, module in sim.model.named_children():
            assert isinstance(module, BaseQuantizationMixin)
            if 'weight' in module.param_quantizers:
                if isinstance(module, torch.nn.Conv2d):
                    assert all(dim == 1 for dim in module.param_quantizers['weight'].shape)
                else:
                    assert any(dim > 1 for dim in module.param_quantizers['weight'].shape)


        # test 3: expect all but Conv and Gemm to be StaticGridPerChannelQuantizer
        per_channel_fields = {"defaults": "True", "Conv": "False", "Gemm": "False"}
        sim = self._test_parse_config_file_op_type_per_channel_helper(per_channel_fields)

        for name, module in sim.model.named_children():
            assert isinstance(module, BaseQuantizationMixin)
            if 'weight' in module.param_quantizers:
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    assert all(dim == 1 for dim in module.param_quantizers['weight'].shape)
                else:
                    assert any(dim > 1 for dim in module.param_quantizers['weight'].shape)


        # test 4: expect all in StaticGridPerTensorQuantizer except Conv which will be in StaticGridPerChannelQuantizer
        per_channel_fields = {"defaults": "False", "Conv": "True", "Gemm": "False"}
        sim = self._test_parse_config_file_op_type_per_channel_helper(per_channel_fields)

        for name, module in sim.model.named_children():
            assert isinstance(module, BaseQuantizationMixin)
            if 'weight' in module.param_quantizers:
                if isinstance(module, torch.nn.Conv2d):
                    assert any(dim > 1 for dim in module.param_quantizers['weight'].shape)
                else:
                    assert all(dim == 1 for dim in module.param_quantizers['weight'].shape)

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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
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
                sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                           config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                           dummy_input=torch.rand(1, 3, 32, 32))
                for _, module in sim.model.named_children():
                    assert isinstance(module, BaseQuantizationMixin)
                    assert len(module.supported_kernels) == 1
                    if isinstance(module, torch.nn.Conv2d):
                        if module.param_quantizers['weight'] is not None:
                            assert all(dim == 1 for dim in module.param_quantizers['weight'].shape)
                        assert module.supported_kernels == [((16, QuantizationDataType.int), (8, QuantizationDataType.int))]
                    else:
                        if module.param_quantizers and module.param_quantizers['weight'] is not None:
                            assert any(dim > 1 for dim in module.param_quantizers['weight'].shape)
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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
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

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32), default_param_bw=16, default_output_bw=16,
                                       default_data_type=QuantizationDataType.int)

            for _, module in sim.model.named_children():
                assert isinstance(module, BaseQuantizationMixin)
                if isinstance(module, torch.nn.Conv2d):
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
                    QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       in_place=True, dummy_input=torch.rand(1, 3, 32, 32))

            # Expected supergroups: (square bracket indicates a supergroup)
            # in -> [conv1->bn1->relu1->maxpool] -> [conv2->bn2->relu2] -> [conv3->relu3->avgpool] -> [conv4] -> [fc] -> out

            for _, module in sim.model.named_children():
                assert isinstance(module, BaseQuantizationMixin)
                # All input quantizers are disabled by config
                assert module.input_quantizers[0] is None

            # First supergroup
            assert sim.model.conv1.output_quantizers[0] is None
            assert sim.model.bn1.output_quantizers[0] is None
            assert sim.model.relu1.output_quantizers[0] is None
            assert sim.model.maxpool.output_quantizers[0] is not None

            # Second supergroup
            assert sim.model.conv2.output_quantizers[0] is None
            assert sim.model.bn2.output_quantizers[0] is None
            assert sim.model.relu2.output_quantizers[0] is not None

            # Third supergroup
            assert model.conv3.output_quantizers[0] is None
            assert sim.model.relu3.output_quantizers[0] is None
            assert sim.model.avgpool.output_quantizers[0] is not None

            # Supergroups with only one operation
            assert model.conv4.output_quantizers[0] is not None
            assert model.fc.output_quantizers[0] is not None

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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32))
            for name, module in sim.model.named_children():
                assert isinstance(module, BaseQuantizationMixin)
                if name in ['conv3', 'ada']:
                    # model.conv3 and model.ada are inputs to add
                    assert module.output_quantizers[0] is not None
                else:
                    assert module.output_quantizers[0] is None
                assert module.input_quantizers[0] is None

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
                x = self.add2(x, inputs[2])
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

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20), torch.randn(1, 8, 11, 11)), in_place=True)
            for name, module in sim.model.named_children():
                assert isinstance(module, BaseQuantizationMixin)
                if name in ('conv1', 'conv3'):
                    assert module.input_quantizers[0] is not None
                elif name == 'add2':
                    assert module.input_quantizers[0] is None
                    assert module.input_quantizers[1] is not None
                else:
                    assert module.input_quantizers[0] is None
                assert module.output_quantizers[0] is None
                if module.input_quantizers[0] is not None:
                    assert not module.input_quantizers[0].symmetric

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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32))
            for name, module in sim.model.named_children():
                assert isinstance(module, BaseQuantizationMixin)
                if name == 'fc':
                    # model.conv3 and model.ada are inputs to add
                    assert module.output_quantizers[0] is not None
                else:
                    assert module.output_quantizers[0] is None
                assert module.input_quantizers[0] is None

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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       in_place=True, dummy_input=torch.rand(1, 3, 32, 32))

            # Expected supergroups: (square bracket indicates a supergroup)
            # in -> [conv1] -> [bn1]-> ... -> [conv3] -----> [(+)->relu3->avgpool] -> [fc] -> out
            #                           |                      ^
            #                           +--> [conv4] -> [ada] -+

            # If add were not part of the supergroup, relu's input quantizer would be enabled
            assert sim.model.relu3.input_quantizers[0] is None

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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       in_place=True, dummy_input=torch.rand(1, 3, 32, 32))
            for _, module in sim.model.named_children():
                assert isinstance(module, BaseQuantizationMixin)
                # Check configs for starts of supergroups
                if module in [model.add, model.conv1, model.conv2]:
                    # If add were not part of the supergroup, relu's input quantizer would be enabled
                    assert module.output_quantizers[0] is None
                else:
                    assert module.output_quantizers[0] is not None
                assert module.input_quantizers[0] is None

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
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32))
            for _, module in sim.model.named_children():
                assert isinstance(module, BaseQuantizationMixin)
                for q in module.input_quantizers:
                    if q is not None:
                        assert q.symmetric
                for q in module.output_quantizers:
                    if q is not None:
                        assert q.symmetric
                for q in module.param_quantizers.values():
                    if q is not None:
                        assert q.symmetric

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

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32).cuda(), in_place=True)
            for name, module in sim.model.named_children():
                assert isinstance(module, BaseQuantizationMixin)
                # Output of add op is input quantized
                if name == 'relu3':
                    assert module.input_quantizers[0] is not None
                else:
                    assert module.input_quantizers[0] is None
                if name in ["conv1", "conv2"]:
                    # Output quantizers of conv1 and conv2 are
                    # disabled due to the subsequent batchnorm
                    assert module.output_quantizers[0] is None
                else:
                    assert module.output_quantizers[0] is not None
                if module.input_quantizers[0] is not None:
                    assert not module.input_quantizers[0].symmetric
                if module.output_quantizers[0] is not None:
                    assert not module.output_quantizers[0].symmetric

                for _, param_quantizer in module.param_quantizers.items():
                    assert param_quantizer is None

    def test_gelu_layernorm_quantsim_config(self):
        """
        Create a network with LayerNorm and GELU
        Override quantization config and check config is applied
        This is a validation for actual entry in the map_torch_types_to_onnx in onnx_utils
        This is used by connected graph to apply op level specific quantsim config.
        :return:
        """

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
            assert sim.model.ln1.param_quantizers['weight'] is None
            assert all(dim == 1 for dim in sim.model.ln1.param_quantizers['bias'].shape)


            # LayerNorm input quantization is disabled by default
            # override with custom config file, this needs appropriate entry in onnx node name mapping
            assert all(dim == 1 for dim in sim.model.ln1.input_quantizers[0].shape)
            assert sim.model.ln1.input_quantizers[0] is not None
            in_quantizer = sim.model.ln1.input_quantizers[0]
            assert in_quantizer is not None # disabled by default, override with config file
            assert isinstance(in_quantizer.encoding_analyzer, MinMaxEncodingAnalyzer)
            assert in_quantizer.bitwidth == 8


            # GELU input quantization is disabled by default
            # override with custom config file, this needs appropriate entry in onnx node name mapping
            assert all(dim == 1 for dim in sim.model.gelu1.input_quantizers[0].shape)
            in_quantizer = sim.model.gelu1.input_quantizers[0]
            assert in_quantizer is not None # disabled by default, override with config file
            assert isinstance(in_quantizer.encoding_analyzer, MinMaxEncodingAnalyzer)
            assert in_quantizer.bitwidth == 8

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

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32), in_place=True,
                                       default_param_bw=8, default_output_bw=8, default_data_type=QuantizationDataType.int)
            sim.compute_encodings(forward_fn, forward_pass_callback_args=None)

            # all quantizers should be quantsim default quantsim dtype and bw  (int 8)
            assert sim.model.conv1.param_quantizers['weight'] is not None
            assert sim.model.conv1.param_quantizers['weight'].bitwidth == 8
            assert isinstance(sim.model.conv1.param_quantizers['weight'], QuantizeDequantize)
            assert isinstance(sim.model.conv1.param_quantizers['weight'], QuantizeDequantize)

            assert sim.model.conv1.output_quantizers[0] is None

            # all quantizers should be quantsim default quantsim dtype and bw  (int 8)
            # that is  QUANTSIM DEFAULT bw / dtype (int 8).
            assert sim.model.fc.param_quantizers['weight'] is not None
            assert sim.model.fc.param_quantizers['bias'] is None
            assert sim.model.fc.param_quantizers['weight'].bitwidth == 8
            assert isinstance(sim.model.fc.param_quantizers['weight'], QuantizeDequantize)
            assert isinstance(sim.model.fc.param_quantizers['weight'], QuantizeDequantize)
            assert sim.model.fc.output_quantizers[0].bitwidth == 8
            assert isinstance(sim.model.fc.output_quantizers[0], QuantizeDequantize)
            assert isinstance(sim.model.fc.output_quantizers[0], QuantizeDequantize)
            assert sim.model.relu1.output_quantizers[0].bitwidth == 8
            assert isinstance(sim.model.relu1.output_quantizers[0], QuantizeDequantize)
            assert isinstance(sim.model.relu1.output_quantizers[0], QuantizeDequantize)

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

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32), in_place=True,
                                       default_data_type=QuantizationDataType.int, default_output_bw=8, default_param_bw=8)
            sim.compute_encodings(forward_fn, forward_pass_callback_args=None)

            # enforce is true, however default quantsim bw / dtype (fp16) is not the config file supported kernels override at index 0.
            # apply override 0 # activation : bw = 16, float # param : bw = 16, float
            assert isinstance(sim.model.conv1.param_quantizers['weight'], FloatQuantizeDequantize)
            assert sim.model.conv1.param_quantizers['weight'].bitwidth == 16

            # output quantizer of conv1 is not instantiated due to supergroup config
            assert sim.model.conv1.output_quantizers[0] is None

            assert isinstance(sim.model.fc.param_quantizers['weight'], FloatQuantizeDequantize)
            assert sim.model.fc.param_quantizers['weight'].bitwidth == 16
            assert sim.model.fc.param_quantizers['bias'] is None
            assert isinstance(sim.model.fc.output_quantizers[0], FloatQuantizeDequantize)
            assert sim.model.fc.output_quantizers[0].bitwidth == 16
            assert isinstance(sim.model.relu1.output_quantizers[0], FloatQuantizeDequantize)
            assert sim.model.relu1.output_quantizers[0].bitwidth == 16

            # remove test config created
            qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False

    @pytest.mark.skip('tf_enhanced is not implemented yet')
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
            assert isinstance(sim.model.conv1.param_quantizers['weight'], FloatQuantizeDequantize)
            assert sim.model.conv1.param_quantizers['weight'].bitwidth == 16

            # output quantizer of conv1 is not instantiated due to supergroup config
            assert sim.model.conv1.output_quantizers[0] is None

            assert isinstance(sim.model.fc.param_quantizers['weight'], FloatQuantizeDequantize)
            assert sim.model.fc.param_quantizers['weight'].bitwidth == 16
            assert sim.model.fc.param_quantizers['bias'] is None
            assert isinstance(sim.model.fc.output_quantizers[0], FloatQuantizeDequantize)
            assert sim.model.fc.output_quantizers[0].bitwidth == 16
            assert isinstance(sim.model.relu1.output_quantizers[0], FloatQuantizeDequantize)
            assert sim.model.relu1.output_quantizers[0].bitwidth == 16

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

    @pytest.mark.skip('tf_enhanced is not implemented yet')
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

            assert isinstance(sim.model.fc.param_quantizers['weight'], QuantizeDequantize)
            assert sim.model.fc.param_quantizers['weight'].bitwidth == 8
            assert sim.model.fc.param_quantizers['bias'] is None
            assert isinstance(sim.model.fc.output_quantizers[0], QuantizeDequantize)
            assert sim.model.fc.output_quantizers[0].bitwidth == 8
            assert isinstance(sim.model.relu1.output_quantizers[0], QuantizeDequantize)
            assert sim.model.relu1.output_quantizers[0].bitwidth == 8

            # at op level (for Conv) check param quantizers are updated to fp16 while output is still retained at int8
            assert isinstance(sim.model.conv1.param_quantizers['weight'], FloatQuantizeDequantize)
            assert sim.model.conv1.param_quantizers['weight'].bitwidth == 16

            # output quantizer of conv1 is not instantiated due to supergroup config
            assert sim.model.conv1.output_quantizers[0] is None

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
            assert isinstance(sim.model.customln1.param_quantizers['weight'], FloatQuantizeDequantize)
            assert sim.model.customln1.param_quantizers['weight'].bitwidth == 16
            assert isinstance(sim.model.customln1.output_quantizers[0], FloatQuantizeDequantize)
            assert sim.model.customln1.output_quantizers[0].bitwidth == 16

            # gelu output should be set to fp16 as it has no output ops following it
            assert isinstance(sim.model.gelu1.output_quantizers[0], FloatQuantizeDequantize)
            assert sim.model.gelu1.output_quantizers[0].bitwidth == 16

            # remove test config created
            qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False

    def test_fp16_back_to_back_overrides(self, enforce_target_dtype_bitwidth_config):
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

            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                       dummy_input=random_input, default_data_type=QuantizationDataType.int,
                                       default_output_bw=8, default_param_bw=8,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'))
            prelu1 = sim.model.prelu1
            prelu2 = sim.model.prelu2
            relu1 = sim.model.relu1
            add1 = sim.model.add1
            prelu3 = sim.model.prelu3
            prelu4 = sim.model.prelu4
            add2 = sim.model.add2

            assert isinstance(prelu1.input_quantizers[0], FloatQuantizeDequantize)
            assert prelu1.input_quantizers[0].is_float16()
            assert isinstance(prelu1.output_quantizers[0], FloatQuantizeDequantize)
            assert prelu1.output_quantizers[0].is_float16()
            assert isinstance(prelu1.param_quantizers['weight'], FloatQuantizeDequantize)
            assert prelu1.param_quantizers['weight'].is_float16()

            assert prelu2.input_quantizers[0] is None
            assert isinstance(prelu2.output_quantizers[0], QuantizeDequantize)
            assert prelu2.output_quantizers[0].bitwidth == 8
            assert isinstance(prelu2.param_quantizers['weight'], FloatQuantizeDequantize)
            assert prelu2.param_quantizers['weight'].is_float16()

            assert relu1.input_quantizers[0] is None
            assert isinstance(relu1.output_quantizers[0], QuantizeDequantize)
            assert relu1.output_quantizers[0].bitwidth == 8

            assert add1.input_quantizers[0] is None
            assert isinstance(add1.input_quantizers[1], QuantizeDequantize)
            assert add1.input_quantizers[1].bitwidth == 8
            assert isinstance(add1.output_quantizers[0], FloatQuantizeDequantize)
            assert add1.output_quantizers[0].is_float16()

            assert prelu3.input_quantizers[0] is None
            assert isinstance(prelu3.output_quantizers[0], FloatQuantizeDequantize)
            assert prelu3.output_quantizers[0].is_float16()
            assert isinstance(prelu3.param_quantizers['weight'], FloatQuantizeDequantize)
            assert prelu3.param_quantizers['weight'].is_float16()

            assert isinstance(prelu4.input_quantizers[0], FloatQuantizeDequantize)
            assert prelu4.input_quantizers[0].is_float16()
            assert isinstance(prelu4.output_quantizers[0], FloatQuantizeDequantize)
            assert prelu4.output_quantizers[0].is_float16()
            assert isinstance(prelu4.param_quantizers['weight'], FloatQuantizeDequantize)
            assert prelu4.param_quantizers['weight'].is_float16()

            assert add2.input_quantizers[0] is None
            assert add2.input_quantizers[1] is None
            assert isinstance(add2.output_quantizers[0], FloatQuantizeDequantize)
            assert add2.output_quantizers[0].is_float16()

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
            config_file = os.path.join(tmp_dir, 'quantsim_config.json')
            qsim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf,
                                        config_file=config_file)
            assert isinstance(qsim.model.softmax.output_quantizers[0].encoding_analyzer, MinMaxEncodingAnalyzer)
            assert torch.allclose(qsim.model.softmax.output_quantizers[0].min, torch.tensor([-5.0]), atol=1e-1)
            assert torch.allclose(qsim.model.softmax.output_quantizers[0].max, torch.tensor([5.0]), atol=1e-1)

    @pytest.mark.parametrize("quant_scheme", [QuantScheme.post_training_tf,
                                              QuantScheme.post_training_percentile,
                                              QuantScheme.training_range_learning_with_tf_init])
    def test_requires_grad_is_set_by_quant_scheme(self, quant_scheme):
        """ Test that requires_grad property of quant params are set correctly with different quant scheme """
        model = SingleResidual()
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

            sim = QuantizationSimModel(model, quant_scheme=quant_scheme,
                                       config_file=os.path.join(tmp_dir, 'quantsim_config.json'),
                                       dummy_input=torch.rand(1, 3, 32, 32), in_place=True)
            for _, module in sim.model.named_children():
                assert isinstance(module, BaseQuantizationMixin)

                for quantizer in itertools.chain(module.input_quantizers,
                                                 module.output_quantizers,
                                                 module.param_quantizers.values()):
                    if quantizer is not None:
                        if quant_scheme in (QuantScheme.post_training_tf, QuantScheme.post_training_percentile):
                            assert not quantizer.min.requires_grad
                            assert not quantizer.max.requires_grad
                        else:
                            assert quantizer.min.requires_grad
                            assert quantizer.max.requires_grad

    @pytest.mark.cuda
    def test_quantsim_device_and_dtype(self):
        model = SingleResidual().cuda()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=torch.rand(1, 3, 32, 32).cuda())

        # All quantization parameters should be set to device=cuda
        for param in sim.model.parameters():
            assert param.is_cuda

        for buffer in sim.model.buffers():
            assert buffer.is_cuda
