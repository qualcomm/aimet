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
import json
import os
import pytest
from aimet_common.defs import QuantizationDataType
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_onnx.quantsim import QuantizationSimModel
from models import models_for_tests


class TestQuantSimConfig:
    """Tests for applying config to QuantizationSimModel"""
    def test_qs_config_dummy_model(self):
        model = models_for_tests.build_dummy_model()
        sim = QuantizationSimModel(model, use_cuda=False)
        assert sim.qc_quantize_op_dict['conv_w'].enabled == True
        assert sim.qc_quantize_op_dict['conv_b'].enabled == False
        assert sim.qc_quantize_op_dict['fc_w'].enabled == True
        assert sim.qc_quantize_op_dict['fc_b'].enabled == False
        assert sim.qc_quantize_op_dict['input'].enabled == True
        assert sim.qc_quantize_op_dict['3'].enabled == False
        assert sim.qc_quantize_op_dict['4'].enabled == True
        assert sim.qc_quantize_op_dict['5'].enabled == True
        assert sim.qc_quantize_op_dict['output'].enabled == True

    def test_default_config(self):
        model = models_for_tests.build_dummy_model()

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
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json', use_cuda=False)
        for name in ['3', '4', '5', 'output']:
            assert sim.qc_quantize_op_dict[name].enabled == True
            assert sim.qc_quantize_op_dict[name].use_symmetric_encodings == False

    def test_param_config(self):
        model = models_for_tests.build_dummy_model()

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
            "params":
                {
                    "weight": {
                        "is_quantized": "True",
                        "is_symmetric": "True"
                    },
                },
            "op_type": {
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json', use_cuda=False)
        for name in ['conv_w', 'fc_w']:
            assert sim.qc_quantize_op_dict[name].enabled == True
            assert sim.qc_quantize_op_dict[name].use_symmetric_encodings == True

        for name in ['conv_b', 'fc_b']:
            assert sim.qc_quantize_op_dict[name].enabled == False
            assert sim.qc_quantize_op_dict[name].use_symmetric_encodings == True

    def test_op_level_config_and_model_output(self):
        model = models_for_tests.build_dummy_model()

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
                        "weight": {
                            "is_quantized": "True",
                            "is_symmetric": "False"
                        }
                    },
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True",
            }
        }
        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json', use_cuda=False)

        assert sim.qc_quantize_op_dict['conv_w'].enabled == True
        assert sim.qc_quantize_op_dict['conv_w'].use_symmetric_encodings == False
        assert sim.qc_quantize_op_dict['input'].enabled == True
        assert sim.qc_quantize_op_dict['input'].use_symmetric_encodings == False
        assert sim.qc_quantize_op_dict['output'].enabled == True

    def test_config_for_model_input(self):
        model = models_for_tests.build_dummy_model()

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

        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json', use_cuda=False)
        assert sim.qc_quantize_op_dict['input'].enabled == True

    def test_parse_config_file_supergroups(self):
        """ Test that supergroup quantization parameters are set correctly when using json config file """
        model = models_for_tests.build_dummy_model()

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
            ],
            "model_input": {},
            "model_output": {}
        }

        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json', use_cuda=False)

        # 3 in conv output, 4 is relu output (even though it was not touched with Conv, relu pattern, it was disabled for
        # relu maxpool pattern
        for name in ['3', '4',]:
            assert sim.qc_quantize_op_dict[name].enabled == False

        assert sim.qc_quantize_op_dict['5'].enabled == True

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_parse_config_file_symmetric_modes(self):
        """ Test that model output quantization parameters are set correctly when using json config file """
        model = models_for_tests.build_dummy_model()

        quantsim_config = {
            "defaults":
            {
                "ops": {},
                "params":
                {
                    "is_symmetric": "True"
                },
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
        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json', use_cuda=False)

        for quantizer in sim.qc_quantize_op_dict.values():
            assert quantizer.use_strict_symmetric == True
            assert quantizer.use_unsigned_symmetric == False

    def test_generate_and_apply_op_level_config(self):
        model = models_for_tests.build_dummy_model()

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
                        "weight": {
                            "is_quantized": "True",
                            "is_symmetric": "False"
                        }
                    },
                    "per_channel_quantization": "True",
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True",
            }
        }
        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json', use_cuda=False)
        assert sim.qc_quantize_op_dict['conv_w'].quant_info.usePerChannelMode == True
        assert sim.qc_quantize_op_dict['fc_w'].quant_info.usePerChannelMode == False


    def test_supported_kernels(self):
        """
        Tests the generated supported_kernels
        """
        model = models_for_tests.single_residual_model()
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

        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json')
        op_to_supported_kernels = sim._op_to_supported_kernel
        for op_name in op_to_supported_kernels:
                assert len(op_to_supported_kernels[op_name]) == 1
                if 'Conv' in op_name:
                    assert op_to_supported_kernels[op_name] == [((16, QuantizationDataType.int), (8, QuantizationDataType.int))]
                else:
                    assert op_to_supported_kernels[op_name] == [((16, QuantizationDataType.float), (16, QuantizationDataType.float))]

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
        supported_kernels_conv = sim.get_supported_kernels()["Conv"]
        assert len(supported_kernels_conv) == 1
        assert supported_kernels_conv == expected_supported_kernels


    def test_matmul_perchannel_config(self, tmp_path):
        model = models_for_tests.weight_matmul_model(in_features=10, out_features=20)
        sim = QuantizationSimModel(model, config_file=get_path_for_per_channel_config())
        assert not sim.qc_quantize_op_dict["weight"].quant_info.usePerChannelMode

    @pytest.mark.parametrize("config", (None, get_path_for_per_channel_config()))
    def test_disable_batchnorm_stats_quantization(self, config):
        model = models_for_tests.batchnorm_model()
        sim = QuantizationSimModel(model, config_file=config)
        assert not sim.qc_quantize_op_dict["batchnorm.input_mean"].enabled
        assert not sim.qc_quantize_op_dict["batchnorm.input_var"].enabled
        assert not sim.qc_quantize_op_dict["batchnorm.bias"].enabled
        assert sim.qc_quantize_op_dict["batchnorm.weight"].enabled
        assert sim.qc_quantize_op_dict["model_output"].enabled
