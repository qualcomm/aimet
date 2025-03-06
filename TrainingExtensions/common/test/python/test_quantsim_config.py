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
""" Module for testing quantsim config feature """

import os
import json
import unittest
from unittest.mock import patch
import jsonschema
from aimet_common.quantsim_config.json_config_importer import _validate_syntax, _validate_semantics, JsonConfigImporter,\
    ConfigDictKeys
from aimet_common.quantsim_config.quantsim_config import _build_list_of_permutations, OnnxConnectedGraphTypeMapper, \
    QuantSimConfigurator, QuantizationDataType


class TestJsonConfigImporter(unittest.TestCase):
    """ Class containing unit tests for json config importer feature """
    def test_validate_syntax(self):
        """ Test syntactic validation for config files """
        # No defaults
        quantsim_config = {
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            _validate_syntax(quantsim_config)

        # Missing ops dict
        quantsim_config = {
            "defaults": {
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "False"
                }
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            _validate_syntax(quantsim_config)

        # Bad value for is_symmetric
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "is_symmetric": "true"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "False"
                }
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            _validate_syntax(quantsim_config)

        # Extra field in ops dict
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "is_symmetric": "True",
                    "extra_field": "True"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "False"
                }
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            _validate_syntax(quantsim_config)

        # Supergroups length less than 2
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "is_symmetric": "True",
                    "extra_field": "True"
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
                    "op_list": ["Conv"]
                }
            ],
            "model_input": {},
            "model_output": {}
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            _validate_syntax(quantsim_config)

        # verify supported_kernels has at least one entry
        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {},
                "supported_kernels": []
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            _validate_syntax(quantsim_config)

        # verify param in supported_kernels has both bitwidth and dtype
        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {},
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16
                        }
                    }
                ]
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            _validate_syntax(quantsim_config)

        # verify param in supported_kernels has a valid bitwidth [4, 8, 16, 32]
        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {},
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 1,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            _validate_syntax(quantsim_config)



    def test_validate_semantics(self):
        """ Test semantic validation for config files """
        # NOTE: using bool True instead of str "True" since validate semantics expects _convert_configs_values_to_bool
        # to already have been run
        # Test that is_input_quantized setting in default ops is caught
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_input_quantized": True
                },
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with self.assertRaises(NotImplementedError):
            _validate_semantics(quantsim_config)

        # Test that is_output_quantized = False setting in default ops is caught
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": False
                },
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with self.assertRaises(NotImplementedError):
            _validate_semantics(quantsim_config)

        # Test that is_input_quantized = False setting in op_type is caught
        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {}
            },
            "params": {},
            "op_type": {
                "Conv": {
                    "is_input_quantized": False
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with self.assertRaises(NotImplementedError):
            _validate_semantics(quantsim_config)

        # Test that is_input_quantized setting = False in model_input is caught
        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {
                "is_input_quantized": False
            },
            "model_output": {}
        }
        with self.assertRaises(NotImplementedError):
            _validate_semantics(quantsim_config)

        # Test that is_output_quantized setting = False in model_output is caught
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
                "is_output_quantized": False
            }
        }
        with self.assertRaises(NotImplementedError):
            _validate_semantics(quantsim_config)


class TestQuantSimConfig(unittest.TestCase):
    """ Class containing unit tests for quantsim config feature """
    def test_build_list_of_permutations(self):
        """ Test that asserts are raised if config file does not exist or is not parsable by json """
        onnx_conn_graph_type_pairs = [
            [["onnx1"], ["conn1_1", "conn1_2", "conn1_3"]],
            [["onnx2"], ["conn2_1", "conn2_2"]],
            [["onnx3"], ["conn3_1", "conn3_2", "conn3_3", "conn_3_4"]]
        ]
        onnx_conn_graph_mapper = OnnxConnectedGraphTypeMapper(onnx_conn_graph_type_pairs)
        op_list = ["onnx1", "onnx2", "onnx3"]
        all_permutations = _build_list_of_permutations(op_list, onnx_conn_graph_mapper)
        self.assertEqual(24, len(all_permutations))
        for permutation in all_permutations:
            self.assertEqual(3, len(permutation))

        # check that all permutations are different
        permutation_sets = [set(permutation) for permutation in all_permutations]
        for index, elem in enumerate(permutation_sets):
            for _, elem_2 in enumerate(permutation_sets[index+1:]):
                self.assertNotEqual(elem, elem_2)

    def test_get_strict_symmetric_flag(self):
        """ test get_strict_symmetric_flag() """

        # config with default strict symmetric flag.
        config = {
            "defaults": {
                "ops": {},
                "params": {},
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }
        with open('./config.json', 'w') as f:
            json.dump(config, f)

        try:
            configs = JsonConfigImporter.import_json_config_file(config_file='./config.json')
            assert not configs[ConfigDictKeys.DEFAULTS].get(ConfigDictKeys.STRICT_SYMMETRIC, False)
        finally:
            if os.path.isfile('./config.json'):
                os.remove('./config.json')

        # not default case.
        config = {
            "defaults": {
                "ops": {},
                "params": {},
                "strict_symmetric": "True"
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }
        with open('./config.json', 'w') as f:
            json.dump(config, f)

        try:
            configs = JsonConfigImporter.import_json_config_file(config_file='./config.json')
            assert configs[ConfigDictKeys.DEFAULTS].get(ConfigDictKeys.STRICT_SYMMETRIC, False)
        finally:
            if os.path.isfile('./config.json'):
                os.remove('./config.json')

    def test_get_unsigned_symmetric_flag(self):
        """ test get_unsinged_symmetric_flag() """

        # config with default unsigned symmetric flag.
        config = {
            "defaults": {
                "ops": {},
                "params": {},
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }
        with open('./config.json', 'w') as f:
            json.dump(config, f)

        try:
            configs = JsonConfigImporter.import_json_config_file(config_file='./config.json')
            assert configs[ConfigDictKeys.DEFAULTS].get(ConfigDictKeys.UNSIGNED_SYMMETRIC, True)
        finally:
            if os.path.isfile('./config.json'):
                os.remove('./config.json')

        # not default case.
        config = {
            "defaults": {
                "ops": {},
                "params": {},
                "unsigned_symmetric": "False"
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }
        with open('./config.json', 'w') as f:
            json.dump(config, f)

        try:
            configs = JsonConfigImporter.import_json_config_file(config_file='./config.json')
            assert not configs[ConfigDictKeys.DEFAULTS].get(ConfigDictKeys.UNSIGNED_SYMMETRIC, True)
        finally:
            if os.path.isfile('./config.json'):
                os.remove('./config.json')

    @patch("aimet_common.quantsim_config.quantsim_config.QuantSimConfigurator.__abstractmethods__", set())
    def test_op_type_default_override_supported_kernel_lookup(self):
        # config with default unsigned symmetric flag.
        config = {
            "defaults": {
                "ops": {},
                "params": {},
            },
            "params": {},
            "op_type": {
                "Type1": {
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
                    ]
                },
                "Type2": {
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
                "Type3": {
                    "is_output_quantized": "True"
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }
        with open('./config.json', 'w') as f:
            json.dump(config, f)

        qsim_config = QuantSimConfigurator(config_file='./config.json', default_data_type=QuantizationDataType.int,
                                           default_output_bw=8, default_param_bw=8)
        assert qsim_config._op_type_default_override_supported_kernel_lookup('Type1', 16, QuantizationDataType.float)
        assert not qsim_config._op_type_default_override_supported_kernel_lookup('Type2', 16,
                                                                                 QuantizationDataType.float)
        assert not qsim_config._op_type_default_override_supported_kernel_lookup('Type3', 16,
                                                                                 QuantizationDataType.float)

        if os.path.isfile('./config.json'):
            os.remove('./config.json')
