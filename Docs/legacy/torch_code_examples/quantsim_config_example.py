# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: skip-file
# defaults start
{"defaults": {
    "ops": {                                # Required dictionary, but can be empty
        "is_output_quantized": "True",      # Optional: Possible settings: True
        "is_symmetric": "False"             # Optional: Possible settings: True, False
    },
    "params": {                             # Required dictionary, but can be empty
        "is_quantized": "True",             # Optional: Possible settings: True, False
        "is_symmetric": "True"              # Optional: Possible settings: True, False
    },
    "strict_symmetric": "False",            # Optional: Possible settings: True, False
    "unsigned_symmetric": "True",           # Optional: Possible settings: True, False
    "per_channel_quantization": "False"     # Optional: Possible settings: True, False
    },
# defaults end
# params start
    "params": {                         # Can specify 0 or more param types
        "weight": {
            "is_quantized": "True",     # Optional: Possible settings: True, False
            "is_symmetric": "True"      # Optional: Possible settings: True, False
        }
    },
# params end
# op_type start
    "op_type": {                                # Can specify 0 or more ONNX op types
        "Gemm": {
            "is_input_quantized": "True",       # Optional: Possible settings: True
            "is_output_quantized": "False",     # Optional: Possible settings: True, False
            "per_channel_quantization": "True", # Optional: Possible settings: True, False
            "params": {                         # Optional, can specify 1 or more param types
                "weight": {
                    "is_quantized": "True",     # Optional: Possible settings: True, False
                    "is_symmetric": "True"      # Optional: Possible settings: True, False
                }
            },
        },
    },
# op_type end
# supergroups start
    "supergroups": [    # Can specify 0 or more supergroup lists made up of ONNX op types
        {
            "op_list": ["Conv", "Relu"]
        },
        {
            "op_list": ["Conv", "Clip"]
        },
        {
            "op_list": ["Add", "Relu"]
        },
        {
            "op_list": ["Gemm", "Relu"]
        }
    ],
# supergroups end
# model_input start
    "model_input": {
        "is_input_quantized": "True"    # Optional: Possible settings: True
    },
# model_input end
# model_output start
    "model_output": {
        "is_output_quantized": "True"   # Optional: Possible settings: True
    }
# model_output end
}
