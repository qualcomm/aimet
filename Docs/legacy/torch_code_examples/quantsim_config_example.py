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
