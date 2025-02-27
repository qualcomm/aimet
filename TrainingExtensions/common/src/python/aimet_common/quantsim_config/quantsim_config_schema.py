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
""" Schema used to validate json configuration file. Docs: https://json-schema.org/learn/ """


QUANTSIM_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "defaults": {
            "type": "object",
            "properties": {
                "ops": {
                    "type": "object",
                    "properties": {
                        "is_input_quantized": {
                            "type": "string",
                            "pattern": "^True$|^False$"
                        },
                        "is_output_quantized": {
                            "type": "string",
                            "pattern": "^True$|^False$"
                        },
                        "is_symmetric": {
                            "type": "string",
                            "pattern": "^True$|^False$"
                        }
                    },
                    "additionalProperties": False
                },
                "params": {
                    "type": "object",
                    "properties": {
                        "is_quantized": {
                            "type": "string",
                            "pattern": "^True$|^False$"
                        },
                        "is_symmetric": {
                            "type": "string",
                            "pattern": "^True$|^False$"
                        }
                    },
                    "additionalProperties": False
                },
                "strict_symmetric": {
                    "type": "string",
                    "pattern": "^True$|^False$"
                },
                "unsigned_symmetric": {
                    "type": "string",
                    "pattern": "^True$|^False$"
                },
                "per_channel_quantization": {
                    "type": "string",
                    "pattern": "^True$|^False$"
                },
                "supported_kernels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "activation": {
                                "type": "object",
                                "properties": {
                                    "bitwidth": {
                                        "type": "integer",
                                        "enum" : [4, 8, 16, 32]
                                    },
                                    "dtype": {
                                        "type": "string",
                                        "pattern": "^int$|^float$"
                                    },
                                },
                                "required": ["bitwidth", "dtype"],
                                "additionalProperties": False
                            },
                            "param": {
                                "type": "object",
                                "properties": {
                                    "bitwidth": {
                                        "type": "integer",
                                        "enum" : [4, 8, 16, 32]
                                    },
                                    "dtype": {
                                        "type": "string",
                                        "pattern": "^int$|^float$"
                                    }
                                },
                                "required": ["bitwidth", "dtype"],
                                "additionalProperties": False
                            },
                        },
                        "required": ["activation", "param"],
                        "additionalProperties": False
                    },
                    "minItems": 1,
                    "additionalItems": False
                },
                "hw_version": {
                    "type": "string"
                }
            },
            "required": ["ops", "params"],
            "additionalProperties": False
        },
        "params": {
            "type": "object",
            "patternProperties": {
                ".*": {
                    "type": "object",
                    "properties": {
                        "is_quantized": {
                            "type": "string",
                            "pattern": "^True$|^False$"
                        },
                        "is_symmetric": {
                            "type": "string",
                            "pattern": "^True$|^False$"
                        }
                    },
                    "additionalProperties": False
                }
            }
        },
        "op_type": {
            "type": "object",
            "patternProperties": {
                ".*": {
                    "type": "object",
                    "properties": {
                        "is_input_quantized": {
                            "type": "string",
                            "pattern": "^True$|^False$"
                        },
                        "is_output_quantized": {
                            "type": "string",
                            "pattern": "^True$|^False$"
                        },
                        "is_symmetric": {
                            "type": "string",
                            "pattern": "^True$|^False$"
                        },
                        "params": {
                            "type": "object",
                            "patternProperties": {
                                ".*": {
                                    "type": "object",
                                    "properties": {
                                        "is_quantized": {
                                            "type": "string",
                                            "pattern": "^True$|^False$"
                                        },
                                        "is_symmetric": {
                                            "type": "string",
                                            "pattern": "^True$|^False$"
                                        }
                                    },
                                    "additionalProperties": False
                                }
                            }
                        },
                        "supported_kernels": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "activation": {
                                        "type": "object",
                                        "properties": {
                                            "bitwidth": {
                                                "type": "integer",
                                                "enum" : [4, 8, 16, 32]
                                            },
                                            "dtype": {
                                                "type": "string",
                                                "pattern": "^int$|^float$"
                                            }
                                        },
                                        "required": ["bitwidth", "dtype"],
                                        "additionalProperties": False
                                    },
                                    "param": {
                                        "type": "object",
                                        "properties": {
                                            "bitwidth": {
                                                "type": "integer",
                                                "enum" : [4, 8, 16, 32]
                                            },
                                            "dtype": {
                                                "type": "string",
                                                "pattern": "^int$|^float$"
                                            }
                                        },
                                        "required": ["bitwidth", "dtype"],
                                        "additionalProperties": False
                                    },
                                },
                                "required": ["activation"],
                                "additionalProperties": False
                            },
                            "minItems": 1,
                            "additionalItems": False
                        },
                        "per_channel_quantization": {
                            "type": "string",
                            "pattern": "^True$|^False$"
                        },
                        "encoding_constraints": {
                            "type": "object",
                            "properties": {
                                "min": {
                                    "type": "number"
                                },
                                "max": {
                                    "type": "number"
                                }
                            },
                            "required": ["min", "max"]
                        },
                    },
                    "additionalProperties": False
                }
            }
        },
        "supergroups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op_list": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 2
                    }
                },
                "required": ["op_list"],
                "additionalProperties": False
            },
            "additionalItems": False
        },
        "supergroup_pass_list": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "minItems": 0
        },
        "model_input": {
            "type": "object",
            "properties": {
                "is_input_quantized": {
                    "type": "string",
                    "pattern": "^True$|^False$"
                }
            },
            "additionalProperties": False
        },
        "model_output": {
            "type": "object",
            "properties": {
                "is_output_quantized": {
                    "type": "string",
                    "pattern": "^True$|^False$"
                }
            },
            "additionalProperties": False
        }
    },
    "required": ["defaults", "params", "op_type", "supergroups", "model_input", "model_output"],
    "additionalProperties": False
}
