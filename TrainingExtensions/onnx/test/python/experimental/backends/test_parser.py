# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import yaml

from aimet_onnx.common.experimental.backends.schema import (
    OpSchema,
    OpSignature,
    Argument,
    FixedRange,
    EqualRange,
    qtype,
)
from aimet_onnx.experimental.backends.parser import parse_onnx_op


def test_op_schema_parsing():
    yaml_config = yaml.safe_load("""
op_type: Conv
type_constraints:
  # W8A8
  - X: uint8
    W: int8[channel_axis=0]
    B: int32[channel_axis=0]
    Y: uint8
  # W8A16
  - X: uint16
    W: int8[channel_axis=0]
    B: int32[channel_axis=0]
    Y: uint16
  # W16A16
  - X: uint16
    W: int16[channel_axis=0]
    B: int32[channel_axis=0]
    Y: uint16
""")
    schemas: set[OpSchema] = parse_onnx_op(yaml_config)
    assert len(schemas) == 3
    assert schemas == {
        # W8A8 per-channel
        OpSchema(
            signature=OpSignature(
                domain="onnx",
                op_type="Conv",
                inputs=(
                    Argument(name="X", qtype=qtype("uint8")),
                    Argument(name="W", qtype=qtype("int8", channel_axis=0)),
                    Argument(name="B", qtype=qtype("int32", channel_axis=0)),
                ),
                outputs=(Argument(name="Y", qtype=qtype("uint8")),),
            ),
            value_constraints=(),
        ),
        # W8A16 per-channel
        OpSchema(
            signature=OpSignature(
                domain="onnx",
                op_type="Conv",
                inputs=(
                    Argument(name="X", qtype=qtype("uint16")),
                    Argument(name="W", qtype=qtype("int8", channel_axis=0)),
                    Argument(name="B", qtype=qtype("int32", channel_axis=0)),
                ),
                outputs=(Argument(name="Y", qtype=qtype("uint16")),),
            ),
            value_constraints=(),
        ),
        # W16A16 per-channel
        OpSchema(
            signature=OpSignature(
                domain="onnx",
                op_type="Conv",
                inputs=(
                    Argument(name="X", qtype=qtype("uint16")),
                    Argument(name="W", qtype=qtype("int16", channel_axis=0)),
                    Argument(name="B", qtype=qtype("int32", channel_axis=0)),
                ),
                outputs=(Argument(name="Y", qtype=qtype("uint16")),),
            ),
            value_constraints=(),
        ),
    }

    yaml_config = yaml.safe_load("""
op_type: Softmax
type_constraints:
  - input: uint8
    output: uint8
  - input: uint16
    output: uint16
value_contraints:
  fix:
    - output: {min: 0.0, max: 1.0}
""")
    schemas = parse_onnx_op(yaml_config)
    assert len(schemas) == 2
    assert schemas == {
        OpSchema(
            signature=OpSignature(
                domain="onnx",
                op_type="Softmax",
                inputs=(Argument(name="input", qtype=qtype("uint8")),),
                outputs=(Argument(name="output", qtype=qtype("uint8")),),
            ),
            value_constraints=(
                FixedRange(
                    arg=Argument(name="output", qtype=qtype("uint8")), min=0.0, max=1.0
                ),
            ),
        ),
        OpSchema(
            signature=OpSignature(
                domain="onnx",
                op_type="Softmax",
                inputs=(Argument(name="input", qtype=qtype("uint16")),),
                outputs=(Argument(name="output", qtype=qtype("uint16")),),
            ),
            value_constraints=(
                FixedRange(
                    arg=Argument(name="output", qtype=qtype("uint16")), min=0.0, max=1.0
                ),
            ),
        ),
    }

    yaml_config = yaml.safe_load("""
op_type: Relu
type_constraints:
  - X: uint8
    Y: uint8
  - X: uint16
    Y: uint16
value_contraints:
  fix:
    - Y: {min: 0.0}
  tie:
    - [X, Y]
""")
    schemas = parse_onnx_op(yaml_config)
    assert len(schemas) == 2
    assert schemas == {
        OpSchema(
            signature=OpSignature(
                domain="onnx",
                op_type="Relu",
                inputs=(Argument(name="X", qtype=qtype("uint8")),),
                outputs=(Argument(name="Y", qtype=qtype("uint8")),),
            ),
            value_constraints=(
                FixedRange(
                    arg=Argument(name="Y", qtype=qtype("uint8")), min=0.0, max=None
                ),
                EqualRange(
                    arg1=Argument(name="X", qtype=qtype("uint8")),
                    arg2=Argument(name="Y", qtype=qtype("uint8")),
                ),
            ),
        ),
        OpSchema(
            signature=OpSignature(
                domain="onnx",
                op_type="Relu",
                inputs=(Argument(name="X", qtype=qtype("uint16")),),
                outputs=(Argument(name="Y", qtype=qtype("uint16")),),
            ),
            value_constraints=(
                FixedRange(
                    arg=Argument(name="Y", qtype=qtype("uint16")), min=0.0, max=None
                ),
                EqualRange(
                    arg1=Argument(name="X", qtype=qtype("uint16")),
                    arg2=Argument(name="Y", qtype=qtype("uint16")),
                ),
            ),
        ),
    }

    yaml_config = yaml.safe_load("""
op_type: ReduceMean
type_constraints:
  - data: uint8
    axes: null
    reduced: uint8
  - data: uint16
    axes: null
    reduced: uint16
value_contraints:
  tie:
    - [data, reduced]
""")
    schemas = parse_onnx_op(yaml_config)
    assert len(schemas) == 2
    assert schemas == {
        OpSchema(
            signature=OpSignature(
                domain="onnx",
                op_type="ReduceMean",
                inputs=(
                    Argument(name="data", qtype=qtype("uint8")),
                    Argument(name="axes", qtype=None),
                ),
                outputs=(Argument(name="reduced", qtype=qtype("uint8")),),
            ),
            value_constraints=(
                EqualRange(
                    arg1=Argument(name="data", qtype=qtype("uint8")),
                    arg2=Argument(name="reduced", qtype=qtype("uint8")),
                ),
            ),
        ),
        OpSchema(
            signature=OpSignature(
                domain="onnx",
                op_type="ReduceMean",
                inputs=(
                    Argument(name="data", qtype=qtype("uint16")),
                    Argument(name="axes", qtype=None),
                ),
                outputs=(Argument(name="reduced", qtype=qtype("uint16")),),
            ),
            value_constraints=(
                EqualRange(
                    arg1=Argument(name="data", qtype=qtype("uint16")),
                    arg2=Argument(name="reduced", qtype=qtype("uint16")),
                ),
            ),
        ),
    }

    yaml_config = yaml.safe_load("""
op_type: Concat
type_constraints:
  - inputs: uint8
    concat_result: uint8
  - inputs: uint16
    concat_result: uint16
value_contraints:
  tie:
    - [inputs, concat_result]
""")
    schemas = parse_onnx_op(yaml_config)
    assert len(schemas) == 2
    assert schemas == {
        OpSchema(
            signature=OpSignature(
                domain="onnx",
                op_type="Concat",
                inputs=(Argument(name="inputs", qtype=qtype("uint8")),),
                outputs=(Argument(name="concat_result", qtype=qtype("uint8")),),
            ),
            value_constraints=(
                EqualRange(
                    arg1=Argument(name="inputs", qtype=qtype("uint8")),
                    arg2=Argument(name="concat_result", qtype=qtype("uint8")),
                ),
            ),
        ),
        OpSchema(
            signature=OpSignature(
                domain="onnx",
                op_type="Concat",
                inputs=(Argument(name="inputs", qtype=qtype("uint16")),),
                outputs=(Argument(name="concat_result", qtype=qtype("uint16")),),
            ),
            value_constraints=(
                EqualRange(
                    arg1=Argument(name="inputs", qtype=qtype("uint16")),
                    arg2=Argument(name="concat_result", qtype=qtype("uint16")),
                ),
            ),
        ),
    }
