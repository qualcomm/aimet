# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import yaml

try:
    from aimet_onnx.common.quantsim_config.v2.parser import (
        OpSchema,
        OpSignature,
        Argument,
        FixedRange,
        EqualRange,
        parse_op,
        qtype,
    )
except ImportError:
    from aimet_torch.common.quantsim_config.v2.parser import (
        OpSchema,
        OpSignature,
        Argument,
        FixedRange,
        EqualRange,
        parse_op,
        qtype,
    )


def test_op_schema_parsing():
    yaml_config = yaml.safe_load("""
op_type: Conv
type_constraints:
  # W8A8
  - X: uint8
    W: int8[channel_axis=0]
    B: int32
    Y: uint8
  # W8A16
  - X: uint16
    W: int8[channel_axis=0]
    B: int32
    Y: uint16
  # W16A16
  - X: uint16
    W: int16[channel_axis=0]
    B: int32
    Y: uint16
""")
    schemas: set[OpSchema] = parse_op(yaml_config)
    assert len(schemas) == 6
    assert schemas == {
        # W8A8 per-tensor (NOTE: per-tensor is trivially supported if per-channel is supported)
        OpSchema(
            signature=OpSignature(
                op_type="Conv",
                inputs=(
                    Argument(name="X", qtype=qtype("uint8")),
                    Argument(name="W", qtype=qtype("int8")),
                    Argument(name="B", qtype=qtype("int32")),
                ),
                outputs=(Argument(name="Y", qtype=qtype("uint8")),),
            ),
            value_constraints=(),
        ),
        # W8A8 per-channel
        OpSchema(
            signature=OpSignature(
                op_type="Conv",
                inputs=(
                    Argument(name="X", qtype=qtype("uint8")),
                    Argument(name="W", qtype=qtype("int8", channel_axis=0)),
                    Argument(name="B", qtype=qtype("int32")),
                ),
                outputs=(Argument(name="Y", qtype=qtype("uint8")),),
            ),
            value_constraints=(),
        ),
        # W8A16 per-tensor (NOTE: per-tensor is trivially supported if per-channel is supported)
        OpSchema(
            signature=OpSignature(
                op_type="Conv",
                inputs=(
                    Argument(name="X", qtype=qtype("uint16")),
                    Argument(name="W", qtype=qtype("int8")),
                    Argument(name="B", qtype=qtype("int32")),
                ),
                outputs=(Argument(name="Y", qtype=qtype("uint16")),),
            ),
            value_constraints=(),
        ),
        # W8A16 per-channel
        OpSchema(
            signature=OpSignature(
                op_type="Conv",
                inputs=(
                    Argument(name="X", qtype=qtype("uint16")),
                    Argument(name="W", qtype=qtype("int8", channel_axis=0)),
                    Argument(name="B", qtype=qtype("int32")),
                ),
                outputs=(Argument(name="Y", qtype=qtype("uint16")),),
            ),
            value_constraints=(),
        ),
        # W16A16 per-tensor (NOTE: per-tensor is trivially supported if per-channel is supported)
        OpSchema(
            signature=OpSignature(
                op_type="Conv",
                inputs=(
                    Argument(name="X", qtype=qtype("uint16")),
                    Argument(name="W", qtype=qtype("int16")),
                    Argument(name="B", qtype=qtype("int32")),
                ),
                outputs=(Argument(name="Y", qtype=qtype("uint16")),),
            ),
            value_constraints=(),
        ),
        # W16A16 per-channel
        OpSchema(
            signature=OpSignature(
                op_type="Conv",
                inputs=(
                    Argument(name="X", qtype=qtype("uint16")),
                    Argument(name="W", qtype=qtype("int16", channel_axis=0)),
                    Argument(name="B", qtype=qtype("int32")),
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
    schemas = parse_op(yaml_config)
    assert len(schemas) == 2
    assert schemas == {
        OpSchema(
            signature=OpSignature(
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
    schemas = parse_op(yaml_config)
    assert len(schemas) == 2
    assert schemas == {
        OpSchema(
            signature=OpSignature(
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
    axes: n/a
    reduced: uint8
  - data: uint16
    axes: n/a
    reduced: uint16
value_contraints:
  tie:
    - [data, reduced]
""")
    schemas = parse_op(yaml_config)
    assert len(schemas) == 2
    assert schemas == {
        OpSchema(
            signature=OpSignature(
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
    schemas = parse_op(yaml_config)
    assert len(schemas) == 2
    assert schemas == {
        OpSchema(
            signature=OpSignature(
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
