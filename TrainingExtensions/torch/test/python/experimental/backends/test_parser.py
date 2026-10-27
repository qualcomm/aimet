# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import yaml

from aimet_torch.experimental.backends.parser import parse_aten_op
from aimet_torch.experimental.backends.schema import (
    OpSchema,
    OpSignature,
    Argument,
    FixedRange,
    EqualRange,
    qtype,
)


def test_aten_op_schema_parsing():
    yaml_config = yaml.safe_load("""
op_type: aten::conv1d | aten::conv2d | aten::conv3d
type_constraints:
  # W8A8
  - input: uint8
    weight: int8[channel_axis=0]
    bias: int32[channel_axis=0]
    stride: null
    padding: null
    dilation: null
    groups: null
    $ret[0]: uint8
  # W8A16
  - input: uint16
    weight: int8[channel_axis=0]
    bias: int32[channel_axis=0]
    stride: null
    padding: null
    dilation: null
    groups: null
    $ret[0]: uint16
  # W16A16
  - input: uint16
    weight: int16[channel_axis=0]
    bias: int32[channel_axis=0]
    stride: null
    padding: null
    dilation: null
    groups: null
    $ret[0]: uint16
""")
    schemas = parse_aten_op(yaml_config)
    assert len(schemas) == 3

    for op_type in ("conv1d", "conv2d", "conv3d"):
        assert schemas.pop(f"aten::{op_type}") == {
            # W8A8 per-channel
            OpSchema(
                signature=OpSignature(
                    domain="aten",
                    op_type=op_type,
                    inputs=(
                        Argument(name="input", qtype=qtype("uint8")),
                        Argument(
                            name="weight",
                            qtype=qtype("int8", channel_axis=0),
                            role="weight",
                        ),
                        Argument(
                            name="bias",
                            qtype=qtype("int32", channel_axis=0),
                            role="bias",
                        ),
                        Argument(name="stride", qtype=None),
                        Argument(name="padding", qtype=None),
                        Argument(name="dilation", qtype=None),
                        Argument(name="groups", qtype=None),
                    ),
                    outputs=(Argument(name="$ret[0]", qtype=qtype("uint8")),),
                ),
                value_constraints=(),
            ),
            # W8A16 per-channel
            OpSchema(
                signature=OpSignature(
                    domain="aten",
                    op_type=op_type,
                    inputs=(
                        Argument(name="input", qtype=qtype("uint16")),
                        Argument(
                            name="weight",
                            qtype=qtype("int8", channel_axis=0),
                            role="weight",
                        ),
                        Argument(
                            name="bias",
                            qtype=qtype("int32", channel_axis=0),
                            role="bias",
                        ),
                        Argument(name="stride", qtype=None),
                        Argument(name="padding", qtype=None),
                        Argument(name="dilation", qtype=None),
                        Argument(name="groups", qtype=None),
                    ),
                    outputs=(Argument(name="$ret[0]", qtype=qtype("uint16")),),
                ),
                value_constraints=(),
            ),
            # W16A16 per-channel
            OpSchema(
                signature=OpSignature(
                    domain="aten",
                    op_type=op_type,
                    inputs=(
                        Argument(name="input", qtype=qtype("uint16")),
                        Argument(
                            name="weight",
                            qtype=qtype("int16", channel_axis=0),
                            role="weight",
                        ),
                        Argument(
                            name="bias",
                            qtype=qtype("int32", channel_axis=0),
                            role="bias",
                        ),
                        Argument(name="stride", qtype=None),
                        Argument(name="padding", qtype=None),
                        Argument(name="dilation", qtype=None),
                        Argument(name="groups", qtype=None),
                    ),
                    outputs=(Argument(name="$ret[0]", qtype=qtype("uint16")),),
                ),
                value_constraints=(),
            ),
        }

    yaml_config = yaml.safe_load("""
op_type: aten::_softmax
type_constraints:
  - self: uint8
    dim: null
    half_to_float: null
    $ret[0]: uint8
  - self: uint16
    dim: null
    half_to_float: null
    $ret[0]: uint16
value_contraints:
  fix:
    - $ret[0]: {min: 0.0, max: 1.0}
""")
    schemas = parse_aten_op(yaml_config)
    assert len(schemas) == 1
    schemas = schemas.pop("aten::_softmax")
    assert schemas == {
        OpSchema(
            signature=OpSignature(
                domain="aten",
                op_type="_softmax",
                inputs=(
                    Argument(name="self", qtype=qtype("uint8")),
                    Argument(name="dim", qtype=None),
                    Argument(name="half_to_float", qtype=None),
                ),
                outputs=(Argument(name="$ret[0]", qtype=qtype("uint8")),),
            ),
            value_constraints=(
                FixedRange(
                    arg=Argument(name="$ret[0]", qtype=qtype("uint8")), min=0.0, max=1.0
                ),
            ),
        ),
        OpSchema(
            signature=OpSignature(
                domain="aten",
                op_type="_softmax",
                inputs=(
                    Argument(name="self", qtype=qtype("uint16")),
                    Argument(name="dim", qtype=None),
                    Argument(name="half_to_float", qtype=None),
                ),
                outputs=(Argument(name="$ret[0]", qtype=qtype("uint16")),),
            ),
            value_constraints=(
                FixedRange(
                    arg=Argument(name="$ret[0]", qtype=qtype("uint16")),
                    min=0.0,
                    max=1.0,
                ),
            ),
        ),
    }

    yaml_config = yaml.safe_load("""
op_type: aten::relu
type_constraints:
  - self: uint8
    $ret[0]: uint8
  - self: uint16
    $ret[0]: uint16
value_contraints:
  fix:
    - $ret[0]: {min: 0.0}
  tie:
    - [self, "$ret[0]"]
""")
    schemas = parse_aten_op(yaml_config)
    assert len(schemas) == 1
    schemas = schemas.pop("aten::relu")
    assert schemas == {
        OpSchema(
            signature=OpSignature(
                domain="aten",
                op_type="relu",
                inputs=(Argument(name="self", qtype=qtype("uint8")),),
                outputs=(Argument(name="$ret[0]", qtype=qtype("uint8")),),
            ),
            value_constraints=(
                FixedRange(
                    arg=Argument(name="$ret[0]", qtype=qtype("uint8")),
                    min=0.0,
                    max=None,
                ),
                EqualRange(
                    arg1=Argument(name="self", qtype=qtype("uint8")),
                    arg2=Argument(name="$ret[0]", qtype=qtype("uint8")),
                ),
            ),
        ),
        OpSchema(
            signature=OpSignature(
                domain="aten",
                op_type="relu",
                inputs=(Argument(name="self", qtype=qtype("uint16")),),
                outputs=(Argument(name="$ret[0]", qtype=qtype("uint16")),),
            ),
            value_constraints=(
                FixedRange(
                    arg=Argument(name="$ret[0]", qtype=qtype("uint16")),
                    min=0.0,
                    max=None,
                ),
                EqualRange(
                    arg1=Argument(name="self", qtype=qtype("uint16")),
                    arg2=Argument(name="$ret[0]", qtype=qtype("uint16")),
                ),
            ),
        ),
    }

    yaml_config = yaml.safe_load("""
op_type: aten::mean
type_constraints:
  - self: uint8
    dtype: null
    $ret[0]: uint8
  - self: uint16
    dtype: null
    $ret[0]: uint16
value_contraints:
  tie:
    - [self, "$ret[0]"]
""")
    schemas = parse_aten_op(yaml_config)
    assert len(schemas) == 1
    schemas = schemas.pop("aten::mean")
    assert schemas == {
        OpSchema(
            signature=OpSignature(
                domain="aten",
                op_type="mean",
                inputs=(
                    Argument(name="self", qtype=qtype("uint8")),
                    Argument(name="dtype", qtype=None),
                ),
                outputs=(Argument(name="$ret[0]", qtype=qtype("uint8")),),
            ),
            value_constraints=(
                EqualRange(
                    arg1=Argument(name="self", qtype=qtype("uint8")),
                    arg2=Argument(name="$ret[0]", qtype=qtype("uint8")),
                ),
            ),
        ),
        OpSchema(
            signature=OpSignature(
                domain="aten",
                op_type="mean",
                inputs=(
                    Argument(name="self", qtype=qtype("uint16")),
                    Argument(name="dtype", qtype=None),
                ),
                outputs=(Argument(name="$ret[0]", qtype=qtype("uint16")),),
            ),
            value_constraints=(
                EqualRange(
                    arg1=Argument(name="self", qtype=qtype("uint16")),
                    arg2=Argument(name="$ret[0]", qtype=qtype("uint16")),
                ),
            ),
        ),
    }

    yaml_config = yaml.safe_load("""
op_type: aten::cat
type_constraints:
  - tensors: uint8
    dim: null
    $ret[0]: uint8
  - tensors: uint16
    dim: null
    $ret[0]: uint16
value_contraints:
  tie:
    - [tensors, "$ret[0]"]
""")
    schemas = parse_aten_op(yaml_config)
    assert len(schemas) == 1
    schemas = schemas.pop("aten::cat")
    assert schemas == {
        OpSchema(
            signature=OpSignature(
                domain="aten",
                op_type="cat",
                inputs=(
                    Argument(name="tensors", qtype=qtype("uint8")),
                    Argument(name="dim", qtype=None),
                ),
                outputs=(Argument(name="$ret[0]", qtype=qtype("uint8")),),
            ),
            value_constraints=(
                EqualRange(
                    arg1=Argument(name="tensors", qtype=qtype("uint8")),
                    arg2=Argument(name="$ret[0]", qtype=qtype("uint8")),
                ),
            ),
        ),
        OpSchema(
            signature=OpSignature(
                domain="aten",
                op_type="cat",
                inputs=(
                    Argument(name="tensors", qtype=qtype("uint16")),
                    Argument(name="dim", qtype=None),
                ),
                outputs=(Argument(name="$ret[0]", qtype=qtype("uint16")),),
            ),
            value_constraints=(
                EqualRange(
                    arg1=Argument(name="tensors", qtype=qtype("uint16")),
                    arg2=Argument(name="$ret[0]", qtype=qtype("uint16")),
                ),
            ),
        ),
    }

    yaml_config = yaml.safe_load("""
op_type: aten::split_with_sizes
type_constraints:
  - "*": uint8
  - "*": uint16
value_contraints:
  tie:
    - ["*"]
""")
    schemas = parse_aten_op(yaml_config)
    assert len(schemas) == 1
    schemas = schemas.pop("aten::split_with_sizes")
    assert schemas == {
        OpSchema(
            signature=OpSignature(
                domain="aten",
                op_type="split_with_sizes",
                inputs=(
                    Argument(name="self", qtype=qtype("uint8")),
                    Argument(name="split_sizes", qtype=None),
                    Argument(name="dim", qtype=None),
                ),
                outputs=(Argument(name="$ret[0]", qtype=qtype("uint8")),),
            ),
            value_constraints=(
                EqualRange(
                    arg1=Argument(name="self", qtype=qtype("uint8")),
                    arg2=Argument(name="$ret[0]", qtype=qtype("uint8")),
                ),
            ),
        ),
        OpSchema(
            signature=OpSignature(
                domain="aten",
                op_type="split_with_sizes",
                inputs=(
                    Argument(name="self", qtype=qtype("uint16")),
                    Argument(name="split_sizes", qtype=None),
                    Argument(name="dim", qtype=None),
                ),
                outputs=(Argument(name="$ret[0]", qtype=qtype("uint16")),),
            ),
            value_constraints=(
                EqualRange(
                    arg1=Argument(name="self", qtype=qtype("uint16")),
                    arg2=Argument(name="$ret[0]", qtype=qtype("uint16")),
                ),
            ),
        ),
    }
