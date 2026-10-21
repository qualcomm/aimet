# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import re
from dataclasses import dataclass
from typing import TypeAlias


_TYPE_STR_PATTERN = re.compile(r"^(?P<dtype>\w+)(?:\[(?P<attrs>.*)\])?$")


@dataclass(frozen=True)
class qtype:
    dtype: str
    channel_axis: int | None = None
    block_axis: int | None = None

    @classmethod
    def from_str(cls, type_str: str) -> "list[qtype]":
        match = _TYPE_STR_PATTERN.match(type_str.strip())
        if match is None:
            raise ValueError(f"Invalid type string: {type_str!r}")

        attrs = {}
        if match.group("attrs"):
            for pair in match.group("attrs").split(","):
                key, value = pair.split("=")
                attrs[key.strip()] = int(value.strip())

        parsed = cls(
            dtype=match.group("dtype"),
            channel_axis=attrs.get("channel_axis"),
            block_axis=attrs.get("block_axis"),
        )

        # Per-channel/per-block quantization trivially supports the per-tensor case,
        # so a specified axis implicitly also allows the axis-less (per-tensor) qtype.
        options = [parsed]
        if parsed.channel_axis is not None or parsed.block_axis is not None:
            options.append(cls(dtype=parsed.dtype))

        return options


@dataclass(frozen=True)
class Argument:
    name: str
    qtype: "qtype | None"
    role: str | None = None

    _allowed_roles = {
        "weight",
        "bias",
        None,
    }

    def __post_init__(self):
        if self.role not in self._allowed_roles:
            raise ValueError(
                f"Invalid role {self.role} for argument {self.name}. "
                f"Allowed roles are: {self._allowed_roles}"
            )


@dataclass(frozen=True)
class FixedRange:
    arg: Argument
    min: float | None
    max: float | None

    def __post_init__(self):
        if self.min is None and self.max is None:
            raise ValueError("At least one of min or max must be specified")

        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min {self.min} cannot be greater than max {self.max}")


@dataclass(frozen=True)
class EqualRange:
    arg1: Argument
    arg2: Argument


ValueConstraint: TypeAlias = FixedRange | EqualRange


@dataclass(frozen=True)
class OpSignature:
    domain: str
    op_type: str
    inputs: tuple[Argument, ...]
    outputs: tuple[Argument, ...]

    def __post_init__(self):
        weights = [arg for arg in self.inputs if arg.role == "weight"]

        if len(weights) > 1:
            raise ValueError(
                f"Op {self.domain}::{self.op_type} has multiple weight arguments: "
                f"{[arg.name for arg in weights]}"
            )

        biases = [arg for arg in self.inputs if arg.role == "bias"]

        if len(biases) > 1:
            raise ValueError(
                f"Op {self.domain}::{self.op_type} has multiple bias arguments: "
                f"{[arg.name for arg in biases]}"
            )


@dataclass(frozen=True)
class OpSchema:
    signature: OpSignature
    value_constraints: tuple[ValueConstraint, ...] = ()
