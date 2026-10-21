# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import onnx
from aimet_onnx.common.experimental.backends.schema import (
    qtype,
    Argument,
    FixedRange,
    EqualRange,
    OpSignature,
    OpSchema,
)
from aimet_onnx.common.onnx._utils import _all_op_schemas


_FLOAT_ONNX_TYPE_STRS = {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}


def _validate_against_onnx_schema(
    onnx_schema: onnx.defs.OpSchema,
    type_constraint: dict,
) -> None:
    expected_names = [
        *(param.name for param in onnx_schema.inputs),
        *(param.name for param in onnx_schema.outputs),
    ]
    actual_names = set(type_constraint.keys())

    if actual_names != set(expected_names):
        raise ValueError(
            f"{onnx_schema.name}: argument names {sorted(actual_names)} do not match "
            f"ONNX schema {expected_names}"
        )

    allowed_type_strs_by_type_param = {
        tc.type_param_str: set(tc.allowed_type_strs)
        for tc in onnx_schema.type_constraints
    }

    for param in itertools.chain(onnx_schema.inputs, onnx_schema.outputs):
        # param.type_str is either a type variable (e.g. "T", resolved via type_constraints)
        # or a concrete type (e.g. "tensor(int64)") directly.
        allowed_type_strs = allowed_type_strs_by_type_param.get(
            param.type_str, {param.type_str}
        )
        can_be_float = bool(allowed_type_strs & _FLOAT_ONNX_TYPE_STRS)
        is_wildcard = type_constraint.get(param.name, "n/a").strip() == "n/a"

        if can_be_float and is_wildcard:
            raise ValueError(
                f"{onnx_schema.name}: tensor {param.name!r} can be FLOAT/FLOAT16/BFLOAT16 and "
                f"must declare an explicit qtype, got a wildcard ('n/a')"
            )

        if not can_be_float and not is_wildcard:
            raise ValueError(
                f"{onnx_schema.name}: tensor {param.name!r} cannot be FLOAT/FLOAT16/BFLOAT16 "
                f"and must not declare a qtype, got {type_constraint.get(param.name)!r}"
            )


def parse_onnx_op(op_entry: dict) -> set[OpSchema]:
    schemas = set()
    op_type = op_entry["op_type"]

    onnx_schema = _all_op_schemas.get(op_type)
    if onnx_schema is None:
        raise ValueError(f"Unknown ONNX op_type: {op_type!r}")

    # Input/output names and their order are taken from the ONNX schema, not from the
    # (unordered) argument list in the yaml's type_constraints.
    input_names = [param.name for param in onnx_schema.inputs]
    output_names = [param.name for param in onnx_schema.outputs]

    value_constraints_config = op_entry.get("value_contraints") or {}
    fix_constraints = value_constraints_config.get("fix") or []
    tie_constraints = value_constraints_config.get("tie") or []

    for type_constraint in op_entry["type_constraints"]:
        _validate_against_onnx_schema(onnx_schema, type_constraint)

        per_arg_options = [
            [(arg_name, None)]
            if type_constraint[arg_name].strip() == "n/a"
            else [(arg_name, qt) for qt in qtype.from_str(type_constraint[arg_name])]
            for arg_name in input_names + output_names
        ]

        for combo in itertools.product(*per_arg_options):
            args = {name: Argument(name=name, qtype=qt) for name, qt in combo}

            value_constraints = tuple(
                FixedRange(
                    arg=args[fix_arg_name],
                    min=bounds.get("min"),
                    max=bounds.get("max"),
                )
                for fix_entry in fix_constraints
                for fix_arg_name, bounds in fix_entry.items()
            ) + tuple(
                EqualRange(arg1=args[arg1_name], arg2=args[arg2_name])
                for tie_group in tie_constraints
                for arg1_name, arg2_name in itertools.combinations(tie_group, 2)
            )

            signature = OpSignature(
                domain="onnx",
                op_type=op_type,
                inputs=tuple(args[name] for name in input_names),
                outputs=tuple(args[name] for name in output_names),
            )
            schemas.add(
                OpSchema(signature=signature, value_constraints=value_constraints)
            )

    return schemas
