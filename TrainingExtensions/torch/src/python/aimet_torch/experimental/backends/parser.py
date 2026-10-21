# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import torch
from torch._C import _jit_get_all_schemas

from aimet_torch.experimental.export.qnn import _qualcomm_aten_ir_ops
from aimet_torch.common.experimental.backends.schema import (
    qtype,
    OpSchema,
    OpSignature,
    Argument,
    FixedRange,
    EqualRange,
)


def _fullname(schema: torch.FunctionSchema) -> str:
    if schema.overload_name:
        return f"{schema.name}.{schema.overload_name}"
    else:
        return schema.name


_all_qualcomm_aten_function_schemas: dict[str, torch.FunctionSchema] = {}
_qualcomm_aten_ir_op_names = {op.name() for op in _qualcomm_aten_ir_ops()}
_all_qualcomm_aten_function_schemas = {
    _fullname(s): s
    for s in _jit_get_all_schemas()  # pylint: disable=not-an-iterable
    if _fullname(s) in _qualcomm_aten_ir_op_names
}
del _qualcomm_aten_ir_op_names, _qualcomm_aten_ir_ops


def _validate_against_aten_schema(
    aten_schema: torch.FunctionSchema,
    type_constraint: dict,
) -> None:
    expected_names = [
        *(arg.name for arg in aten_schema.arguments),
        *(f"$ret[{i}]" for i, _ in enumerate(aten_schema.returns)),
    ]
    actual_names = set(type_constraint.keys())

    if actual_names != set(expected_names):
        raise ValueError(
            f"{_fullname(aten_schema)}: argument names {sorted(actual_names)} do not match "
            f"torch FunctionSchema {expected_names}"
        )

    for arg in itertools.chain(aten_schema.arguments, aten_schema.returns):
        type_str = str(arg.type)
        is_tensor_type = type_str in ("Tensor", "List[Tensor]", "Optional[Tensor]")
        is_wildcard = type_constraint.get(arg.name, "n/a").strip() == "n/a"

        if not is_tensor_type and not is_wildcard:
            raise ValueError(
                f"{_fullname(aten_schema)}: tensor {arg.name!r} (type: {type_str}) is not a Tensor argument "
                f"and must not declare a qtype, got {type_constraint.get(arg.name)!r}"
            )


def parse_aten_op(op_entry: dict) -> set[OpSchema]:
    schemas = set()
    op_type_fullname = op_entry["op_type"]

    torch_schema = _all_qualcomm_aten_function_schemas.get(op_type_fullname)
    if torch_schema is None:
        raise ValueError(f"Unknown ATen op_type: {op_type_fullname!r}")

    input_names = [arg.name for arg in torch_schema.arguments]
    output_names = [f"$ret[{i}]" for i, _ in enumerate(torch_schema.returns)]

    value_constraints_config = op_entry.get("value_contraints") or {}
    fix_constraints = value_constraints_config.get("fix") or []
    tie_constraints = value_constraints_config.get("tie") or []

    for type_constraint in op_entry["type_constraints"]:
        _validate_against_aten_schema(torch_schema, type_constraint)

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

            domain, op_type = op_type_fullname.split("::")
            inputs = tuple(args[name] for name in input_names)
            outputs = tuple(args[name] for name in output_names)
            signature = OpSignature(
                domain=domain,
                op_type=op_type,
                inputs=inputs,
                outputs=outputs,
            )
            schemas.add(
                OpSchema(signature=signature, value_constraints=value_constraints)
            )

    return schemas
