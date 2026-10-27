# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import torch
from torch._C import _jit_get_all_schemas

from aimet_torch.experimental.export.qnn import _qualcomm_aten_ir_ops
from aimet_torch.experimental.backends.schema import (
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


# fmt: off
_tensor_roles = {
    #   op_type                                  | arg_name  | role
    # -------------------------------------------------------------------
    ("aten::conv2d",                               "weight"): "weight",
    ("aten::conv2d",                               "bias"  ): "bias",
    ("aten::_native_batch_norm_legit_no_training", "weight"): "weight",
    ("aten::_native_batch_norm_legit_no_training", "bias"  ): "bias",
}
# fmt: on


def _is_tensor_type(arg: torch.Argument) -> bool:
    type_str = str(arg.type)
    return type_str in ("Tensor", "List[Tensor]", "Optional[Tensor]")


def _concretize_wildcard_type_constraint(
    aten_schema: torch.FunctionSchema,
    type_constraint: dict,
) -> dict:
    """Replace wildcard argument names in the type constraint with the actual argument names from the ATen schema."""
    if "*" not in type_constraint:
        return type_constraint

    wildcard_value = type_constraint["*"]
    concrete_type_constraint = {
        arg.name: wildcard_value if _is_tensor_type(arg) else None
        for arg in aten_schema.arguments
    }
    concrete_type_constraint |= {
        f"$ret[{i}]": wildcard_value if _is_tensor_type(ret) else None
        for i, ret in enumerate(aten_schema.returns)
    }
    return concrete_type_constraint


def _validate_against_aten_schema(
    aten_schema: torch.FunctionSchema,
    type_constraint: dict,
) -> None:
    if "*" in type_constraint:
        if len(type_constraint) > 1:
            raise ValueError(
                f"{_fullname(aten_schema)}: type constraint {type_constraint} contains a wildcard '*' "
                "and must not contain any other argument names"
            )
        return

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
        is_tensor_type = _is_tensor_type(arg)
        qtype_specified = type_constraint.get(arg.name)

        if not is_tensor_type and qtype_specified:
            raise ValueError(
                f"{_fullname(aten_schema)}: tensor {arg.name!r} (type: {arg.type}) is not a Tensor argument "
                f"and must not declare a qtype, got {type_constraint.get(arg.name)!r}"
            )


def parse_aten_op(op_entry: dict) -> dict[str, set[OpSchema]]:
    op_type_fullname = op_entry["op_type"]

    # Wildcard: Match with all ATen ops
    if op_type_fullname == "*":
        schemas = {}
        for schema in _all_qualcomm_aten_function_schemas.values():
            schemas |= parse_aten_op({**op_entry, "op_type": _fullname(schema)})
        return schemas
    else:
        schemas = {op_type_fullname: set()}

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
        type_constraint = _concretize_wildcard_type_constraint(
            torch_schema, type_constraint
        )

        args = {
            arg_name: Argument(
                name=arg_name,
                qtype=(
                    qtype.from_str(type_str)
                    if (type_str := type_constraint.get(arg_name))
                    else None
                ),
                role=_tensor_roles.get((op_type_fullname, arg_name)),
            )
            for arg_name in itertools.chain(input_names, output_names)
        }

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
        schemas[op_type_fullname].add(
            OpSchema(signature=signature, value_constraints=value_constraints)
        )

    return schemas
