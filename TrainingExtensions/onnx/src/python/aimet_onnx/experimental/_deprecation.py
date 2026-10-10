# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for packages that have graduated out of aimet_onnx.experimental"""

import importlib
import pkgutil
import sys
import warnings
from types import ModuleType

from aimet_onnx.common.utils import AimetLogger  # pylint: disable=import-error, no-name-in-module

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def _alias_deprecated_package(
    old_name: str, new_name: str, removal_version: str
) -> ModuleType:
    """
    Make ``old_name`` and all of its submodules aliases of ``new_name``.

    Every submodule of ``new_name`` is imported and registered in ``sys.modules`` under
    the old prefix, so deep imports (``from <old_name>.foo import bar``) and
    ``mock.patch("<old_name>.foo.bar")`` keep working, and both paths resolve to the
    same module objects.

    :param old_name: Fully qualified name of the deprecated package
    :param new_name: Fully qualified name of the package that replaces it
    :param removal_version: AIMET release in which ``old_name`` will be removed
    :return: The new package, to be installed as ``sys.modules[old_name]``
    """
    msg = (
        f"`{old_name}` has been moved to `{new_name}` and will be removed in "
        f"AIMET {removal_version}. Import from `{new_name}` instead."
    )
    # stacklevel=3 points at the importing module; importlib frames are skipped
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    _logger.error(msg)

    new_pkg = importlib.import_module(new_name)
    for info in pkgutil.walk_packages(new_pkg.__path__, prefix=f"{new_name}."):
        importlib.import_module(info.name)

    for name, module in list(sys.modules.items()):
        if name == new_name or name.startswith(f"{new_name}."):
            sys.modules[old_name + name[len(new_name) :]] = module

    return new_pkg
