:orphan:

.. _build-from-source:

####################
Building from source
####################

AIMET provides a single script, ``scripts/all/build_and_test.py``, to build wheels, run unit
tests, and build documentation across Linux, Windows, and macOS. It manages virtual environment
creation, system dependency installation, and CMake/scikit-build-core configuration so you don't
need to set any of that up by hand.

Prerequisites
==============

- Python 3.10 or newer.
- `uv <https://docs.astral.sh/uv/getting-started/installation/>`_: the script installs it
  automatically if missing (``--task install_deps``), or install it yourself with:

  .. code-block:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh

- (Optional, GPU builds only) NVIDIA CUDA Toolkit. See NVIDIA's
  `installation guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_.

Quick start
===========

.. code-block:: bash

    # cd to the AIMET root directory
    cd aimet/

    # Build aimet-torch (CPU)
    python3 scripts/all/build_and_test.py --task build --package aimet_torch

    # Build aimet-torch (GPU/CUDA)
    python3 scripts/all/build_and_test.py --task build --package aimet_torch --gpu

    # Build aimet-onnx (CPU)
    python3 scripts/all/build_and_test.py --task build --package aimet_onnx

    # Build aimet-onnx (GPU/CUDA)
    python3 scripts/all/build_and_test.py --task build --package aimet_onnx --gpu

    # Build documentation (wheelhouse/Docs/index.html)
    python3 scripts/all/build_and_test.py --task build --package docs

The script creates and reuses a venv at ``aimet-dev/`` by default (override with ``--venv``), and
installs system dependencies on first run (``install_deps``, a dependency of ``build``) -- no
separate setup step is required.

Editable installs for local development
=========================================

.. code-block:: bash

    python3 scripts/all/build_and_test.py --task build --package aimet_torch --gpu -e

Running tests
=============

``test`` builds the package first (if needed) and then runs its unit test suite:

.. code-block:: bash

    # Build (if needed) and run aimet-onnx unit tests
    python3 scripts/all/build_and_test.py --task test --package aimet_onnx

    # Run tests only, skipping the build step
    python3 scripts/all/build_and_test.py --task test --package aimet_torch --only

    # Force a rebuild before testing
    python3 scripts/all/build_and_test.py --task test --package aimet_torch --force-rebuild

Common options
==============

.. list-table::
   :widths: 20 60
   :header-rows: 1

   * - Option
     - Description
   * - ``--package``
     - ``aimet_torch``, ``aimet_onnx``, or ``docs``.
   * - ``--gpu``
     - Enable GPU/CUDA support (default: CPU only).
   * - ``--editable``, ``-e``
     - Install in editable mode for local development.
   * - ``--force-rebuild``
     - Force a rebuild/reinstall even if the package is already installed.
   * - ``--parallel``, ``-n``
     - Number of parallel pytest workers (default: 8).
   * - ``--venv``
     - Virtualenv directory to create/use (default: ``aimet-dev``).
   * - ``--only``
     - Run only the requested task, skipping its dependencies.
   * - ``--dry-run``
     - Print the execution plan without running it.

Run ``python3 scripts/all/build_and_test.py --task list`` to see every available task, or
``--help`` for the full option reference.
