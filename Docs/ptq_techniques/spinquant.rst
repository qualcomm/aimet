.. _ptq-spinquant:

#########
SpinQuant
#########

.. note::
   This feature is currently experimental. The API may change in the future.

Context
=======

SpinQuant (`arXiv:2405.16406 <https://arxiv.org/pdf/2405.16406>`_) is a post-training quantization
technique that reduces activation outliers by inserting orthogonal Hadamard rotations at key points
in the model. Because the rotations are absorbed into adjacent weight matrices, the final model
architecture is unchanged.

AIMET implements **R1** and **R2 rotations** (fixed Hadamard, no optimization).

* **R1** fuses RMSNorm scale weights into downstream linear layers, then applies a Hadamard
  rotation across the residual stream to reduce outliers in Q/K/V/O and gate/up/down projections.
  Enabled by default.
* **R2** applies a per-head Hadamard rotation across the attention head dimension, absorbed into
  V and O projections to reduce outliers in the attention output. Disabled by default; enable via
  ``enable_r2=True``. Not supported on architectures with fused QKV projections (e.g. Phi3).

The ONNX ``apply_spinquant`` API exposes these as boolean flags ``enable_r1`` and ``enable_r2``.

.. list-table:: Supported architectures
   :widths: 30 20 20
   :header-rows: 1

   * - Model family
     - PyTorch
     - ONNX
   * - ``LlamaForCausalLM``
     - ✓
     - ✓
   * - ``Qwen2ForCausalLM``, ``Qwen3ForCausalLM``
     - ✓
     - ✓
   * - ``MistralForCausalLM``
     - ✓
     - ✓
   * - ``Phi3ForCausalLM``
     - ✓
     - ✓
   * - ``Qwen2.5-VL``, ``Qwen3-VL``
     - ✓
     - ✓

.. note::
   Support for additional model families is added continuously as new architectures are validated.
   See the :ref:`release notes <rn-index>` for the latest additions.

Workflow
========

Prerequisites
-------------

To use SpinQuant, you need:

- A pre-trained model loaded from HuggingFace.
- **ONNX only**: the model must be exported to ONNX — `Step 2`_ handles this.

.. note::
   For a complete working example, see
   `Examples/torch/quantize.py <https://github.com/qualcomm/aimet/blob/develop/Examples/torch/quantize.py>`_
   or
   `Examples/onnx/quantize.py <https://github.com/qualcomm/aimet/blob/develop/Examples/onnx/quantize.py>`_
   (run with ``--recipe pcq_spinquant``).

Procedure
---------

.. _Step 1:

Step 1: Load model
~~~~~~~~~~~~~~~~~~

Load the HuggingFace model and wrap it with ``ONNXExportableModuleWithCache`` to enable JIT tracing
with a static graph.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_spinquant.py
            :language: python
            :start-after: # [model-setup]
            :end-before: # End of [model-setup]

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_spinquant.py
            :language: python
            :start-after: # [model-setup]
            :end-before: # End of [model-setup]

.. _Step 2:

Step 2: Create QuantizationSimModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a :ref:`QuantizationSimModel <quantsim-index>` with the desired quantization configuration.

**ONNX only**: this step instead exports the model to ONNX and analyzes its decoder-stack structure
with :ref:`analyze_llm_topology <apiref-onnx-spinquant>`. The sim is created in `Step 3`_, *after*
the rotation, because ONNX SpinQuant rewrites the float graph and R3 inserts new ops that only a sim
built on the rotated graph can wrap in quantizers. A topology describes the model, not the sim, so
derive it once here; `Step 3`_ hands it to SpinQuant, which uses it to place every rotation.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_spinquant.py
            :language: python
            :start-after: # [create-sim]
            :end-before: # End of [create-sim]

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_spinquant.py
            :language: python
            :start-after: # [export-onnx]
            :end-before: # End of [export-onnx]

.. _Step 3:

Step 3: Apply SpinQuant
~~~~~~~~~~~~~~~~~~~~~~~~

Apply SpinQuant to the model. This fuses RMSNorm scale weights into downstream linear layers and
applies the R1 Hadamard rotation to all weight matrices in-place.

.. important::
   ``apply_spinquant`` must be called **before** ``compute_encodings``. The rotation modifies float
   weight initializers; ``compute_encodings`` must run afterward to calibrate quantizer scales on
   the rotated weights. For ONNX it must also run **before** the sim is created, so the ONNX tab
   below rotates the float model and then builds the sim on the rotated graph.

.. note::
   **ONNX only**: SpinQuant is told where each rotation goes, rather than finding out itself. Pass
   the ``LlmTopology`` analyzed in `Step 2`_ as ``topology`` — locating model structure belongs to
   AIMET's topology analysis, and SpinQuant only rotates what it reports. The argument is optional
   today: omitting it analyzes the topology internally and warns. Passing it is the recommended
   usage, and may become required in a future release.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_spinquant.py
            :language: python
            :start-after: # [spinquant-apply]
            :end-before: # End of [spinquant-apply]

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_spinquant.py
            :language: python
            :start-after: # [spinquant-apply]
            :end-before: # End of [spinquant-apply]

        .. literalinclude:: ../snippets/onnx/apply_spinquant.py
            :language: python
            :start-after: # [create-sim]
            :end-before: # End of [create-sim]

Step 4: Compute activation encodings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibrate activation quantizers by running the model through a representative dataset.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_spinquant.py
            :language: python
            :start-after: # [compute-encodings]
            :end-before: # End of [compute-encodings]

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_spinquant.py
            :language: python
            :start-after: # [compute-encodings]
            :end-before: # End of [compute-encodings]

After completing these steps, export the quantized model:

- **PyTorch**: ``quantsim.export(...)``
- **ONNX**: ``quantsim.export(...)``

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/spinquant.rst
            :start-after: # start-after

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../apiref/onnx/spinquant.rst
            :start-after: # start-after
