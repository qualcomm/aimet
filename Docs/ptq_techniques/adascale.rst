.. _ptq-adascale:

#########
AdaScale
#########

.. note::
   This feature is currently experimental. The API may change in the future.

Context
=======

AdaScale is a post-training quantization (PTQ) technique that recovers accuracy lost during INT4
weight quantization without any fine-tuning. It works by learning optimal per-weight scaling
parameters through Blockwise Knowledge Distillation (BKD): the quantized output of each transformer
block is optimized to match its FP32 equivalent until the two converge.

AdaScale is based on `FlexRound <https://arxiv.org/abs/2306.00317>`_ and integrates learnable weight
clipping from `OmniQuant <https://arxiv.org/abs/2308.13137>`_.

A *block* is a transformer decoder layer that accepts a single activation tensor as input and
produces a single activation tensor as output. For all supported model families, decoder layers are
contiguous by default so no special configuration is required.

Workflow
========

Prerequisites
-------------

To use AdaScale, you need:

- A pre-trained model loaded from HuggingFace. Supported model families:
  ``Llama``, ``Qwen2``, ``Mistral``, ``Phi3``, ``Qwen3``.
  PyTorch additionally supports vision-language models: ``Qwen2.5-VL``, ``Qwen3-VL``.
- **ONNX only**: the model must be exported to ONNX with the input naming convention required by
  AdaScale — `Step 2`_ handles this.

.. note::
   For a complete working example including all steps below, see
   `Examples/torch/quantize.py <https://github.com/qualcomm/aimet/blob/develop/Examples/torch/quantize.py>`_
   or
   `Examples/onnx/quantize.py <https://github.com/qualcomm/aimet/blob/develop/Examples/onnx/quantize.py>`_
   (run with ``--recipe pcq_spinquant_adascale``).

.. _example-script:

Procedure
---------

.. _Step 1:

Step 1: Load model
~~~~~~~~~~~~~~~~~~

Load the HuggingFace model and wrap it with ``ONNXExportableModuleWithCache`` to enable JIT tracing
with a static graph — required for both the PyTorch and ONNX workflows.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adascale.py
            :language: python
            :start-after: # [model-setup]
            :end-before: # End of [model-setup]

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_adascale.py
            :language: python
            :start-after: # [model-setup]
            :end-before: # End of [model-setup]

.. _Step 2:

Step 2: Create QuantizationSimModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a :ref:`QuantizationSimModel <quantsim-index>` with the desired quantization configuration.
For ONNX, this step also exports the model to ONNX first — the ONNX tab includes the
``torch.onnx.export`` call that produces the correctly named inputs required by AdaScale.

The ONNX tab additionally analyzes the model's decoder-stack structure with
:ref:`analyze_llm_topology <apiref-onnx-adascale>`, on the **float** model and before the sim is
created. A topology describes the model, not the sim, so derive it once here; `Step 3`_ hands it to
AdaScale, which uses it to locate the decoder blocks.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adascale.py
            :language: python
            :start-after: # [create-sim]
            :end-before: # End of [create-sim]

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_adascale.py
            :language: python
            :start-after: # [create-sim]
            :end-before: # End of [create-sim]

.. _Step 3:

Step 3: Apply AdaScale
~~~~~~~~~~~~~~~~~~~~~~~

Apply AdaScale to find optimal weight quantization encodings for each supported block.

``_prefill_inputs`` collects the full model inputs (including KV cache tensors) from the calibration
dataset; AdaScale derives per-block activations internally and uses them for BKD.

.. note::
   **ONNX only**: AdaScale is told where the decoder blocks are, rather than finding them itself.
   Pass the ``LlmTopology`` analyzed from the float model in `Step 2`_ as ``topology`` — locating
   model structure belongs to AIMET's topology analysis, and AdaScale only optimizes the blocks it
   reports. The argument is optional today: omitting it discovers the topology internally from the
   sim's graph and warns. Passing it is the recommended usage, and may become required in a future
   release.
This is the most time-consuming step; expect 2–6 hours depending on model size and iteration count
(see the timing column in :ref:`Quantization recipes for LLMs <quantization-genai-recipe>`).

``ADASCALE_NUM_BATCHES`` and ``ADASCALE_NUM_ITERATIONS`` trade accuracy against runtime. The values
below are validated per model size; if your model is not listed, start from the closest row.
See :ref:`Quantization recipes for LLMs <quantization-genai-recipe>` for full results.

.. list-table::
   :widths: 35 20 20
   :header-rows: 1

   * - Model
     - ``num_batches``
     - ``num_iterations``
   * - Qwen/Qwen2.5-0.5B-Instruct
     - 128
     - 2048
   * - meta-llama/Llama-3.2-1B-Instruct
     - 128
     - 2048
   * - Qwen/Qwen2.5-1.5B-Instruct
     - 128
     - 1024
   * - meta-llama/Llama-3.2-3B-Instruct
     - 128
     - 1024
   * - Qwen/Qwen3-4B
     - 128
     - 512
   * - microsoft/Phi-3.5-mini-instruct
     - 32
     - 256

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adascale.py
            :language: python
            :start-after: # [adascale-apply]
            :end-before: # End of [adascale-apply]

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_adascale.py
            :language: python
            :start-after: # [adascale-apply]
            :end-before: # End of [adascale-apply]

Step 4: Compute activation encodings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AdaScale optimizes weight encodings only. This step calibrates the remaining activation quantizers
by running the model through a representative dataset.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adascale.py
            :language: python
            :start-after: # [compute-encodings]
            :end-before: # End of [compute-encodings]

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_adascale.py
            :language: python
            :start-after: # [compute-encodings]
            :end-before: # End of [compute-encodings]

After completing these steps, export the quantized model:

- **PyTorch**: ``quantsim.onnx.export(...)``
- **ONNX**: ``quantsim.export(...)``

See
`Examples/torch/quantize.py <https://github.com/qualcomm/aimet/blob/develop/Examples/torch/quantize.py>`_
or
`Examples/onnx/quantize.py <https://github.com/qualcomm/aimet/blob/develop/Examples/onnx/quantize.py>`_
for the export invocation.

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/adascale.rst
            :start-after: # start-after

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../apiref/onnx/adascale.rst
            :start-after: # start-after
