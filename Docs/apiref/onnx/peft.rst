.. _apiref-onnx-peft:

##############################
aimet_onnx.peft
##############################

.. note::

    This API is experimental and may change between releases.

``aimet_onnx.peft`` adds one or more LoRA adapters to an ONNX model. The result runs like any
other ONNX model, or can be quantized with AIMET. To quantize a model with adapters, see
:ref:`aimet_onnx.experimental.lora <apiref-onnx-lora>`.

What is a LoRA adapter?
=======================

A LoRA adapter is a small add-on that changes what a model does without changing the model's own
weights. It is trained with the Hugging Face `peft <https://github.com/huggingface/peft>`_ library
and saved as a folder with two files:

.. code-block:: text

    my_adapter/
        adapter_config.json          # settings: rank (r), alpha, which layers it changes
        adapter_model.safetensors    # the adapter's weights

For each layer it changes, an adapter has two small weight matrices, ``A`` and ``B``. The layer's
output becomes:

.. code-block:: text

    output = original_layer(x) + (x @ A) @ B * (alpha / r)

``r`` (the rank) is the size of ``A`` and ``B``. Both ``r`` and ``alpha`` come from
``adapter_config.json``.

Quick start
===========

.. code-block:: python

    import onnx
    from aimet_onnx.peft import load_lora_adapters

    base_model = onnx.load("base_model.onnx")

    lora_model = load_lora_adapters(
        base_model,
        [("math", "adapters/math"), ("code", "adapters/code")],  # (name, adapter folder)
    )

    onnx.save(lora_model, "lora_model.onnx")

``lora_model`` is an ordinary ONNX model: run it in onnxruntime, or pass it to
``aimet_onnx.QuantizationSimModel``. ``base_model`` is not changed.

``peft`` and ``safetensors`` must be installed.

What load_lora_adapters does
============================

For every layer that an adapter changes, ``load_lora_adapters`` adds a small extra path next to
the layer and adds its output to the layer's output.

When you pass **several adapters**, they are all active at the same time, and their effects add
up. If two adapters change the same layer, that layer gets **one** extra path that holds both
adapters' weights side by side. Its size ``r`` is the sum of their sizes. For example, adapters
with ``r=4`` and ``r=2`` on the same layer give one path with ``r=6``.

The names of the base model's inputs, outputs and tensors stay the same. This matters for
quantization: encodings computed on the base model can be loaded into the model with adapters.

To change which adapters are active or their scales, call ``load_lora_adapters`` again on the
**base** model. You can't add adapters to a model that already has them.

.. note::

    The order of the ``adapters`` list decides where each adapter's weights sit inside the shared
    path. If you reuse encodings across runs, always pass the adapters in the same order.

Preparing the base model
========================

``load_lora_adapters`` finds layers by name. An adapter refers to a layer as, for example,
``model.layers.0.self_attn.q_proj``, and ``load_lora_adapters`` looks for an ONNX node with the
same name.

#. **Export from PyTorch.** Both ``torch.onnx.export(..., dynamo=False)`` and ``dynamo=True`` work.
   With ``dynamo=True`` the nodes get generic names, so fix them first:

   .. code-block:: python

       from aimet_onnx.prepare_passes.fix_node_names_in_dynamo_exported_onnx import (
           fix_node_names_pass,
       )

       base_model = fix_node_names_pass(base_model)

#. **Extra wrapper around the model.** If the model was exported inside another module, every
   layer name may start with an extra ``model.`` (``model.model.layers.0...`` instead of
   ``model.layers.0...``). ``load_lora_adapters`` notices this, matches the names anyway, and logs
   a warning saying so.

#. **Smaller copy of the model.** Adapter layers the model doesn't have are skipped with a
   warning, as ``peft`` does in PyTorch. So an adapter trained on a 28-layer model works on a
   2-layer copy for quick tests. If none of the adapter's layers are in the model, it raises an
   error.

#. **Large models.** Load the model without reading its weights into memory:

   .. code-block:: python

       base_model = onnx.load("base_model.onnx", load_external_data=False)

   The new model still points to the same weight files, so save it in the same folder as the base
   model (or copy the weight files next to it).

What is supported
=================

.. list-table::
    :header-rows: 1
    :widths: 15 45 40

    * -
      - Supported
      - Not supported
    * - Layers
      - ``Linear`` (exported as ``MatMul`` or ``Gemm``), ``Conv1d`` / ``Conv2d`` / ``Conv3d``
      - Grouped or dilated convolutions, ``Conv`` with ``auto_pad``
    * - Adapter type
      - LoRA
      - Other ``peft`` methods (IA3, prefix tuning, ...)
    * - Adapter settings
      - ``r``, ``lora_alpha``, ``target_modules``, ``lora_dropout``, ``init_lora_weights`` =
        ``True``, ``False`` or ``"gaussian"``
      - ``bias``, ``use_rslora``, ``use_dora``, ``rank_pattern``, ``alpha_pattern``,
        ``modules_to_save``, and any other setting changed from its default
    * - Weights file
      - ``adapter_model.safetensors``
      - ``adapter_model.bin`` (save with ``safe_serialization=True``)
    * - Number types
      - float32, float16 or float64. The adapter must use the same type as the base model
      - bfloat16 adapters. Nothing is converted for you; save the adapter with ``model.half()``
        for a float16 base

If an adapter uses something that isn't supported, ``load_lora_adapters`` raises an error that
names the setting instead of producing a model that gives wrong results.

Adapters as model inputs
========================

``load_lora_adapters_as_inputs`` adds the same extra paths as ``load_lora_adapters``, but their
``A`` and ``B`` weights and the scale are **model inputs**, so you switch adapters by feeding
different inputs instead of rebuilding the model. It raises if the adapters have different layers,
ranks or number types. See :ref:`One set of encodings for every adapter (LoRA v1)
<apiref-onnx-lora-v1>` for how to quantize such a model.

Examples
========

The tests show the complete flows, with every step written out and checked.

``TrainingExtensions/onnx/test/python/test_onnx_peft.py``:

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - Test
      - What it shows
    * - ``TestLoadLoraAdapters.test_worked_example_two_adapters``
      - **The complete example.** Export a model, save two adapters with ``peft``, call
        ``load_lora_adapters``, run the result, and check it against ``peft`` in PyTorch
    * - ``TestLoadLoraAdapters.test_each_layer_kind_matches_torch_peft``
      - Layer forms beyond the worked example (``Gemm``, ``Conv1d``)
    * - ``TestLoadLoraAdapters.test_quantsim_quantizes_the_lora_tensors``
      - Putting the result into ``QuantizationSimModel``
    * - ``TestLoadLoraAdapters.test_adapter_layers_missing_from_base_are_skipped``
      - An adapter trained on more layers than the model has
    * - ``TestLoadLoraAdapters.test_pad_rank``
      - ``pad_rank=True``: every path gets the same size, and the output doesn't change
    * - ``TestLoadLoraAdapters.test_external_data_base``
      - Large models with weights in separate files

Common errors
=============

.. list-table::
    :header-rows: 1
    :widths: 35 35 30

    * - Error message
      - What it means
      - What to do
    * - ``No MatMul/Gemm/Conv with a static weight matches module path '...'``
      - None of the adapter's layers are in the model. The message lists the closest names found
      - Check that the adapter was trained on this model
    * - ``... call fix_node_names_pass first``
      - The model was exported with ``dynamo=True`` and its node names weren't fixed
      - Run ``fix_node_names_pass`` on the model first
    * - ``LoraConfig.<setting>=... is not supported``
      - The adapter uses a setting that isn't supported
      - See `What is supported`_
    * - ``... re-save the adapter in the base model's dtype``
      - The adapter and the model use different number types (e.g. float32 and float16)
      - Save the adapter in the model's type
    * - ``... doesn't fit the base layer``
      - The adapter's weights are the wrong size for the layer
      - Check that the adapter was trained on this model
    * - ``The model already contains LoRA branches``
      - The model passed in already has adapters
      - Call ``load_lora_adapters`` on the original base model

API
===

..
  # start-after

.. autofunction:: aimet_onnx.peft.load_lora_adapters

.. autofunction:: aimet_onnx.peft.load_lora_adapters_as_inputs

.. autofunction:: aimet_onnx.peft.get_lora_tensor_names

.. autoclass:: aimet_onnx.peft.LoraTensorNames
    :members:

..
  # end-before
