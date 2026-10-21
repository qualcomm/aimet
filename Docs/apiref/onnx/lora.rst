.. _apiref-onnx-lora:

##############################
aimet_onnx.experimental.lora
##############################

.. note::

    This API is experimental and may change between releases.

``aimet_onnx.experimental.lora`` quantizes ONNX models that have LoRA adapters. Adapters are added
to the model with :ref:`aimet_onnx.peft <apiref-onnx-peft>`, which also explains what an adapter
is, how to prepare the base model and which adapters are supported.

Quantizing a model with adapters
================================

Build the quantization simulation **after** calling ``aimet_onnx.peft.load_lora_adapters``. AIMET decides which
tensors to quantize when the simulation is created, so the adapter weights are only quantized if
they are already in the model:

.. code-block:: python

    from aimet_onnx import QuantizationSimModel

    lora_model = load_lora_adapters(base_model, adapters)
    sim = QuantizationSimModel(lora_model, ...)
    sim.compute_encodings(calibration_data)

Quantize the base once, then each concurrency
---------------------------------------------

A *concurrency* is a set of adapters that are active together. The usual flow keeps the base
model's encodings fixed and computes encodings only for the adapter parts, once per concurrency:

#. **Base model.** Build a sim on the base model, run any techniques that change its weights (such
   as AdaScale), calibrate, and export it with ``sim.export(...)``. This writes the float model,
   with the updated weights, and its encodings.
#. **Add the adapters** to the **exported** model with ``load_lora_adapters``. The base tensor
   names don't change, so the base encodings still match.
#. **Build a second sim** on the result, with the same settings as the base sim.
#. **Share the base output quantizers.** Make each Add that merges an adapter path reuse its base
   layer's output quantizer instead of needing its own. ``lora_tensors.merges`` lists the
   ``(base layer output, merged output)`` pairs. Neither the Add nor the ops that follow it then
   have encodings to calibrate and store per concurrency. Then load the base encodings. The order
   matters: the Add's quantizer carries settings AIMET derives from the tensor's consumers (for
   example, attention), and they have to be copied onto the base quantizer before the load, or
   the loaded encodings won't match.
#. **Configure the adapter quantizers, calibrate and export.** Loading the base encodings with
   ``allow_overwrite=False`` freezes the base quantizers, so calibration only computes the
   adapter encodings.

The helpers take the adapter tensor names that ``get_lora_tensor_names`` returns. Call it on the
model ``load_lora_adapters`` returned, not on the sim's graph; the names are the same in the sim:

.. code-block:: python

    from aimet_onnx.quantsim import load_encodings_to_sim
    from aimet_onnx.experimental.lora import (
        get_lora_encodings,
        set_lora_bitwidth,
    )
    from aimet_onnx.peft import get_lora_tensor_names, load_lora_adapters

    lora_tensors = get_lora_tensor_names(lora_model)
    sim = QuantizationSimModel(lora_model, ...)  # same settings as the base sim

    # each merged output reuses its base layer's output quantizer
    shared = {}
    for base_output, merged_output in lora_tensors.merges:
        base_qtzr = sim.qc_quantize_op_dict[base_output]
        base_qtzr._merge_constraints(sim.qc_quantize_op_dict[merged_output])
        shared[merged_output] = base_qtzr
    sim.set_quantizers(shared)

    load_encodings_to_sim(sim, "base.encodings", strict=False, allow_overwrite=False,
                          disable_missing_quantizers=False)
    set_lora_bitwidth(sim, lora_tensors, param_type="int8", activation_type="int16")
    sim.compute_encodings(adapter_calibration_data)
    lora_encodings = get_lora_encodings(sim, lora_tensors)  # this concurrency's adapter encodings

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Helper
      - What it does
    * - ``set_lora_bitwidth``
      - Sets the bit widths of the adapter weights and activations, and turns off the scale
        quantizers
    * - ``unfreeze_lora_quantizers``
      - Unfreezes the adapter quantizers and clears them, to calibrate them again
    * - ``get_lora_encodings`` / ``set_lora_encodings``
      - Saves a concurrency's adapter encodings and loads them back into a new sim (frozen), so you
        don't calibrate again
    * - ``disable_lora_quantizers`` / ``enable_lora_calibration``
      - Turns the adapter quantizers off (e.g. while a technique runs on the base layers) and back
        on at the adapter bit widths
    * - ``get_lora_node_names_for_exclusion``
      - The adapter-path node names, to exclude from techniques such as SeqMSE or AdaScale


.. _apiref-onnx-lora-v1:

One set of encodings for every adapter (LoRA v1)
------------------------------------------------

When only one adapter runs at a time and every adapter is compatible (same layers and rank), one
model and one set of encodings can serve them all. ``load_lora_adapters_as_inputs`` adds the same
extra paths as ``load_lora_adapters``, but their ``A`` and ``B`` weights and the scale are **model
inputs**. You switch adapters by feeding different inputs, not by rebuilding the model:

.. code-block:: python

    import random
    import numpy as np
    from aimet_onnx.experimental.lora import set_lora_bitwidth
    from aimet_onnx.peft import get_lora_tensor_names, load_lora_adapters_as_inputs

    adapters = [("math", "adapters/math"), ("code", "adapters/code")]
    # checks the adapters are compatible; the first one's layers and rank build the model.
    # adapter_inputs: adapter name -> its values for the LoRA inputs
    lora_model, adapter_inputs = load_lora_adapters_as_inputs(base_model, adapters)

    # zeros run like the base
    adapter_inputs["base"] = {k: np.zeros_like(v) for k, v in adapter_inputs["math"].items()}

    sim = QuantizationSimModel(lora_model, param_type="int4", activation_type="int16")
    set_lora_bitwidth(sim, get_lora_tensor_names(lora_model), param_type="int8",
                      activation_type="int16")

    def calibrate(session):
        for batch in calibration_data:
            session.run(None, {**batch, **adapter_inputs[random.choice(list(adapter_inputs))]})

    sim.compute_encodings(calibrate)
    sim.export("out", "model")  # one model and one encodings file for every adapter

* ``A`` and ``B`` keep the names they have with ``load_lora_adapters``. Every path shares one
  scale input, ``lora_scale``, which each adapter's inputs set to its ``alpha / r``.
* The adapter weights are model inputs, so AIMET quantizes them like activations: their ranges
  cover every adapter seen during calibration. ``set_lora_bitwidth`` sets their bit width
  (``param_type``). Like every scale, ``lora_scale`` is not quantized.
* Every run needs every adapter input. Feed ``adapter_inputs["base"]`` to run without an
  adapter, for example for techniques that tune the base weights such as SeqMSE. Add the adapter
  inputs to each batch after batching, so they don't get a batch dimension.
* ``load_lora_adapters_as_inputs`` raises if the adapters have different layers, ranks or number
  types. To add an adapter later, call it again with the full list.

Examples
========

``TrainingExtensions/onnx/test/python/test_lora.py`` quantizes a small Llama with three adapters:

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - Test
      - What it shows
    * - ``test_float_lora_model_matches_torch_peft``
      - Two different adapters active at once match ``peft`` in PyTorch
    * - ``test_lora_v1_one_encoding_set_for_all_adapters``
      - One set of encodings shared by every adapter; adapters are model inputs, switched through
        the adapter inputs
    * - ``test_lora_v2_full_encoding_set_per_adapter``
      - A separate, complete set of encodings for each adapter
    * - ``test_lora_v3_frozen_base_lora_encodings_per_concurrency``
      - Base encodings computed once and frozen; adapter encodings computed per concurrency

``TestLoraAdapterQuantization`` in the same file has one short test per helper.


API
===

..
  # start-after

.. autofunction:: aimet_onnx.experimental.lora.set_lora_bitwidth

.. autofunction:: aimet_onnx.experimental.lora.unfreeze_lora_quantizers

.. autofunction:: aimet_onnx.experimental.lora.get_lora_encodings

.. autofunction:: aimet_onnx.experimental.lora.set_lora_encodings

.. autofunction:: aimet_onnx.experimental.lora.disable_lora_quantizers

.. autofunction:: aimet_onnx.experimental.lora.enable_lora_calibration

.. autofunction:: aimet_onnx.experimental.lora.get_lora_node_names_for_exclusion

..
  # end-before
