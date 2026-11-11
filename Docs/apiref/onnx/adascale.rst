.. _apiref-onnx-adascale:

##########################################
aimet_onnx.adascale
##########################################

..
  # start-after

.. rubric:: Top level APIs

..
  # start-after toplevel

.. autofunction:: aimet_onnx.adascale.adascale_optimizer.apply_adascale
    :no-index:

``adascale_model_config`` is an ``AdaScaleModelConfig`` tuning the per-weight learning rates used
during optimization. It is not meant to be constructed directly — look it up by model family from
``adascale_model_config_dict`` instead::

    from aimet_onnx.adascale.adascale_optimizer import adascale_model_config_dict

    adascale_model_config = adascale_model_config_dict[generator.config.model_type]

..
  # end-before toplevel

.. rubric:: Model topology

AdaScale optimizes one decoder block at a time and takes those block boundaries from an
``LlmTopology``, which describes the structure of the decoder stack. Analyze the float model
before creating the sim, then pass the result as ``topology``::

    topology = analyze_llm_topology(onnx_model, model_type="llama")
    sim = QuantizationSimModel(onnx_model, ...)
    apply_adascale(sim, inputs, adascale_model_config, topology=topology)

..
  # start-after topology

.. autofunction:: aimet_onnx.llm_topology.analyze_llm_topology
    :no-index:

.. autoclass:: aimet_onnx.llm_topology.LlmTopology
    :members:
    :no-index:

..
  # end-before topology
  # end-before
