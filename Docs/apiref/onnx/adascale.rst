.. _apiref-onnx-adascale:

##########################################
aimet_onnx.experimental.adascale
##########################################

..
  # start-after

**Top level APIs**

.. autofunction:: aimet_onnx.experimental.adascale.adascale_optimizer.apply_adascale

.. autoclass:: aimet_onnx.experimental.adascale.adascale_optimizer.AdaScaleModelConfig
    :members:

**Model topology**

AdaScale optimizes one decoder block at a time and takes those block boundaries from an
``LlmTopology``, which describes the structure of the decoder stack. Analyze the float model
before creating the sim, then pass the result as ``topology``::

    topology = analyze_llm_topology_by_norm_count(onnx_model)
    sim = QuantizationSimModel(onnx_model, ...)
    apply_adascale(sim, inputs, adascale_model_config, topology=topology)

.. autofunction:: aimet_onnx.experimental.llm_topology.analyze_llm_topology_by_norm_count

.. autoclass:: aimet_onnx.experimental.llm_topology.LlmTopology
    :members:

..
  # end-before
