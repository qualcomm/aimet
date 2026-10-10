.. _apiref-onnx-adascale:

##########################################
aimet_onnx.adascale
##########################################

..
  # start-after

**Top level APIs**

.. autofunction:: aimet_onnx.adascale.adascale_optimizer.apply_adascale

.. autoclass:: aimet_onnx.adascale.adascale_optimizer.AdaScaleModelConfig
    :members:

**Model topology**

AdaScale optimizes one decoder block at a time and takes those block boundaries from an
``LlmTopology``, which describes the structure of the decoder stack. Analyze the float model
before creating the sim, then pass the result as ``topology``::

    topology = analyze_llm_topology(onnx_model, model_type="llama")
    sim = QuantizationSimModel(onnx_model, ...)
    apply_adascale(sim, inputs, adascale_model_config, topology=topology)

.. autofunction:: aimet_onnx.llm_topology.analyze_llm_topology

.. autoclass:: aimet_onnx.llm_topology.LlmTopology
    :members:

..
  # end-before
