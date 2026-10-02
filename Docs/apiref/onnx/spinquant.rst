.. _apiref-onnx-spinquant:

##########################################
aimet_onnx.experimental.spinquant
##########################################

..
  # start-after

**Top level APIs**

.. autofunction:: aimet_onnx.experimental.spinquant.apply_spinquant

**Model topology**

SpinQuant places every rotation using an ``LlmTopology``, which describes the structure of the
decoder stack: which linears read from and write to the residual stream (R1), which are the V and O
projections (R2), and which Q/K edges feed each QKᵀ MatMul (R3). Analyze the model, then pass the
result as ``topology``::

    topology = analyze_llm_topology(onnx_model, model_type="llama")
    apply_spinquant(onnx_model, topology=topology)
    sim = QuantizationSimModel(onnx_model, ...)   # built on the rotated graph

.. autofunction:: aimet_onnx.experimental.llm_topology.analyze_llm_topology

.. autoclass:: aimet_onnx.experimental.llm_topology.LlmTopology
    :members:

..
  # end-before
