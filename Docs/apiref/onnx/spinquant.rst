.. _apiref-onnx-spinquant:

##########################################
aimet_onnx.spinquant
##########################################

..
  # start-after

.. rubric:: Top level APIs

..
  # start-after toplevel

.. autofunction:: aimet_onnx.spinquant.apply_spinquant
    :no-index:

..
  # end-before toplevel

.. rubric:: Model topology

..
  # start-after topology

SpinQuant places every rotation using an ``LlmTopology``, which describes the structure of the
decoder stack: which linears read from and write to the residual stream (R1), which are the V and O
projections (R2), and which Q/K edges feed each QKᵀ MatMul (R3). Analyze the model, then pass the
result as ``topology``::

    topology = analyze_llm_topology(onnx_model, model_type="llama")
    apply_spinquant(onnx_model, topology=topology)
    sim = QuantizationSimModel(onnx_model, ...)   # built on the rotated graph

.. autofunction:: aimet_onnx.llm_topology.analyze_llm_topology
    :no-index:

.. autoclass:: aimet_onnx.llm_topology.LlmTopology
    :members:
    :no-index:

..
  # end-before topology
  # end-before
