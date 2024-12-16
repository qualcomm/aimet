:orphan:

.. _api-torch-multi-gpu:

=========================
PyTorch Multi-GPU support
=========================

Currently AIMET supports models using Multi-GPU in data parallel mode with the following features

1) Cross-Layer Equalization (CLE)
2) Quantization Aware Training (QAT)

A user can create a Data Parallel model using torch APIs. For example::

    # Instantiate a torch model and pass it to DataParallel API
    model = torch.nn.DataParallel(model)

**Multi-GPU with CLE**

For using multi-GPU with CLE, you can pass the above created model directly to the CLE API
:ref:`Cross-Layer Equalization API<api-torch-cle>`

NOTE: CLE doesn't actually make use of multi-GPU, it is only integrated as a part of work-flow so that user need not move the model
back and forth from single gpu to multi-GPU and back.

**Multi-GPU with Quantization Aware Training**

For using multi-GPU with QAT,

1) Create a QuantizationSim as shown in :ref:`Quantization Simulation API<api-torch-quantsim>`  using a torch model (Not in DataParallel mode)
2) Perform compute encodings (NOTE: Do not use a forward function that moves the model to multi-gpu and back)
3) Move sim model to DataParallel::

    sim.model = torch.nn.DataParallel(sim.model)

4) Perform Eval/Training