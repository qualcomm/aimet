.. _torch-multi-gpu:

=========================
PyTorch multi-GPU support
=========================

Currently AIMET supports models using Multi-GPU in data parallel mode with the following features

#. Cross-layer equalization (CLE)
#. Quantization-aware training (QAT)

A user can create a Data Parallel model using torch APIs. For example::

    # Instantiate a torch model and pass it to DataParallel API
    model = torch.nn.DataParallel(model)

Multi-GPU with CLE
==================

For using multi-GPU with CLE, you can pass the above created model directly to the CLE API
:ref:`Cross-Layer Equalization API<api-torch-cle>`

.. note::
    CLE doesn't actually make use of multi-GPU, it is only integrated as a part of work-flow so that user need not move the model
    back and forth from single gpu to multi-GPU and back.

Multi-GPU with quantization-aware training
==========================================

For using multi-GPU with QAT,

#. Create a :class:`QuantizationSimModel` for your pre-trained PyTorch model (Not in DataParallel mode)
#. Perform :func:`QuantizationSimModel.compute_encodings` (NOTE: Do not use a forward function that moves the model to multi-gpu and back)
#. Move Quantsim model to DataParallel::

    sim.model = torch.nn.DataParallel(sim.model)

#. Perform eval and/or training.

