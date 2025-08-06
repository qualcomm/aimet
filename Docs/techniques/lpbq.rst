.. include:: ../abbreviation.txt

.. _techniques-lpbq:

#######################################
Low-Power Blockwise Quantization (LPBQ)
#######################################

|qnn| supports an alternative to blockwise quantization called *Low-Power Blockwise Quantization* (*LPBQ*).

.. note::
   To read about generic blockwise quantization, see :ref:`Blockwise Quantization <techniques-blockwise>`

In LPBQ, blockwise encodings at a lower bit width are adjusted such that they lie on a common higher-bit-width per-channel grid.

Following are the benefits of LPBQ over Blockwise Quantization (BQ):

* Enables models to run on existing per-channel kernels
* Generated encodings are storage-efficient than BQ

|qnn| has the following restrictions on LPBQ:

* Blockwise quantization runs on weight (not activation) quantizers only
* Block size must be set to one for the output channel dimension
* Input channel dimension must be divisible by Block size

Apply LPBQ
==========

This section walks through how to enable LPBQ in :ref:`Post-Training Quantization <techniques-ptq>` workflow.

.. image:: ../images/techniques/lpbq.png

LPBQ workflow looks like the following:

1. Create QuantizationSimModel
2. Enable LPBQ for select operations (additional step on top of :ref:`Post-Training Quantization <techniques-ptq>` workflow
3. Create a calibration callback to be used for computing quantization parameters
4. Compute encodings
5. Evaluation
6. Export the model


Apply LPBQ on select set of modules in PyTorch or operations in ONNX with :func:`set_grouped_blockwise_quantization_for_weights`.

This function can be called multiple times to set different LPBQ configuration for target set of modules or operations.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. code-block:: Python

            # 1. Create QuantizationSimModel
            # ...

            from aimet_torch.quantsim.config_utils import set_grouped_blockwise_quantization_for_weights


            # 2. Apply LPBQ
            set_grouped_blockwise_quantization_for_weights(
                sim=quantsim,
                arg=[torch.nn.Linear],
                bitwidth=4,
                symmetric=True,
                decompressed_bw=8,
                block_size=64,
                block_grouping=-1,
            )

            # Continue with calibration

    .. tab-item:: ONNX
        :sync: onnx

        .. code-block:: Python

            # 1. Create QuantizationSimModel
            # ...

            from aimet_onnx.quantsim import set_blockwise_quantization_for_weights

            # 2. Apply LPBQ
            set_grouped_blockwise_quantization_for_weights(
                sim=quantsim,
                op_types=("Gemm", "MatMul", "Conv"),
                bitwidth=4,
                decompressed_bw=8,
                block_size=64,
                excluded_nodes = ['conv1', 'linear10']
            )

            # Continue with calibration

API
===

**Top-level API to configure LPBQ quantization**

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. autofunction:: aimet_torch.v2.quantsim.config_utils.set_grouped_blockwise_quantization_for_weights
            :noindex:

        This utility enables you to configure quantized layers to use grouped blockwise quantization by supplying a `decompressed_bw`,  `block_size`, and `block_grouping`. Similar to :func:`set_blockwise_quantization_for_weights`, `block_grouping` can be a single value. In this case the input_channel's dimension is assigned the value, and all other dimensions are assigned a value of one.

        Different layers can have different numbers of blocks for the input channels dimension for the same block size. If you assign -1 as the single `block_grouping` value, the input channels dimension automatically uses a `block_grouping` value equal to the number of blocks in any affected layer. This enbles you to configure all affected layers to LPBQ quantization with a single API call.

    .. tab-item:: ONNX
        :sync: onnx

        .. autofunction:: aimet_onnx.quantsim.set_grouped_blockwise_quantization_for_weights
            :noindex:
