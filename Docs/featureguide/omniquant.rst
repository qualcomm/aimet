.. _featureguide-OmniQuant:

#########
OmniQuant
#########

Context
=======

OmniQuant is a PTQ technique which improves the accuracy of the quantized model by computing optimal quantization parameters for weights. OmniQuant is based on : https://arxiv.org/abs/2308.13137.
Omniquant comprises of 2 components `Learnable Weight Clipping (LWC)` and `Learnable Equivalent Transformation (LET)`.

OmniQuant introduces trainable parameter `scale` in the weight quantizers of every supported module and performs BKD (Blockwise Knowledge Distillation) by comparing quantized output of every supported block with its FP32 equivalent.
The trainable parameter `scale` is learnt pairwise in Omniquant.
From OmniQuant perspective, a block is defined as a non-leaf module which takes in one activation input tensor and outputs one activation tensor.

Warning: This feature is currently experimental. This feature is currently supported for llama3.2, Qwen2.5, Deepseek Distill for Qwen 2.5

Workflow
========

Prerequisites
-------------

To use OmniQuant, you must:

- Use PyTorch. OmniQuant does not support other frameworks yet
- Load a pre-trained model
- Create a dataloader for the model

Procedure
---------

Setup
~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_omniquant.py
            :language: python
            :start-after: # [setup]
            :end-before: # End of [setup]

        .. literalinclude:: ../snippets/torch/apply_omniquant.py
            :language: python
            :start-after: # [prepare-dataloader]
            :end-before: # End of [prepare-dataloader]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

Step 1
~~~~~~

Use AIMET's :ref:`quantization simulation<quantsim-index>` to create a QuantSimModel object.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_omniquant.py
            :language: python
            :start-after: # [create-sim]
            :end-before: # End of [create-sim]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.


Step 2
~~~~~~

Apply apply_omniquant to decide optimal quantization encodings for parameters of supported layers.
It is recommended to use a minimum of 800 iterations when applying apply_omniquant regardless of the dataloader batch size.
The `learnt scales` are dumped as safetensors when we do apply_omniquant. These scales can be used for quantizing lora adapters.
The usage of the dumped scale is not supported in the current release.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_omniquant.py
            :language: python
            :start-after: # [apply-omniquant]
            :end-before: # End of [apply-omniquant]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

Step 3
~~~~~~

Compute encodings for remaining parameters of the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_omniquant.py
            :language: python
            :start-after: # [compute_encodings]
            :end-before: # End of [compute_encodings]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

Step 4
~~~~~~

Evaluate the quantized model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_omniquant.py
            :language: python
            :start-after: # [evaluation]
            :end-before: # End of [evaluation]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

Step 5
~~~~~~

If the resulting quantized accuracy is satisfactory, export the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_omniquant.py
            :language: python
            :start-after: # [export]
            :end-before: # End of [export]

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/omniquant.rst
            :start-after: # start-after

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.


