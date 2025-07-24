.. _ptq-spinquant:

#########
SpinQuant
#########

Context
=======

SpinQuant is a PTQ technique which improves the accuracy of the quantized model by inserting rotations at specific points in the model to help with outliers in activation quantization: https://arxiv.org/pdf/2405.16406.

In the paper, 4 rotation types are described: R1, R2, R3, and R4 rotations. **The current AIMET implementation of SpinQuant enables R1 rotations without optimization only.**
As these rotations can be merged with adjacent layer weights, the final model architecture will not be changed.

Applying rotations does not require a quantized model, so either an FP model or a quantized model can be used as input.
Since rotations need to be inserted at well known points in the model, the feature determines proper insertion points through use of a mapping table to define pre-determined insertion points for known model types.

Currently supported model types include

- LlamaForCausalLM
- Qwen2ForCausalLM
- MistralForCausalLM

We expose the mapping dictionary as a module level object in case users need to register their own insertion points for other model types.

.. note::
   This feature is currently marked as experimental. The API may change in the future.

.. note::
   This feature is currently only supported for PyTorch framework.

.. note::
   Only R1 rotations without optmization are currently supported.

Workflow
========

Prerequisites
-------------

To use SpinQuant, you must:

- Load a pre-trained model

Procedure
---------

Setup
~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_spinquant.py
            :language: python
            :start-after: # [setup]
            :end-before: # End of [setup]

Step 1
~~~~~~

Register RMSNorm fusion locations and R1 insertion points if needed. The below code shows an example for a model of type MyModel.
RMSNorm fusion is a transformation for folding RMSNorm weights into adjacent linear layers. The resulting RMSNorm op will have weights of all 1's and bias 0.
The folded model is mathematically equivalent to the original model in floating point computation, and is necessary for R1 transforms to be added later.

To prepare SpinQuant to take effect for a model other than what is already supported, users need to register the model type with two functions:

- A function which, when given a model object, returns a list of tuples, where each tuple consists of an rmsnorm layer and a list of linear layers it should fuse with
- A function which, when given a model object, returns a list of tuples, where each tuple consists of a linear layer and a boolean.
- A boolean of True denotes R1 fusion occurring before the linear, while False denotes R1 fusion occurring after the linear.

For typical HuggingFace models which share similar architecture, _default_rmsnorm_linear_pairs_func() and _default_r1_fusion_func() can be used. For example, Llama, Qwen, and Mistral all share the same functions.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_spinquant.py
            :language: python
            :start-after: # [register-rmsnorm-r1-points]
            :end-before: # End of [register-rmsnorm-r1-points]

Step 2
~~~~~~

Apply SpinQuant to the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_spinquant.py
            :language: python
            :start-after: # [apply-spinquant]
            :end-before: # End of [apply-spinquant]

Step 3
~~~~~~

The subsequent steps are not strictly to do with SpinQuant, but serve as an example for how to quantize the model and evaluate.
Use AIMET's :ref:`quantization simulation<quantsim-index>` to create a QuantSimModel object.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_spinquant.py
            :language: python
            :start-after: # [create-sim]
            :end-before: # End of [create-sim]

Step 4
~~~~~~

Instantiate a dataloader and compute encodings for remaining parameters of the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_spinquant.py
            :language: python
            :start-after: # [prepare-dataloader]
            :end-before: # End of [prepare-dataloader]

        .. literalinclude:: ../snippets/torch/apply_spinquant.py
            :language: python
            :start-after: # [compute_encodings]
            :end-before: # End of [compute_encodings]

Step 5
~~~~~~

At this point, the quantized model is ready to be evaluated.

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/spinquant.rst
            :start-after: # start-after
