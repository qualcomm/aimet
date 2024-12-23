.. _apiref-torch-model-validator:

###########################
aimet_torch.model_validator
###########################

AIMET provides a model validator utility to help check whether AIMET feature can be applied on a Pytorch model. The
model validator currently checks for the following conditions:

- No modules are reused
- Operations have modules associated with them and are not defined as :mod:`torch.nn.functional` (excluding a set of known operations)

In this section, we present models failing the validation checks, and show how to run the model validator, as well as
how to fix the models so the validation checks pass.

**Example 1: Model with reused modules**

We begin with the following model, which contains two relu modules sharing the same module instance.

.. literalinclude:: ../../legacy/torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: ModelWithReusedNodes
   :emphasize-lines: 13, 15

Import the model validator:

.. literalinclude:: ../../legacy/torch_code_examples/model_validator_code_example.py
   :language: python
   :lines: 44

Run the model validator on the model by passing in the model as well as model input:

.. literalinclude:: ../../legacy/torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: validate_example_model

For each validation check run on the model, a logger print will appear::

    Utils - INFO - Running validator check <function validate_for_reused_modules at 0x7f127685a598>

If the validation check finds any issues with the model, the log will contain information for how to resolve the model::

    Utils - WARNING - The following modules are used more than once in the model: ['relu1']
    AIMET features are not designed to work with reused modules. Please redefine your model to use distinct modules for
    each instance.

Finally, at the end of the validation, any failing checks will be logged::

    Utils - INFO - The following validator checks failed:
    Utils - INFO -     <function validate_for_reused_modules at 0x7f127685a598>

In this case, the validate_for_reused_modules check informs that the relu1 module is being used multiple times in the
model. We rewrite the model by defining a separate relu instance for each usage:

.. literalinclude:: ../../legacy/torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: ModelWithoutReusedNodes
   :emphasize-lines: 9, 16

Now, after rerunning the model validator, all checks pass::

    Utils - INFO - Running validator check <function validate_for_reused_modules at 0x7ff577373598>
    Utils - INFO - Running validator check <function validate_for_missing_modules at 0x7ff5703eff28>
    Utils - INFO - All validation checks passed.

**Example 2: Model with functionals**

We start with the following model, which uses a torch linear functional layer in the forward pass:

.. literalinclude:: ../../legacy/torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: ModelWithFunctionalLinear
   :emphasize-lines: 17

Running the model validator shows the validate_for_missing_modules check failing::

    Utils - INFO - Running validator check <function validate_for_missing_modules at 0x7f9dd9bd90d0>
    Utils - WARNING - Ops with missing modules: ['matmul_8']
    This can be due to several reasons:
    1. There is no mapping for the op in ConnectedGraph.op_type_map. Add a mapping for ConnectedGraph to recognize and
    be able to map the op.
    2. The op is defined as a functional in the forward function, instead of as a class module. Redefine the op as a
    class module if possible. Else, check 3.
    3. This op is one that cannot be defined as a class module, but has not been added to ConnectedGraph.functional_ops.
    Add to continue.
    Utils - INFO - The following validator checks failed:
    Utils - INFO - 	<function validate_for_missing_modules at 0x7f9dd9bd90d0>

The check has identified matmul_8 as an operation with a missing pytorch module. In this case, it is due to reason #2
in the log, in which the layer has been defined as a functional in the forward function. We rewrite the model by
defining the layer as a module instead in order to resolve the issue.

.. literalinclude:: ../../legacy/torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: ModelWithoutFunctionalLinear
   :emphasize-lines: 10, 20

API
===

..
  # start-after

.. autoclass:: aimet_torch.model_validator.model_validator.ModelValidator
    :members:

..
  # end-before
