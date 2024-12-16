:orphan:

.. _api-torch-architecture_checker:

**EXPERIMENTAL**

==================
Architecture Checker API
==================

.. autofunction:: aimet_torch.arch_checker.arch_checker.ArchChecker.check_model_arch


AIMET PyTorch Architecture Checker helps checks for sub-optimal model construct and provides potential option to
update the model to be more performant. The architecture checker currently checks for the following conditions:

- Convolution layers for optimal channel size.
- Activation functions that are not performant.
- Batch Normalization layer than cannot be folded.
- Intermediate convolution layer in sequence of convolution layer having padding.


In this section, we present models failing the architecture checks, and show how to run the architecture checker.

**Example 1: Model with not enough channels**

We begin with the following model, which contains a convolution layer with channel less that 32.

.. literalinclude:: ../torch_code_examples/architecture_checker_code_example.py
   :language: python
   :pyobject: ModelWithNotEnoughChannels
   :emphasize-lines: 6

Import the architecture checker:

.. literalinclude:: ../torch_code_examples/architecture_checker_code_example.py
   :language: python
   :lines: 42

Run the checker on the model by passing in the model as well as the model input:

.. literalinclude:: ../torch_code_examples/architecture_checker_code_example.py
   :language: python
   :pyobject: example_check_for_number_of_conv_channels

the convolution layer in the model has one fewer channel, the following logger print will appear::

    Utils - INFO - Graph/Node: ModelWithNotEnoughChannels.conv1: Conv2d(3, 31, kernel_size=(2, 2), stride=(2, 2), padding=(2, 2), bias=False) fails check: {'_check_conv_channel_32_base', '_check_conv_channel_larger_than_32'}

A HTML file with the following content is generated.

.. list-table:: HTML report content
   :widths: 25 25 50
   :header-rows: 1

   * - Graph/Layer_name
     - Issue
     - Recommendation
   * - ModelWithNotEnoughChannels.conv1
     - The channel size of input/output tensor of this convolution is smaller than 32
     - Try adjusting the channels to multiple of 32 to get better performance.
   * - ModelWithNotEnoughChannels.conv1
     - The channel size of input/output tensor of this convolution is not a multiple of 32
     - Try adjusting the channels to multiple of 32 to get better performance.
**Example 2: Model with non-performant activation**

We begin with the following model, which contains a convolution layer with channel less that 32.

.. literalinclude:: ../torch_code_examples/architecture_checker_code_example.py
   :language: python
   :pyobject: ModelWithPrelu
   :emphasize-lines: 8

Run the checker on the model by passing in the model as well as the model input:

.. literalinclude:: ../torch_code_examples/architecture_checker_code_example.py
   :language: python
   :pyobject: example_check_for_non_performant_activations

the PReLU layer in model is consider non-performant compared to ReLU, the following logger print will appear::

    Utils - INFO - Graph/Node: ModelWithPrelu.prelu1: PReLU(num_parameters=1) fails check: {'_activation_checks'}

**Example 3: Model with standalone batch normalization layer**

We begin with the following model, which contains a convolution layer with channel less that 32.

.. literalinclude:: ../torch_code_examples/architecture_checker_code_example.py
   :language: python
   :pyobject: ModelWithNonfoldableBN
   :emphasize-lines: 12

Run the checker on the model by passing in the model as well as the model input:

.. literalinclude:: ../torch_code_examples/architecture_checker_code_example.py
   :language: python
   :pyobject: example_check_for_standalone_bn

the AveragePool layer prevents the BatchNormalization layer to be folded with the Convolution layer, the following logger print will appear::

    Utils - INFO - Graph/Node: ModelWithNonfoldableBN.bn1: BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) fails check: {'_check_batch_norm_fold'}


