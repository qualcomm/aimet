.. _install-quick-start:

#####################
Quick Start
#####################

This page describes how to quickly install the latest version of AIMET for the ONNX and PyTorch framework.

For all the framework variants and compute platforms, see :ref:`Installation <install-index>`.

.. _install-quick-start-platform:

Tested platform
===============

aimet-torch and aimet-onnx have been validated on the following platform:

* 64-bit Intel x86-compatible processor
* Python 3.10
* Ubuntu 22.04
* For GPU variants:
    * Nvidia GPU card (Compute capability 5.2 or later)
    * Nvidia driver version 455 or later (using the latest driver is recommended; both CUDA and cuDNN are supported)

Installing AIMET
================

Install AIMET from PyPI

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
      :sync: onnx

      .. code-block:: bash

        pip install aimet-onnx

    .. tab-item:: PyTorch
      :sync: torch

      .. code-block:: Bash

        pip install aimet-torch

Verifying the installation
==========================

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
      :sync: onnx

      .. code-block:: Python

        import aimet_onnx
        print(aimet_onnx.__version__)

    .. tab-item:: PyTorch
      :sync: torch

      .. code-block:: Python

        import aimet_torch
        print(aimet_torch.__version__)


Quantize a small model quickly with AIMET
=========================================

Create a :class:`QuantizationSimModel`, perform calibration,
and then evaluate it.

**Step 1:** Handle imports and other setup.

.. tab-set::
  :sync-group: platform

  .. tab-item:: ONNX
    :sync: onnx

    .. literalinclude:: ../../snippets/onnx/installation_verification.py
      :language: python
      :start-after: # [step_1]
      :end-before: # End of [step_1]

  .. tab-item:: PyTorch
    :sync: torch

    .. literalinclude:: ../../snippets/torch/installation_verification.py
      :language: python
      :start-after: # [step_1]
      :end-before: # End of [step_1]


**Step 2:** Create a :class:`QuantizationSimModel` and ensure the model contains quantization operations.

.. tab-set::
  :sync-group: platform

  .. tab-item:: ONNX
    :sync: onnx

    .. literalinclude:: ../../snippets/onnx/installation_verification.py
      :language: python
      :start-after: # [step_2]
      :end-before: # End of [step_2]

  .. tab-item:: PyTorch
    :sync: torch

    .. literalinclude:: ../../snippets/torch/installation_verification.py
      :language: python
      :start-after: # [step_2]
      :end-before: # End of [step_2]

**Step 3:** Calibrate the model. This example uses random values as input. In real-world cases, calibration should be performed using a representative dataset.


.. tab-set::
  :sync-group: platform

  .. tab-item:: ONNX
    :sync: onnx

    .. literalinclude:: ../../snippets/onnx/installation_verification.py
      :language: python
      :start-after: # [step_3]
      :end-before: # End of [step_3]

  .. tab-item:: PyTorch
    :sync: torch

    .. literalinclude:: ../../snippets/torch/installation_verification.py
      :language: python
      :start-after: # [step_3]
      :end-before: # End of [step_3]

**Step 4:** Evaluate the model.

Infer directly on QuantSim model to check quantized model's accuracy.

.. tab-set::
  :sync-group: platform

  .. tab-item:: ONNX
    :sync: onnx

    .. literalinclude:: ../../snippets/onnx/installation_verification.py
      :language: python
      :start-after: # [step_4]
      :end-before: # End of [step_4]

  .. tab-item:: PyTorch
    :sync: torch

    .. literalinclude:: ../../snippets/torch/installation_verification.py
      :language: python
      :start-after: # [step_4]
      :end-before: # End of [step_4]

Sample output of QuantSim model is shown below:

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. rst-class:: script-output

          .. code-block:: none

            [array([[-0.4599525 ,  0.35107604,  0.43178225, ..., -0.45040053,
                      0.1450607 ,  0.23799022],
                    [-0.4132449 ,  0.20722957,  0.60808927, ..., -0.5315115 ,
                      -0.01675645,  0.22884297],
                    [-0.4677236 ,  0.3576329 ,  0.5317543 , ..., -0.50366503,
                      -0.01392324, -0.0897725 ],
                    ...,
                    [-0.4503196 ,  0.3851556 ,  0.56810045, ..., -0.6998855 ,
                      0.03513189,  0.36678016],
                    [-0.27045077,  0.28065038,  0.46723792, ..., -0.24665177,
                      -0.11899511,  0.03658897],
                    [-0.43477735,  0.35536635,  0.62274104, ..., -0.5091695 ,
                      -0.11446196,  0.10984787]], dtype=float32)]

    .. tab-item:: PyTorch
      :sync: torch
        
      .. rst-class:: script-output

        .. code-block:: none

          DequantizedTensor([[-0.4186,  0.2494,  0.5203,  ..., -0.5985,  0.0303, 0.0086],
                             [-0.4236,  0.2259,  0.3209,  ..., -0.4933, -0.0234, 0.1080],
                             [-0.4082,  0.1676,  0.5803,  ..., -0.4130, -0.1609, -0.0252],
                              ...,
                             [-0.3258,  0.3724,  0.4404,  ..., -0.4881, -0.0870, 0.1108],
                             [-0.3687,  0.3706,  0.5825,  ..., -0.3178,  0.0422, -0.0600],
                             [-0.3603,  0.3587,  0.6014,  ..., -0.5430, -0.1279, 0.2029]], grad_fn=<AliasBackward0>)

Now you are all set to use AIMET to quantize your model.
Try out :ref:`end-to-end example <techniques-ptq>` from Post-Training Quantization example with ImageNet dataset.