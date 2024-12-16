.. _install-quick-start:

###########
Quick Start
###########

This page describes how to quickly install the latest version of AIMET for PyTorch framework.

For all the framework variants and compute platform, see :ref:`Installation <install-index>`.

Prerequisites
=============

The AIMET package requires the following host platform setup:

* 64-bit Intel x86-compatible processor
* Python 3.8–3.11
* Ubuntu 22.04
* For GPU variants:
    * Nvidia GPU card (Compute capability 5.2 or later)
    * Nvidia driver version 455 or later (using the latest driver is recommended; both CUDA and cuDNN are supported)

The following software versions are required for the quick install:

* CUDA Toolkit 12.0
* PyTorch 2.2

Installation
============

Type the following command to install AIMET for PyTorch framework using pip package manager.

.. code-block:: bash

    python3 -m pip install "aimet-torch>=2"

Verification
==========

Type the following command to ensure AIMET is installed via pip. 

.. code-block:: bash

    python3 -m pip show aimet-torch

If installed properly, this command will produce no warnings and display information about the package. 


Let's run some sample PyTorch code to confirm that we can create QuantSim and perform calibration:

**Step 1**: Let's handle imports and other setup.   

.. literalinclude:: ../snippets/torch/installation_verification.py
            :language: python
            :start-after: [step_1]
            :end-before: [step_2]

**Step 2**: We will create QuantSim and ensure the model contains quantization ops. 

.. literalinclude:: ../snippets/torch/installation_verification.py
            :language: python
            :start-after: [step_2]
            :end-before: [step_3]

The model should be composed of Quantized nn.Modules, similar to the output shown below:

:: 
   
  MobileNetV2(
  (features): Sequential(
    (0): Conv2dNormActivation(
      (0): QuantizedConv2d(
        3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        (param_quantizers): ModuleDict(
          (weight): QuantizeDequantize(shape=(32, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
        )
        (input_quantizers): ModuleList(
          (0): QuantizeDequantize(shape=(), qmin=0, qmax=65535, symmetric=False)
        )
        (output_quantizers): ModuleList(
          (0): None
        )
      )
    )
    ...
  )
      

**Step 3**: We perform calibration. As a proof of concept, random input is being passed in. However, dataloaders are commonly used in real world cases.  

.. literalinclude:: ../snippets/torch/installation_verification.py
            :language: python
            :start-after: [step_3]
            :end-before: [step_4]

**Step 4**: We perform evaluation.  

.. literalinclude:: ../snippets/torch/installation_verification.py
            :language: python
            :start-after: [step_4]

The output generated should be of type DequantizedTensor and similar to the one shown below. 

:: 

  DequantizedTensor([[-1.7466,  0.8405,  1.8606,  ..., -0.9714,  0.8366, 2.2363],
                   [-1.6091,  1.0449,  1.7788,  ..., -0.9904,  1.0861, 2.2431],
                   [-1.5307,  0.8442,  1.5157,  ..., -0.7793,  0.6327, 2.3861],
                   ...,
                   [-1.3610,  1.4499,  2.2068,  ..., -0.8188,  1.1155, 2.5962],
                   [-1.1619,  1.2217,  2.1050,  ..., -0.5301,  0.9150, 2.1458],
                   [-1.6340,  0.9826,  2.2459,  ..., -1.0769,  0.9054, 2.2315]], 
                   device='cuda:0', grad_fn=<AliasBackward0>)


See the :ref:`User guide <opt-guide-index>` to read about the model optimization workflow.

See the :ref:`Examples <examples-index>` to try AIMET quantization techniques on your pre-trained models.


