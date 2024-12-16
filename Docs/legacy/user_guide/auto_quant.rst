.. _ug-auto-quant:


###############
AIMET AutoQuant
###############

Overview
========

AIMET offers a suite of neural network post-training quantization techniques. Often, applying these techniques in a specific sequence results in better accuracy and performance. 

The AutoQuant feature analyzes the model, determines the best sequence of AIMET quantization techniques, and applies these techniques. You can specify the accuracy drop that can be tolerated in the AutoQuant API.
As soon as this threshold accuracy is reached, AutoQuant stops applying quantization techniques.

Without the AutoQuant feature, you must manually try combinations of AIMET quantization techniques. This manual process is error-prone and time-consuming.

Workflow
========

The workflow looks like this:


    .. image:: ../images/auto_quant_v2_flowchart.png


Before entering the optimization workflow, AutoQuant prepares by:

1. Checking the validity of the model and converting the model into an AIMET quantization-friendly format (`Prepare Model`).
2.  Selecting the best-performing quantization scheme for the given model (`QuantScheme Selection`)

After the prepration steps, AutoQuant proceeds to try three techniques:

1. BatchNorm folding            
2. :ref:`Cross-Layer Equalization (CLE) <ug-post-training-quantization>`
3. :ref:`AdaRound <ug-adaround>`

These techniques are applied in a best-effort manner until the model meets the allowed accuracy drop.
If applying AutoQuant fails to satisfy the evaluation goal, AutoQuant returns the model that returned the best results.

AutoQuant API
=============

See the AutoQuant API for your AIMET variant:

- :ref:`AutoQuant for PyTorch<api-torch-auto-quant>`
- :ref:`AutoQuant for ONNX<api-onnx-auto-quant>`

