.. _featureguide-cle:

########################
Cross-layer equalization
########################

Context
=======
Quantization of floating point models into lower bitwidths introduces quantization noise on the weights and activations, which often leads to reduced model performance. To minimize quantization noise, there are a variety of post training quantization (PTQ) techniques offered by AIMET. You can learn more about these techniques `here <https://arxiv.org/pdf/1906.04721>`_.

AIMET's cross-layer equalization tool involves the following techniques:

- **Batch Norm Folding**: This feature folds batch norm layers into adjacent convolutional and linear layers. You can learn more :ref:`here <_featureguide-bnf>`
  
- **Cross Layer Scaling**: In some models, the parameter ranges for different channels in a layer show a wide variance (as shown below). This feature attempts to equalize the distribution of weights per channel of consecutive layers. Thus, different channels have a similar range and the same quantization parameters can be used for weights across all channels.

    .. figure:: ../images/cross_layer_scaling.png

- **High Bias Fold**: Cross layer scaling may result in high bias parameter values for some layers. This technique folds some of the bias of a layer into the subsequent layer's parameters. This feature requires batch norm parameters to operate on and will not be applied otherwise. 

Workflow
========

Setup
~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch
        
        .. literalinclude:: ../snippets/torch/apply_cle.py
            :start-after: [setup]
            :end-before: [step_1]

    .. tab-item:: TensorFlow
        :sync: tf

        .. literalinclude:: ../snippets/tensorflow/apply_cle.py
            :language: python
            :start-after: # pylint: disable=missing-docstring

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Load the model for cross-layer equalization. In this code example, we will convert PyTorch MobileNetV2 to ONNX and use it in the subsequent code. 
            
            It's recommended to simplify the ONNX model before applying AIMET functionalities. After simplification, we find that the model contains consecutive convolutions, which can be optimized through cross-layer equalization. 

        .. literalinclude:: ../snippets/onnx/apply_cle.py
            :language: python
            :start-after: # Step 1
            :end-before: [step_1]

        **Output**
        ::

            MobileNetV2(
              (features): Sequential(
                (0): Conv2dNormActivation(
                  (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                  (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU6(inplace=True)
                )
                ...
            )
            *** Before cross-layer equalization ***

            model.graph.node[4]:
            /features/features.1/conv/conv.1/Conv

            model.graph.node[5]:
            /features/features.2/conv/conv.0/conv.0.0/Conv

            Prev Conv weight
            [[[[ 1.83640555e-01]]
              [[ 6.34215236e-01]]
              [[ 8.44993666e-02]]
              ...
              [[-6.70130579e-17]]
              [[-1.37757687e-02]]
              [[ 9.16839484e-03]]]]

            Next Conv weight
            [[[[-8.41059163e-02]]
              [[-1.12039044e-01]]
              [[-2.72468403e-02]]
              ...
              [[ 9.46642041e-01]]
              [[ 4.35139937e-03]]
              [[ 2.57021021e-02]]]]

Step 1
~~~~~~

Execute AIMET cross-layer equalization API

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch
        
        .. literalinclude:: ../snippets/torch/apply_cle.py
            :language: python
            :start-after: [step_1]

    .. tab-item:: TensorFlow
        :sync: tf

        To be filled

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Execute AIMET cross-layer equalization API

        .. literalinclude:: ../snippets/onnx/apply_cle.py
            :language: python
            :start-after: [step_1]

        **Output**
        ::
            *** After cross-layer equalization ***

            Prev Conv weight
            [[[[ 6.28238320e-02]]
              [[ 2.16966406e-01]]
              [[ 2.89074164e-02]]
              ...
              [[-2.44632760e-17]]
              [[-5.02887694e-03]]
              [[ 3.34694423e-03]]]]

            Next Conv weight
            [[[[-2.4585028e-01]]
              [[-3.5856506e-01]]
              [[-3.3467390e-02]]
              ...
              [[ 1.2930528e+00]]
              [[ 1.6213797e-02]]
              [[ 7.0406616e-02]]]]

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/cle.rst
            :start-after: _apiref-torch-cle:

    .. tab-item:: TensorFlow
        :sync: tf

        .. include:: ../apiref/tensorflow/cle.rst
           :start-after: _apiref-keras-cle:

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../apiref/onnx/cle.rst
           :start-after: _apiref-onnx-cle:
