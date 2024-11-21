.. _featureguide-cle:

########################
Cross-layer equalization
########################

Context
=======

To be filled

Workflow
========

Code example
------------

Step 1
~~~~~~

Load the model for cross-layer equalization.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        To be filled

    .. tab-item:: TensorFlow
        :sync: tf

        To be filled

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Load the model for cross-layer equalization. In this code example, we will convert PyTorch MobileNetV2 to ONNX and use it in the subsequent code

        .. literalinclude:: ../snippets/onnx/apply_cle.py
            :language: python
            :start-after: # Step 1
            :end-before: # End of step 1

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

Step 2
~~~~~~

Apply preparation step if necessary

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        To be filled

    .. tab-item:: TensorFlow
        :sync: tf

        To be filled

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            It's recommended to simplify the ONNX model before applying AIMET functionalities.
            After simplification, we find that the model contains consecutive convolutions, which can be optimized through cross-layer equalization

        .. literalinclude:: ../snippets/onnx/apply_cle.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

        **Output**
        ::

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

Step 3
~~~~~~

Execute AIMET cross-layer equalization API

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        To be filled

    .. tab-item:: TensorFlow
        :sync: tf

        To be filled

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Execute AIMET cross-layer equalization API

        .. literalinclude:: ../snippets/onnx/apply_cle.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

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
