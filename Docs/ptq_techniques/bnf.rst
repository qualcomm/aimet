.. include:: ../abbreviation.txt

.. _ptq-bnf:

##################
Batch norm folding
##################

Context
=======

Batch norm folding (BNF) is a technique widely used in deep learning inference runtimes, including |qnn|_.
In BNF, batch normalization layers are folded into the weights and biases of adjacent convolution layers to eliminate unnecessary computations.

To accurately simulate inference in these runtimes, perform BNF on the floating-point model before applying quantization. Doing so not only speeds performance (inferences per second) but also often improves the accuracy of the quantized model by removing redundant computations and requantization. AIMET enables you to apply BNF to the pre-quantized model as a precursor to simulating on-target behavior in the quantization simulation (QuantSim) model.

Workflow
========

Procedure
---------

Step 1
~~~~~~

Load the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_bnf.py
            :language: python
            :start-after: # Step 1
            :end-before: # End of step 1

        .. rst-class:: script-output
    
          .. code-block:: none

            MobileNetV2(
              (features): Sequential(
                (0): Conv2dNormActivation(
                  (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                  (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU6(inplace=True)
                )
                ...
            )

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../snippets/onnx/apply_bnf.py
            :language: python
            :start-after: # pylint: disable=missing-docstring
            :end-before: # End of step 1

        .. rst-class:: script-output

          .. code-block:: none

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

Prepare the model, if required by the model framework.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        No preparation step is needed for PyTorch.

    .. tab-item:: ONNX
        :sync: onnx

        We recommend that you simplify the ONNX model as follows.

        .. literalinclude:: ../snippets/onnx/apply_bnf.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

        .. rst-class:: script-output

          .. code-block:: none

            *** Before batch norm folding ***

            model.graph.node[0]:
            name: "/features/features.0/features.0.0/Conv"

            model.graph.node[1]:
            name: "/features/features.0/features.0.1/BatchNormalization"

            Conv weight:
            [[[[-6.31080866e-02 -1.87656835e-01 -1.51876003e-01]
               [-4.93787616e-01 -6.42477691e-01 -5.89348674e-01]
               [-6.80053532e-01 -9.74478185e-01 -7.63172388e-01]]
              ...
              [[ 1.24257803e-02 -4.73242160e-03 -1.81884710e-02]
               [ 2.32141271e-01  7.22583652e-01  1.21250950e-01]
               [-2.59643137e-01 -7.18673885e-01 -9.19778645e-02]]]]

Step 3
~~~~~~

Perform the batch norm folding.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_bnf.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

        .. rst-class:: script-output

          .. code-block:: none

            *** Before batch norm folding ***

            model.features[0][0]:
            Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

            model.features[0][1]:
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


            *** After batch norm folding ***

            model.features[0][0]:
            Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            model.features[0][1]:
            Identity()

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Execute the AIMET BNF API.

        .. literalinclude:: ../snippets/onnx/apply_bnf.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3


        .. rst-class:: script-output

          .. code-block:: none

            *** After batch norm folding ***

            model.graph.node[0]:
            name: "/features/features.0/features.0.0/Conv"

            model.graph.node[1]:
            name: "/features/features.0/features.0.2/Clip"

            Conv weight:
            [[[[-2.00183112e-02 -5.95260113e-02 -4.81760912e-02]
               [-1.56632766e-01 -2.03798249e-01 -1.86945379e-01]
               [-2.15717569e-01 -3.09111059e-01 -2.42083430e-01]]
               ...
              [[ 1.21066449e-02 -4.61087702e-03 -1.77213307e-02]
               [ 2.26179108e-01  7.04025269e-01  1.18136823e-01]
               [-2.52974629e-01 -7.00215936e-01 -8.96155685e-02]]]]


API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/bnf.rst
            :start-after: # start-after

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../apiref/onnx/bnf.rst
           :start-after: # start-after
