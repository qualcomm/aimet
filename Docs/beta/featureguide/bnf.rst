.. include:: ../abbreviation.txt

.. _featureguide-bnf:

##################
Batch norm folding
##################

Context
=======

Batch norm folding is a technique widely used in deep learning inference runtimes, including the |qnn|_.
Batch normalization layers are typically folded into the weights and biases of adjacent convolution layers whenever possible to eliminate unnecessary computations.
To accurately simulate inference in these runtimes, it is generally advisable to perform batch norm folding on the floating-point model before applying quantization.
Doing so not only results in a speedup in inferences per second by avoiding unnecessary computations but also often improves the accuracy of the quantized model by removing redundant computations and requantization.
We aim to simulate this on-target behavior by performing batch norm folding here.

Workflow
========

Code example
------------

Step 1
~~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. container:: tab-heading

            Load the model for batch norm folding. In this code example, we will use MobileNetV2

        .. literalinclude:: ../snippets/torch/apply_bnf.py
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

    .. tab-item:: TensorFlow
        :sync: tf

        .. container:: tab-heading

            Load the model for batch norm folding. In this code example, we will use MobileNetV2

        .. literalinclude:: ../snippets/tensorflow/apply_bnf.py
            :language: python
            :start-after: # pylint: disable=missing-docstring
            :end-before: # End of step 1

        **Output**
        ::

            Model: "mobilenetv2_1.00_224"
            __________________________________________________________________________________________________
             Layer (type)                   Output Shape         Param #     Connected to
            ==================================================================================================
             input_1 (InputLayer)           [(None, 224, 224, 3  0           []
                                            )]

             Conv1 (Conv2D)                 (None, 112, 112, 32  864         ['input_1[0][0]']
                                            )

             bn_Conv1 (BatchNormalization)  (None, 112, 112, 32  128         ['Conv1[0][0]']
                                            )

             Conv1_relu (ReLU)              (None, 112, 112, 32  0           ['bn_Conv1[0][0]']
                                            )
             ...

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Load the model for batch norm folding. In this code example, we will convert PyTorch MobileNetV2 to ONNX and use it in the subsequent code

        .. literalinclude:: ../snippets/onnx/apply_bnf.py
            :language: python
            :start-after: # pylint: disable=missing-docstring
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

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. container:: tab-heading

            No preparation step is needed for PyTorch.

    .. tab-item:: TensorFlow
        :sync: tf

        .. container:: tab-heading

            AIMET provides TensorFlow `prepare_model` API, which performs preprocessing on the user model if necessary

        .. literalinclude:: ../snippets/tensorflow/apply_bnf.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

        **Output**
        ::

            *** Before batch norm folding ***

            prepared_model.layers[1]:
            <class 'keras.layers.convolutional.conv2d.Conv2D'>

            prepared_model.layers[2]:
            <class 'keras.layers.normalization.batch_normalization.BatchNormalization'>

            Conv weight:
            [[[[-1.71659231e-01 -3.33731920e-01  5.30122258e-02 -5.93232973e-21
                 2.08742931e-01 -1.20433941e-01  1.75700430e-02 -3.10708203e-22
                -9.62498877e-03  1.90229788e-01 -3.67278278e-01  3.95997976e-22
              ...
                 3.87471542e-02 -3.67677957e-02 -3.23011987e-02 -4.83861901e-02
                 1.23156421e-02 -5.57984132e-03 -6.53976866e-04 -1.92511864e-02
                -2.09685047e-22  1.19186290e-01 -2.52912678e-02  2.02078857e-02]]]]

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            It's recommended to simplify the ONNX model before applying AIMET functionalities

        .. literalinclude:: ../snippets/onnx/apply_bnf.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

        **Output**
        ::

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

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. container:: tab-heading

            Execute AIMET batch norm folding API

        .. literalinclude:: ../snippets/torch/apply_bnf.py
            :language: python
            :start-after: # Step 2
            :end-before: # End of step 2

        **Output**
        ::

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

    .. tab-item:: TensorFlow
        :sync: tf

        .. container:: tab-heading

            Execute AIMET batch norm folding API

        .. literalinclude:: ../snippets/tensorflow/apply_bnf.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3

        **Output**
        ::

            *** After batch norm folding ***

            folded_model.layers[1]:
            <class 'keras.layers.convolutional.conv2d.Conv2D'>

            folded_model.layers[2]:
            <class 'keras.layers.activation.relu.ReLU'>

            Conv weight
            [[[[-3.01457286e-01 -1.49024737e+00  6.10569119e-01 -1.29590677e-19
                 1.51547194e-01 -1.51446089e-01  1.38100997e-01 -4.89249423e-21
                -5.16245179e-02  4.64579314e-01 -2.44408584e+00  1.22219264e-20
                 ...
                 1.67510852e-01 -2.60713138e-02 -1.05549544e-01 -2.53403008e-01
                 1.39502389e-02 -1.54620111e-02 -1.97294299e-02 -9.41715762e-02
                -6.88260233e-21  8.95088911e-02 -1.87630311e-01  2.48399768e-02]]]]

    .. tab-item:: ONNX
        :sync: onnx

        .. container:: tab-heading

            Execute AIMET batch norm folding API

        .. literalinclude:: ../snippets/onnx/apply_bnf.py
            :language: python
            :start-after: # Step 3
            :end-before: # End of step 3


        **Output**
        ::

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

    .. tab-item:: TensorFlow
        :sync: tf

        .. include:: ../apiref/tensorflow/bnf.rst
           :start-after: # start-after

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../apiref/onnx/bnf.rst
           :start-after: # start-after
