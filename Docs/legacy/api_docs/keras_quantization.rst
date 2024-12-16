==================================
AIMET TensorFlow Quantization APIs
==================================

In order to make full use of AIMET Quantization features, there are several guidelines users are encouraged to follow
when defining Keras models. AIMET provides APIs which can automate some of the model definition changes and checks
whether AIMET Quantization features can be applied on Keras model.

.. toctree::
   :titlesonly:
   :hidden:

    Model Guidelines <keras_model_guidelines>
    Model Preparer API<keras_model_preparer>
    Quant Analyzer API<keras_quant_analyzer>
    Quantization Simulation API<keras_quantsim>
    Adaptive Rounding API<keras_adaround>
    Cross-Layer Equalization API<keras_cross_layer_equalization>
    BN Re-estimation APIs<keras_batchnorm_re_estimation>

Users should first invoke Model Preparer API before using any of the AIMET Quantization features.
   - :ref:`Model Guidelines<api-keras-model-guidelines>`: Guidelines for defining Keras models
   - :ref:`Model Preparer API<api-keras-model-preparer>`: Allows user to automate model definition changes

AIMET Quantization for Keras provides the following functionality
   - :ref:`Quant Analyzer API<api-keras-quant-analyzer>`: Analyzes the model and points out sensitive layers to quantization
   - :ref:`Quantization Simulation API<api-keras-quantsim>`: Allows ability to simulate inference and training on quantized hardware
   - :ref:`Adaptive Rounding API<api-keras-adaround>`: Post-training quantization technique to optimize rounding of weight tensors
   - :ref:`Cross-Layer Equalization API<api-keras-cle>`: Post-training quantization technique to equalize layer parameters
   - :ref:`BatchNorm Re-estimation API<api-keras-bn-reestimation>`: Quantization-aware training technique to counter potential instability of batchnorm statistics (i.e. running mean and variance)
