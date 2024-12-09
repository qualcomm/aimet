.. _glossary:

########
Glossary
########

.. glossary::

   Accelerator
      A :term:`device` with specialized processors such as GPUs dedicated to AI computation.

   Accuracy
      A measure of the percentage of correct predictions made by a model.

   Activation
      The output of a node's activation function, passed as an input to the subsequent layer of the network. 

   Activation Quantization
      The process of converting the output values (:term:`activations`) of nodes from high precision (for example, 32-bit floating point) to lower precision (for example, 8-bit integer), reducing computation and memory requirements during :term:`inference`.

   AdaRound
      A technique used to minimize :term:`quantization` errors by carefully selecting how to round weights. AdaRound is especially powerful in retaining accuracy of models that undergo aggressive :term:`quantization`.

   AI Model Efficiency Toolkit
      An open-source software library developed by the :term:`Qualcomm Innovation Center`, providing a suite of :term:`quantization` and :term:`compression` technologies that reduce the computational load and memory usage of deep learning models.

   AIMET
      :term:`AI Model Efficiency Toolkit`.

   AutoQuant
      A feature that automatically chooses optimal :term:`quantization` parameters to automate the process of model quantization.

   Batch Normalization
      A technique for normalizing a layer's input to accelerate the convergence of deep network models.

   BN
     :term: `Batch Normalization`.

   Batch Normalization Folding (BN Folding)
      A model optimization technique that merges :term:`Batch Normalization` layers to eliminate the need to compute :term:`Batch Normalization` during :term:`inference`. 

   CNN
      :term:`Convolutional neural network`.

   Compression
      The process of reducing the memory footprint and computational requirements of a neural network.

   Convolutional Layer
      A model layer that contains a set of filters that interact with an input to create an :term:`activation` map.

   Convolutional Neural Network
      A deep learning model that uses convolutional layers to extract features from input data, such as images.

   Device
      A portable computation platform such as a mobile phone or a laptop. 

   DLF
      Dynamic Layer Fusion.

   Dynamic Layer Fusion
      A method for merging adjacent layers to decrease computational load during :term:`inference`.

   Edge device
      A device at the "edge" of the network. Typically a personal computation device such as a mobile phone or a laptop.

   Encoding
      The representation of model parameters (weights) and :term:`activations` in a compressed, quantized format. Different encoding schemes embody tradeoffs between model accuracy and efficiency.

   FP32
      32-bit floating-point precision, the default data type for representing weights and :term:`activations` in most deep learning frameworks.

   Inference
      The process of employing a trained AI model for its intended purpose: prediction, classification, content generation, etc.

   INT8
      8-bit integer precision, commonly used by AIMET to reduce the memory size and computational demands during :term:`inference`.

   KL Divergence
      Kullback-Leibler Divergence. A measure of the difference between two probability distributions. Used during :term:`quantization` calibration to maintain a similar distribution of :term:`activations` to the original floating-point model.

   Layer
      How nodes are organized in a model. The nodes in a layer are connected to the previous and subsequent layer via :term:`weights`.
   
   Layer-wise quantization
      A :term:`quantization` method where each layer is quantized independently. Used to achieve balance between model accuracy and computational efficiency by more aggressively compressing layers that have minimal impact on model performance.

   LoRA MobileNet
      A family of :term:`convolutional neural network` architectures developed at Google optimized to operate efficiently with constrained computational resources.

   Model
      A computational structure made up of :term:`layers` of :term:`nodes` connected by :term:`weights`.

   Neural Network Compression Framework
      Another :term:`compression` and optimization toolkit similar to AIMET.

   Node
      A computation unit in a :model:`model`. Each node performs a mathematical function on an input to produce an output. 

   Normalization
      Scaling a feature such as a :term:`layer` to standardize the range of the feature.

   NNCF
      :term:`Neural Network Compression Framework`.

   ONNX
      :term:`Open Neural Network Exchange`.

   Open Neural Network Exchange
      An open-source format for the representation of neural network models across different AI frameworks.

   Per-channel Quantization
      A :term:`quantization` method where each channel of a :term:`convolutional layer` is quantized independently, reducing the quantization error compared to a global quantization scheme.

   Post-Training Quantization
      A technique for applying :term:`quantization` to a neural network after it has been trained using full-precision data, avoiding the need for retraining.

   Pruning
      Systematically removing less important neurons, weights, or connections from a model.

   PTQ
      :term:`Post-Training Quantization`.

   PyTorch
      A open-source deep learning framework developed by Facebook's AI Research lab (FAIR), widely used in research environments.

   QAT
      :term:`Quantization Aware Training`.

   QDO
      Quantize and dequantize operations.

   Qualcomm Innovation Center
      A division of Qualcomm, Inc. responsible for developing advanced technologies and open-source projects, including AIMET.

   Quantization
      A model :term:`compression` technique that reduces the bits used to represent each weight and :term:`activation` in a neural network, typically from floating-point 32-bit numbers to 8-bit integers.

   Quantization-Aware Training
      A technique in which :term:`quantization` is simulated throughout the training process so that the network adapts to the lower precision during training.

   Quantization Simulation
      A tool within AIMET that simulates the effects of :term:`quantization` on a model to predict how quantization will affect the model's performance.

   QuantSim
      :term:`Quantization Simulation`.

   QUIC
      :term:`Qualcomm Innovation Center`.

   Target Hardware Accelerator
      Specialized hardware designed to accelerate AI :term:`inference` tasks. Examples include GPUs, TPUs, and custom ASICs, for example Qualcomm's Cloud AI 100 inference accelerator. 

   Target Runtime
      A model quantized for use on a low bitwidth platform, typically an :term:`edge device`.

   TensorFlow
      A widely-used open-source deep learning framework developed by Google. 

   TorchScript
      An intermediate representation for :term:`PyTorch` models that enables running them independently of the Python environment, making them more suitable for production deployment.

   Variant
      The combination of machine learning framework (:term:`PyTorch`, :term:`TensorFlow`, or :term:`ONNX`) and processor (Nvidia version or CPU) that determines which version of the AIMET API to install.

   Weights
      Parameters that collectively represent features in a model.