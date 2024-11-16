.. _glossary:

########
Glossary
########

.. glossary::

   Accelerator
      A device with specialized processors such as GPUs dedicated to AI calculations.

   Activation
      The output of a node's activation function, passed as an input to the subsequent layer of the network. 

   Activation Quantization
      The process of converting the output values (activations) of nodes from high precision (for example, 32-bit floating point) to lower precision (for example, 8-bit integer), reducing computation and memory requirements during inference.

   AdaRound
      A technique used to minimize quantization errors by carefully selecting how to round weights AdaRound is especially powerful in retaining accuracy of models that undergo aggressive quantization.

   AI Model Efficiency Toolkit
      An open-source software library developed by the [Qualcomm Innovation Center](#qualcomm-innovation-center), providing a suite of quantization and compression technologies that reduce the computational load and memory usage of deep learning models.

   AIMET
      AI Model Efficiency Toolkit.

   AutoQuant
      A feature that automatically chooses optimal quantization parameters to automate the process of model quantization.

   Batch Normalization
      A technique for normalizing a layer's input to accelerate the convergence of deep network models.

   BN
      Batch Normalization.

   Batch Normalization Folding (BN Folding)
      A model optimization technique that merges Batch Normalization layers to eliminate the need to compute Batch Normalization during inference. 

   CNN
      Convolutional neural network.

   Compression
      The process of reducing the memory footprint and computational requirements of a neural network.

   Convolutional Layer
      A model layer that contains a set of filters that interact with an input to create an activation map.

   Convolutional Neural Network
      A deep learning model that uses convolutional layers to extract features from input data, such as images.

   Device
      A portable computation platform such as a mobile phone or a laptop. 

   DLF
      Dynamic Layer Fusion.

   Dynamic Layer Fusion
      A method for merging adjacent layers to decrease computational load during inference.

   Edge device
      A device at the "edge" of the network. Typically a personal computation device such as a mobile phone or a laptop.

   Encoding
      The representation of model parameters (weights) and activations in a compressed, quantized format. Different encoding schemes embody tradeoffs between model accuracy and efficiency.

   FP32
      32-bit floating-point precision, the default data type for representing weights and activations in most deep learning frameworks.

   Inference
      The process of employing a trained AI model for its intended purpose: prediction, classification, content generation, etc.

   INT8
      8-bit integer precision, commonly used by AIMET to reduce the memory size and computational demands during inference.

   KL Divergence
      Kullback-Leibler Divergence. A measure of the difference between two probability distributions. Used during quantization calibration to maintain a similar distribution of activations to the original floating-point model.

   Layer-wise Quantization
      A quantization method where each layer is quantized independently. Used to achieve balance between model accuracy and computational efficiency by more aggressively compressing layers that have minimal impact on model performance.

   LoRA MobileNet
      A family of convolutional neural network architectures developed at Google optimized to operate efficiently with constrained computational resources.

   Neural Network Compression Framework
      Another compression and optimization toolkit similar to AIMET.

   NNCF
      Neural Network Compression Framework.

   ONNX
      Open Neural Network Exchange.

   Open Neural Network Exchange
      An open-source format for the representation of neural network models across different AI frameworks.

   Per-channel Quantization
      A quantization method where each channel of a convolutional layer is quantized independently, reducing the quantization error compared to a global quantization scheme.

   Post-Training Quantization
      A technique for applying quantization to a neural network after it has been trained using full-precision data, avoiding the need for retraining.

   Pruning
      Systematically removing less important neurons, weights, or connections from a model.

   PTQ
      Post-Training Quantization.

   PyTorch
      A open-source deep learning framework developed by Facebook's AI Research lab (FAIR), widely used in research environments.

   QAT
      Quantization Aware Training.

   Qualcomm Innovation Center
      A division of Qualcomm, Inc. responsible for developing advanced technologies and open-source projects, including AIMET.

   Quantization
      A model compression technique that reduces the bits used to represent each weight and activation in a neural network, typically from floating-point 32-bit numbers to 8-bit integers.

   Quantization-Aware Training
      A technique in which quantization is simulated throughout the training process so that the network adapts to the lower precision during training.

   QuantSim
      Quantization Simulation.
    
   Quantization Simulation
      A tool within AIMET that simulates the effects of quantization on a model to predict how quantization will affect the model's performance.

   QUIC
      Qualcomm Innovation Center.

   Target Hardware Accelerator
      Specialized hardware designed to accelerate AI inference tasks. Examples include GPUs, TPUs, and custom ASICs, for example Qualcomm's Cloud AI 100 inference accelerator. 

   TensorFlow
      A widely-used open-source deep learning framework developed by Google. 

   TorchScript
      An intermediate representation for PyTorch models that enables running them independently of the Python environment, making them more suitable for production deployment.

   Variant
      The combination of machine learning framework (PyTorch, TensorFlow, or ONNX) and processor (Nvidia version or CPU) that determines which version of the AIMET API to install.