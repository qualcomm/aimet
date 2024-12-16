.. _ug-release-notes:

###################
AIMET Release Notes
###################

Release Notes for Qualcomm AI Model Efficiency ToolKit (AIMET)

1.22.2
~~~~~~

*Tensorflow*

- Added support for supergroups : MatMul + Add
- Added support for TF-Slim BN name with backslash
- Added support for Depthwise + Conv in CLS

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.22.2
- User guide: https://quic.github.io/aimet-pages/releases/1.22.2/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.22.2/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.22.1
~~~~~~

- Added support for QuantizableMultiHeadAttention for PyTorch nn.transformer layers by @quic-kyuykim
- Support functional conv2d in model preparer by @quic-kyuykim
- Enable qat with multi gpu by @quic-mangal
- Optimize forward pass logic of PyTorch QAT 2.0 by @quic-geunlee
- Fix functional depthwise conv support on model preparer by @quic-kyuykim
- Fix bug in model validator to correctly identify functional ops in leaf module by @quic-klhsieh
- Support dynamic functional conv2d in model preparer by @quic-kyuykim
- Added updated default runtime config, also a per-channel one. Fixed n… by @quic-akhobare
- Include residing module info in model validator by @quic-klhsieh
- Support for Keras MultiHeadAttention Layer by @quic-ashvkuma

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.22.1
- User guide: https://quic.github.io/aimet-pages/releases/1.22.1/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.22.1/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html


1.22.0
~~~~~~

- Support for simulation and QAT for PyTorch transformer models (including support for torch.nn mha and encoder layers)

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.22.0
- User guide: https://quic.github.io/aimet-pages/releases/1.22.0/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.22.0/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.21.0
~~~~~~

- New feature: PyTorch QuantAnalyzer - Visualize per-layer sensitivity and per-quantizer PDF histograms
- New feature: TensorFlow AutoQuant - Automatically apply various AIMET post-training quantization techniques
- PyTorch QAT with Range Learning: Added support for Per Channel Quantization
- PyTorch: Enabled exporting of encodings for multi-output leaf module
- TensorFlow Adaround
    - Added ability to use configuration file in API to adapt to a specific runtime target
    - Added Per-Channel Quantization support
- TensorFlow QuantSim: Added support for FP16 inference and QAT
- TensorFlow Per Channel Quantization
    - Fixed speed and accuracy issues
    - Fixed zero accuracy for 16-bits per channel quantization
    - Added support for DepthWise Conv2d Op
- Multiple other bug fixes

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.21.0
- User guide: https://quic.github.io/aimet-pages/releases/1.21.0/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.21.0/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.20.0
~~~~~~

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.20.0
- User guide: https://quic.github.io/aimet-pages/releases/1.20.0/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.20.0/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html


1.19.1.py37
~~~~~~~~~~~

- PyTorch: Added CLE support for Conv1d, ConvTranspose1d and Depthwise Separable Conv1d layers
- PyTorch: Added High-Bias Fold support for Conv1D layer
- PyTorch: Modified Elementwise Concat Op to support any number of tensors
- Minor dependency fixes

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.19.1.py37
- User guide: https://quic.github.io/aimet-pages/releases/1.19.1/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.19.1/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.19.1
~~~~~~

- PyTorch: Added CLE support for Conv1d, ConvTranspose1d and Depthwise Separable Conv1d layers
- PyTorch: Added High-Bias Fold support for Conv1D layer
- PyTorch: Modified Elementwise Concat Op to support any number of tensors
- Minor dependency fixes

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.19.1
- User guide: https://quic.github.io/aimet-pages/releases/1.19.1/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.19.1/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.18.0.py37
~~~~~~~~~~~

- Multiple bug fixes
- Additional feature examples for PyTorch and TensorFlow

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.18.0.py37
- User guide: https://quic.github.io/aimet-pages/releases/1.18.0/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.18.0/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.18.0
~~~~~~

- Multiple bug fixes
- Additional feature examples for PyTorch and TensorFlow

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.18.0
- User guide: https://quic.github.io/aimet-pages/releases/1.18.0/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.18.0/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.17.0.py37
~~~~~~~~~~~

- Add Adaround TF feature
- Added Examples for Torch quantization, and Channel Pruning & Spatial SVD compression

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.17.0.py37
- User guide: https://quic.github.io/aimet-pages/releases/1.17.0.py37/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.17.0.py37/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.17.0
~~~~~~

- Add Adaround TF feature
- Added Examples for Torch quantization, and Channel Pruning & Spatial SVD compression

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.17.0
- User guide: https://quic.github.io/aimet-pages/releases/1.17.0/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.17.0/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.16.2.py37
~~~~~~~~~~~

- Added a new post-training quantization feature called AdaRound, which stands for AdaptiveRounding
- Quantization simulation and QAT now also support recurrent layers (RNN, LSTM, GRU)

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.16.2.py37
- User guide: https://quic.github.io/aimet-pages/releases/1.16.2.py37/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.16.2.py37/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.16.2
~~~~~~

- Added a new post-training quantization feature called AdaRound, which stands for AdaptiveRounding
- Quantization simulation and QAT now also support recurrent layers (RNN, LSTM, GRU)

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.16.2
- User guide: https://quic.github.io/aimet-pages/releases/1.16.2/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.16.2/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.16.1.py37
~~~~~~~~~~~

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.16.1.py37
- User guide: https://quic.github.io/aimet-pages/releases/1.16.1.py37/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.16.1.py37/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.16.1
~~~~~~

- Added separate packages for CPU and GPU models. This allows users with CPU-only hosts to run AIMET.
- Added separate packages for PyTorch and TensorFlow. Reduces the number of dependencies that users would need to install.

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.16.1
- User guide: https://quic.github.io/aimet-pages/releases/1.16.1/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.16.1/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.16.0
~~~~~~

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.16.0
- User guide: https://quic.github.io/aimet-pages/releases/1.16.0/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.16.0/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.14.0
~~~~~~

**Documentation**

- Release main page: https://github.com/quic/aimet/releases/tag/1.14.0
- User guide: https://quic.github.io/aimet-pages/releases/1.14.0/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.14.0/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html

1.13.0
~~~~~~

**Documentation**

- User guide: https://quic.github.io/aimet-pages/releases/1.13.0/user_guide/index.html
- API documentation: https://quic.github.io/aimet-pages/releases/1.13.0/api_docs/index.html
- Documentation main page: https://quic.github.io/aimet-pages/index.html
