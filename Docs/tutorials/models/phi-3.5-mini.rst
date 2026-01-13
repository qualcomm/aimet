microsoft/Phi-3.5-mini-instruct
===============================

Precision settings:

- Weights: INT4, except for:
    - ``LM Head``: INT8
- Activations: INT16, except for:
    - ``KV Cache``: INT8

Hyperparameters:

- AdaScale: ``num_batches=128``, ``num_iterations=256``
- SequentialMSE: ``num_batches=20``
- Calibration: ``num_batches=20``


.. list-table::
    :widths: 50 18 18 3 3 5 3
    :header-rows: 1

    * - Technique
      - Quantized With
      - Evaluated On
      - PPL
      - MMLU
      - Time (hh:mm:ss)
      - CUDA (GB)
    * - FP32
      - N/A
      - Both
      - 5.77
      - 68.89
      - 00:00:08
      - 16.17
    * - PCQ + SpinQuant + AdaScale
      - ``aimet-torch``
      - ``aimet-onnx``
      - 6.58
      - 62.62
      - 04:16:53
      - 48.03
    * - PCQ + SpinQuant + AdaScale
      - ``aimet-onnx``
      - ``aimet-onnx``
      - 6.50
      - 62.51
      - 01:51:43
      - 61.85
    * - LPBQ + SequentialMSE
      - ``aimet-torch``
      - ``aimet-onnx``
      - 6.45
      - 64.63
      - 02:03:41
      - 37.64
    * - LPBQ + SequentialMSE
      - ``aimet-onnx``
      - ``aimet-onnx``
      - 6.41
      - 63.90
      - 01:32:36
      - 75.62
