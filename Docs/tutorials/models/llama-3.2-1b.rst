meta-llama/Llama-3.2-1B-Instruct
================================

Precision settings:

- Weights: INT4, except for:
    - ``LM Head``: INT8
- Activations: INT16, except for:
    - ``KV Cache``: INT8

Hyperparameters:

- AdaScale: ``num_batches=128``, ``num_iterations=2048``
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
      - 12.14
      - 46.06
      - 00:00:14
      - 6.34
    * - PCQ + SpinQuant + AdaScale
      - ``aimet-torch``
      - ``aimet-onnx``
      - 13.67
      - 42.25
      - 02:31:06
      - 20.89
    * - PCQ + SpinQuant + AdaScale
      - ``aimet-onnx``
      - ``aimet-onnx``
      - 13.68
      - 41.82
      - 01:53:17
      - 46.38
    * - LPBQ + SequentialMSE
      - ``aimet-torch``
      - ``aimet-onnx``
      - 14.07
      - 43.09
      - 00:44:38
      - 28.52
    * - LPBQ + SequentialMSE
      - ``aimet-onnx``
      - ``aimet-onnx``
      - 13.84
      - 43.53
      - 00:20:44
      - 34.79
