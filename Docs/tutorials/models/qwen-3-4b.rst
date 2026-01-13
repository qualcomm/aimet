Qwen/Qwen3-4B
=============

Precision settings:

- Weights: INT4, except for:
    - ``LM Head``: INT8
- Activations: INT16, except for:
    - ``KV Cache``: INT8

Hyperparameters:

- AdaScale: ``num_batches=128``, ``num_iterations=512``
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
      - 12.41
      - 70.06
      - 00:00:10
      - 17.02
    * - PCQ + SpinQuant + AdaScale
      - ``aimet-torch``
      - ``aimet-onnx``
      - 13.85
      - 65.07
      - 06:41:32
      - 47.71
    * - PCQ + AdaScale
      - ``aimet-onnx``
      - ``aimet-onnx``
      - 13.79
      - 62.33
      - 04:34:22
      - 71.3
    * - LPBQ + SequentialMSE
      - ``aimet-torch``
      - ``aimet-onnx``
      - 13.10
      - 65.66
      - 02:41:48
      - 39.42
    * - LPBQ + SequentialMSE
      - ``aimet-onnx``
      - ``aimet-onnx``
      - 12.77
      - 65.36
      - 01:35:29
      - 63.61
