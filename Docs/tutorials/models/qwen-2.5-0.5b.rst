Qwen/Qwen2.5-0.5B-Instruct
==========================

Precision settings:

- Weights: INT4, except for:
    - ``LM Head``: INT8
- Activations: INT16

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
      - 13.14
      - 46.30
      - 00:00:13
      - 3.68
    * - PCQ + SpinQuant + AdaScale
      - ``aimet-torch``
      - ``aimet-onnx``
      - 13.89
      - 44.19
      - 03:19:37
      - 13.37
    * - PCQ + SpinQuant + AdaScale
      - ``aimet-onnx``
      - ``aimet-onnx``
      - 13.82
      - 42.65
      - 01:16:54
      - 34.01
    * - LPBQ + SequentialMSE
      - ``aimet-torch``
      - ``aimet-onnx``
      - 15.32
      - 42.33
      - 00:22:39
      - 14.25
    * - LPBQ + SequentialMSE
      - ``aimet-onnx``
      - ``aimet-onnx``
      - 15.30
      - 43.26
      - 00:11:33
      - 20.43
