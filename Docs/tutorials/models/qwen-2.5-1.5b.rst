Qwen/Qwen2.5-1.5B-Instruct
==========================

Precision settings:

- Weights: INT4, except for:
    - ``LM Head``: INT8
- Activations: INT16

Hyperparameters:

- AdaScale: ``num_batches=128``, ``num_iterations=1024``
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
      - 54.65
      - 00:00:10
      - 7.78
    * - PCQ + SpinQuant + AdaScale
      - ``aimet-torch``
      - ``aimet-onnx``
      - 13.57
      - 49.81
      - 03:03:17
      - 22.62
    * - PCQ + SpinQuant + AdaScale
      - ``aimet-onnx``
      - ``aimet-onnx``
      - 13.35
      - 50.27
      - 02:13:33
      - 42.97
    * - LPBQ + SequentialMSE
      - ``aimet-torch``
      - ``aimet-onnx``
      - 14.86
      - 49.25
      - 01:07:43
      - 26.01
    * - LPBQ + SequentialMSE
      - ``aimet-onnx``
      - ``aimet-onnx``
      - 14.33
      - 49.97
      - 00:37:52
      - 34.40
