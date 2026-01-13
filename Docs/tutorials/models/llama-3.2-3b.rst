meta-llama/Llama-3.2-3B-Instruct
================================

Precision settings:

- Weights: INT4, except for:
    - ``LM Head``: INT8
- Activations: INT16, except for:
    - ``KV Cache``: INT8

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
      - 10.13
      - 60.74
      - 00:00:10
      - 13.90
    * - PCQ + SpinQuant + AdaScale
      - ``aimet-torch``
      - ``aimet-onnx``
      - 11.01
      - 58.09
      - 06:35:22
      - 41.24
    * - PCQ + AdaScale
      - ``aimet-onnx``
      - ``aimet-onnx``
      - 11.14
      - 56.79
      - 04:49:36
      - 47.35
    * - LPBQ + SequentialMSE
      - ``aimet-torch``
      - ``aimet-onnx``
      - 10.69
      - 59.08
      - 02:41:44
      - 51.11
    * - LPBQ + SequentialMSE
      - ``aimet-onnx``
      - ``aimet-onnx``
      - 10.55
      - 59.29
      - 01:13:12
      - 59.41
