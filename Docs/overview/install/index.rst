.. _install-index:

############
Installation
############

This page provides instructions for installing the latest version of AIMET across supported frameworks: ONNX, PyTorch.

Please select one installation method that best suits your environment:

- :ref:`Install via PyPI <default-package>`
- :ref:`Alternative packages from Github release <alternative-packages>`
- :ref:`Build from source <building-from-source>`

.. _default-package:

PyPI
====

Install AIMET from PyPI

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
      :sync: onnx

      .. code-block:: bash

        pip install aimet-onnx
        # Optional: To accelerate quantization with CUDA
        pip install onnxruntime-gpu

    .. tab-item:: PyTorch
      :sync: torch

      .. code-block:: Bash

        pip install aimet-torch


.. note::
    aimet_tf is not available on PyPI, please download aimet_tf package from https://github.com/quic/aimet/releases

.. _alternative-packages:

Alternative packages
====================

Install the latest version of AIMET for supported framework and compute platforms including
ONNX and PyTorch from the .whl files hosted at https://github.com/quic/aimet/releases.

Prerequisites
-------------

The AIMET package requires the following host platform setup. The following prerequisites apply
to all frameworks variants.

* 64-bit Intel x86-compatible processor
* OS: Ubuntu 22.04 LTS
* Python 3.10
* For GPU variants:
    * Nvidia GPU card (Compute capability 5.2 or later)
    * Nvidia driver version 455 or later (using the latest driver is recommended; both CUDA and cuDNN are supported)

Choose and install a package
----------------------------

Use one of the following commands to install AIMET based on your choice of framework and compute platform.

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        With CUDA 12.x:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_onnx-|version|\+cu126\ |whl_suffix|

        With CPU only:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_onnx-|version|\+cpu\ |whl_suffix|

    .. tab-item:: PyTorch
        :sync: torch

        With CUDA 12.x:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_torch-|version|\+cu126\ |torch_whl_suffix|

        With CPU only:

        .. parsed-literal::

            python3 -m pip install |download_url|\ |version|/aimet_torch-|version|\+cpu\ |torch_whl_suffix|

Verifying the installation
--------------------------

Verify your installation using the following instructions.

**Step 1:** Handle imports and other setup.

.. code-block:: python

    import numpy as np
    from aimet_common import libpymo

    x = np.random.randn(100)

    quant_scheme = libpymo.QuantizationMode.QUANTIZATION_TF
    analyzer = libpymo.EncodingAnalyzerForPython(quant_scheme)

**Step 2:** Compute scale and offset.

.. code-block:: python

    bitwidth = 8
    is_symmetric, strict_symmetric, unsigned_symmetric = True, False, True
    use_cuda = False
    analyzer.updateStats(x, use_cuda)
    encoding, _ = analyzer.computeEncoding(bitwidth, is_symmetric, strict_symmetric, unsigned_symmetric)

    print(f'Min: {encoding.min}, Max: {encoding.max}, Scale(delta): {encoding.delta}, Offset: {encoding.offset}')

The encodings values should be similar to the one shown below.

.. rst-class:: script-output

    .. code-block:: none

        Min: -3.3734087606114667, Max: 3.3470540046691895, Scale(delta): 0.026354755942277083, Offset: -128.0

**Step 3:** Perform quantize-dequantize.

.. code-block:: python

    quantizer = libpymo.TensorQuantizationSimForPython()
    out = quantizer.quantizeDequantize(x,
                                       encoding,
                                       libpymo.RoundingMode.ROUND_NEAREST,
                                       bitwidth,
                                       use_cuda)
    print(out)

The quantized-dequantized output should be similar to the one shown below.

.. rst-class:: script-output

    .. code-block:: none

        [-1.291383    0.36896658  1.0541903  -1.2123188  -2.2137995   1.2650282
         -0.23719281  0.10541902  0.50074035 -0.05270951 -0.94877124  0.
          0.10541902  0.52709514 -0.7115784   2.2401543  -0.34261182  2.0293162
          0.34261182 -0.6061594  -0.36896658 -0.6588689  -1.5022211  -0.10541902
         -1.4758663  -0.8433522   0.7115784  -0.23719281  0.44803086 -0.94877124
          0.18448329 -1.0014807   0.55344987 -0.13177378  0.15812853 -0.7115784
         -0.4216761   1.1068997  -0.07906426  1.6603496   0.55344987 -0.47438562
         -0.6325141   0.4216761  -1.4495116   1.5549306  -0.6325141  -1.2123188
          0.50074035  1.291383    0.07906426 -1.2123188  -2.0820258   1.0014807
         -0.18448329 -0.4216761   1.0278355  -0.21083805  0.52709514  1.6867044
         -0.68522364  1.0278355  -0.55344987 -0.26354757  0.10541902 -0.02635476
          0.6588689  -0.34261182 -0.05270951  3.347054    0.07906426 -1.080545
         -0.57980466  1.4231569  -0.6588689   1.291383   -0.13177378  0.31625706
         -0.36896658  0.05270951 -0.81699747 -1.4231569  -1.1068997  -0.68522364
          0.7115784  -1.2650282  -0.7115784   0.50074035  0.28990233 -0.73793316
          0.21083805  2.4246376  -0.15812853  0.52709514 -0.02635476 -0.13177378
         -1.8711877   0.4216761  -0.55344987 -0.76428795]

Old versions
------------

You can also view the release notes for older AIMET versions at https://github.com/quic/aimet/releases.
Follow the documentation corresponding to that release to select and install the appropriate AIMET package.

.. _building-from-source:

Building from source
====================

For most users, the recommended way to install AIMET is by using the pre-built package available through the pip package manager, as it provides the most straightforward and reliable experience.

However, if you prefer to work with the latest source code or plan to contribute to AIMET development, you’ll need to build it from source.
To do so, follow the steps outlined for building the latest AIMET codebase manually, see :ref:`Build AIMET from source <build-from-source>`.

.. |torch_whl_suffix| replace:: \-py310-none-any.whl
.. |whl_suffix| replace:: \-cp310-cp310-manylinux_2_34_x86_64.whl
.. |download_url| replace:: \https://github.com/quic/aimet/releases/download/
