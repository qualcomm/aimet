.. _examples-index:

########
Examples
########

AIMET end-to-end examples are **Jupyter Notebooks** that are intended to:

- Familiarize you with the AIMET APIs,
- Demonstrate how to apply AIMET to a pre-trained model from PyTorch, TensorFlow and ONNX frameworks,
- Teach you how to use AIMET quantization and compression techniques.

For a discussion of quantization and compression techniques, see :doc:`Optimization User Guide <../opt-guide/index>`.

For the API reference, see :doc:`API reference<../apiref/index>`

Browse the notebooks
====================

The following tables provide links to viewable HTML versions of the jupyter notebooks for AIMET quantization
and compression features. Instructions after the tables describe how to run the notebooks.

**Model Quantization Examples**

.. list-table::
   :widths: 50 20 20 20
   :header-rows: 1

   * - Features
     - PyTorch
     - TensorFlow
     - ONNX
   * - Quantization simulation (QuantSim)
     - `Link <torch/quantization/qat.ipynb>`_
     - `Link <tensorflow/quantization/keras/qat.ipynb>`_
     - `Link <onnx/quantization/quantsim.ipynb>`_
   * - Quantization-aware training (QAT)
     - `Link <torch/quantization/qat_range_learning.ipynb>`_
     - `Link <tensorflow/quantization/keras/qat_range_learning.ipynb>`_
     - Not implemented.
   * - Cross-layer equalization (CLE)
     - `Link <torch/quantization/cle_bc.ipynb>`_
     - `Link <tensorflow/quantization/keras/quantsim_cle.ipynb>`_
     - `Link <onnx/quantization/cle.ipynb>`_
   * - Adaptive rounding (AdaRound)
     - `Link <torch/quantization/adaround.ipynb>`_
     - `Link <tensorflow/quantization/keras/adaround.ipynb>`_
     - `Link <onnx/quantization/adaround.ipynb>`_
   * - Automatic quantization (AutoQuant)
     - `Link <torch/quantization/autoquant.ipynb>`_
     - `Link <tensorflow/quantization/keras/autoquant.ipynb>`_
     - Not implemented.
   * - Automatic mixed precision (AMP)
     - `Link <torch/quantization/AMP.ipynb>`_
     - `Link <tensorflow/quantization/keras/KerasAMP.ipynb>`_
     - `Link <onnx/quantization/AMP.ipynb>`_
   * - BatchNorm re-estimation
     - `Link <torch/quantization/bn_reestimation.ipynb>`_
     - `Link <tensorflow/quantization/keras/bn_reestimation.ipynb>`_
     - Not implemented.
   * - Quant analyzer
     - `Link <torch/quantization/quant_analyzer.ipynb>`_
     - `Link <tensorflow/quantization/keras/quant_analyzer.ipynb>`_
     - Not implemented.

**Model Compression Examples**

.. list-table::
   :widths: 50 20
   :header-rows: 1

   * - Features
     - PyTorch
   * - Channel Pruning
     - `Link <torch/compression/channel_pruning.ipynb>`_
   * - Spatial SVD
     - `Link <torch/compression/spatial_svd.ipynb>`_
   * - Spatial SVD + Channel Pruning
     - `Link <torch/compression/spatial_svd_channel_pruning.ipynb>`_

Running the notebooks
=====================

To run the notebooks, follow the instructions below.

1. Run the notebook server
--------------------------

1. Install the Jupyter metapackage using the following command.
   (Prepend the command with ``sudo -H`` if necessary to grant admin privilege.)

   .. code-block:: bash

      python3 -m pip install jupyter

2. Start the notebook server as follows:

   .. code-block:: bash

      jupyter notebook --ip=* --no-browser &

   The command generates and displays a URL in the terminal.

3. Copy and paste the URL into your browser.

4. Install AIMET and its dependencies using the instructions in :doc:`AIMET installation </install/index>`.


2. Download the example notebooks and related code
--------------------------------------------------

Set up your workspace using the following steps:

1. Set a workspace path:

   .. code-block:: bash

      WORKSPACE="<absolute_path_to_workspace>"

2. Create and move to the workspace:

   .. code-block:: bash

      mkdir $WORKSPACE && cd $WORKSPACE

3. Identify the release tag (``<release_tag>``) of the AIMET package that you're working with at: https://github.com/quic/aimet/releases.

4. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/quic/aimet.git --branch <release_tag>

5. Update the path environment variable:

   .. code-block:: bash

      export PYTHONPATH=$PYTHONPATH:${WORKSPACE}/aimet

6. The dataloader, evaluator, and trainer used in the examples are for the ImageNet dataset.
   Download the ImageNet dataset from:
   https://www.image-net.org/download.php

3. Run the notebooks
--------------------

1. Navigate to one of the following paths in your local repository directory and launch your
   chosen jupyter notebook (`.ipynb` extension):

**Model quantization notebooks**

- Examples/torch/quantization/
- Examples/tensorflow/quantization/keras/
- Examples/onnx/quantization/

**Model compression notebooks**

- Examples/torch/compression/

2. Follow the instructions in the notebook to execute the code.
