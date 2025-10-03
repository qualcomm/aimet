.. _examples-index:

########
Example Notebooks
########

AIMET’s end-to-end examples are provided as **Jupyter Notebooks** designed to help you:

- Get familiar with the AIMET APIs
- Learn how to apply AIMET to pre-trained models from PyTorch, ONNX, and TensorFlow frameworks
- Understand and implement AIMET’s quantization and compression techniques

For a deeper dive into these techniques, refer to the Optimization User Guide.
For detailed API documentation, see the API Reference, see :doc:`API reference<../apiref/index>`

Browse the notebooks
====================

The tables below provide links to viewable HTML versions of Jupyter Notebooks that showcase AIMET’s quantization and compression features.

Following the tables, you’ll find step-by-step instructions on how to run these notebooks locally or in your preferred environment.

**Model Quantization Examples**

.. list-table::
   :widths: 50 20 20
   :header-rows: 1

   * - Features
     - ONNX
     - PyTorch
   * - Quantization simulation (QuantSim)
     - `Link <notebooks/onnx/quantization/quantsim.ipynb>`_
     - `Link <notebooks/torch/quantization/qat.ipynb>`_
   * - Quantization-aware training (QAT)
     - Not supported.
     - `Link <notebooks/torch/quantization/qat_range_learning.ipynb>`_
   * - Cross-layer equalization (CLE)
     - `Link <notebooks/onnx/quantization/cle.ipynb>`_
     - `Link <notebooks/torch/quantization/cle_bc.ipynb>`_
   * - Adaptive rounding (AdaRound)
     - `Link <notebooks/onnx/quantization/adaround.ipynb>`_
     - `Link <notebooks/torch/quantization/adaround.ipynb>`_
   * - Automatic quantization (AutoQuant)
     - Not implemented.
     - `Link <notebooks/torch/quantization/autoquant.ipynb>`_
   * - Automatic mixed precision (AMP)
     - `Link <notebooks/onnx/quantization/AMP.ipynb>`_
     - `Link <notebooks/torch/quantization/AMP.ipynb>`_
   * - BatchNorm re-estimation
     - Not implemented.
     - `Link <notebooks/torch/quantization/bn_reestimation.ipynb>`_
   * - Quant analyzer
     - `Link <notebooks/onnx/quantization/quant_analyzer.ipynb>`_
     - `Link <notebooks/torch/quantization/quant_analyzer.ipynb>`_

**Model Compression Examples**

.. list-table::
   :widths: 50 20
   :header-rows: 1

   * - Features
     - PyTorch
   * - Channel Pruning
     - `Link <notebooks/torch/compression/channel_pruning.ipynb>`_
   * - Spatial SVD
     - `Link <notebooks/torch/compression/spatial_svd.ipynb>`_
   * - Spatial SVD + Channel Pruning
     - `Link <notebooks/torch/compression/spatial_svd_channel_pruning.ipynb>`_

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

- Examples/onnx/quantization/
- Examples/torch/quantization/

**Model compression notebooks**

- Examples/torch/compression/

2. Follow the instructions in the notebook to execute the code.

.. toctree::
  :hidden:

    Quantization Workflow <quantization-workflow>
    QuantAnalyzer <quant_analyzer>
    Automatic Mixed Precision <amp>
