.. _ug-examples:

.. image:: ../images/logo-quic-on@h68.png

##############
AIMET examples
##############

AIMET examples are *Jupyter Notebooks* that are intended to:

- Familiarize you with the AIMET APIs
- Demonstrate how to apply AIMET to a model
- Teach you how to use AIMET quantization and compression techniques

For a discussion of quantization techniques, see :doc:`/user_guide/model_quantization`.

For a discussion of compression techniques, see :doc:`/user_guide/model_compression`.

For the API reference, see:

- :doc:`/api_docs/torch` for PyTorch
- :doc:`/api_docs/keras` for TensorFlow
- :doc:`/api_docs/onnx` for ONNX

Browse the notebooks
====================

The following tables provide links to viewable versions of the notebooks for AIMET quantization and compression features. Instructions after the tables describe how to run the notebooks.

**Model Quantization Examples**

.. list-table::
   :widths: 32 12 12 12
   :header-rows: 1

   * - Feature
     - PyTorch
     - TensorFlow
     - ONNX
   * - QuantSim / Quantization-Aware Training (QAT)
     - `Link <../Examples/torch/quantization/qat.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/keras/qat.ipynb>`_
     - `Link <../Examples/onnx/quantization/quantsim.ipynb>`_  (no training)
   * - QAT with Range Learning
     - `Link <../Examples/torch/quantization/qat_range_learning.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/keras/qat_range_learning.ipynb>`_
     -
   * - Cross-Layer Equalization (CLE)
     - `Link <../Examples/torch/quantization/cle_bc.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/keras/quantsim_cle.ipynb>`_
     - `Link <../Examples/onnx/quantization/cle.ipynb>`_
   * - Adaptive Rounding (AdaRound)
     - `Link <../Examples/torch/quantization/adaround.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/keras/adaround.ipynb>`_
     - `Link <../Examples/onnx/quantization/adaround.ipynb>`_
   * - AutoQuant
     - `Link <../Examples/torch/quantization/autoquant.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/keras/autoquant.ipynb>`_
     -

|
**Model Compression Examples**

.. list-table::
   :widths: 40 12
   :header-rows: 1

   * - Feature
     - PyTorch
   * - Channel Pruning
     - `Link <../Examples/torch/compression/channel_pruning.ipynb>`_
   * - Spatial SVD
     - `Link <../Examples/torch/compression/spatial_svd.ipynb>`_
   * - Spatial SVD + Channel Pruning
     - `Link <../Examples/torch/compression/spatial_svd_channel_pruning.ipynb>`_


|
Running the notebooks
=====================

To run the notebooks, follow the instructions below.

1. Run the notebook server
--------------------------

1. Install the Jupyter metapackage using the following command. 
   (Prepend the command with ``sudo -H`` if necessary to grant admin privilege.)

   .. code-block:: shell

      python3 -m pip install jupyter

2. Start the notebook server as follows:

   .. code-block:: shell
    
      jupyter notebook --ip=* --no-browser &

   The command generates and displays a URL in the terminal. 
   
3. Copy and paste the URL into your browser.

4. Install AIMET and its dependencies using the instructions in :doc:`AIMET installation </install/index>`.


2. Download the example notebooks and related code
--------------------------------------------------

Set up your workspace using the following steps:

1. Set a workspace path:

   .. code-block:: shell
      
      WORKSPACE="<absolute_path_to_workspace>"

2. Create and move to the workspace:

   .. code-block:: shell
    
      mkdir $WORKSPACE && cd $WORKSPACE

3. Identify the release tag (``<release_tag>``) of the AIMET package that you're working with at: https://github.com/quic/aimet/releases.

4. Clone the repository:

   .. code-block:: shell

      git clone https://github.com/quic/aimet.git --branch <release_tag>

5. Update the path environment variable:

   .. code-block:: shell

      export PYTHONPATH=$PYTHONPATH:${WORKSPACE}/aimet

6. The dataloader, evaluator, and trainer used in the examples are for the ImageNet dataset. 
   Download the ImageNet dataset from: 
   https://www.image-net.org/download.php


3. Run the notebooks
--------------------

1. Navigate to one of the following paths in the local repository directory and launch your 
   chosen Jupyter Notebook (`.ipynb` extension):

   - `Examples/torch/quantization/`
   - `Examples/torch/compression/`
   - `Examples/tensorflow/quantization/keras/`

2. Follow the instructions in the notebook to execute the code.

