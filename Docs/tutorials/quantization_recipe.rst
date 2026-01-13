.. include:: ../abbreviation.txt

.. _quantization-genai-recipe:


#############################
Quantization recipes for LLMs
#############################


This document presents the quantization and evaluation workflow for large language models (LLMs)
models using ``aimet-torch`` and ``aimet-onnx``. The objective is to communicate performance expectations through
two reference recipes applied to the following LLMs:

- `LLaMA 3.2 1B Instruct <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`_
- `LLaMA 3.2 3B Instruct <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>`_
- `Qwen 2.5 0.5B Instruct <https://huggingface.co/Qwen/Qwen2.5-0.5B>`_
- `Qwen 2.5 1.5B Instruct <https://huggingface.co/Qwen/Qwen2.5-1.5B>`_
- `Qwen 3 4B <https://huggingface.co/Qwen/Qwen3-4B>`_
- `Phi 3.5 mini instruct <https://huggingface.co/microsoft/Phi-3.5-mini-instruct>`_

The exported artifacts from the recipes are not directly compatible with |qnn|_ (QAIRT) and require additional adaptation steps for deployment on the target hardware. Refer to the `model adaptation guide <https://github.com/quic/ai-hub-models/blob/main/tutorials/llm/onboarding.md>`_ for more details when deploying on the target hardware.


System Requirements
===================

The quantization process requires a machine with:

- Operating System: Linux
- Hardware: CUDA-enabled GPU

GPU Memory Requirements:

- Minimum: 40GB VRAM


Recipes
=======

We present two recipes for INT4 weights, INT16 activations quantization using combinations of :ref:`Post-Training Quantization (PTQ) <featureguide-index>` techniques available in ``aimet-torch`` and ``aimet-onnx``:

#. PCQ + SpinQuant + AdaScale
    - :ref:`Per-Channel Quantization (PCQ) <techniques-ptq>` - Uses per-output channel scales for weights on linear layers.
    - :ref:`SpinQuant <ptq-spinquant>` - A PTQ technique that improves the accuracy by inserting rotations at specific points in the model to mitigate activation outliers.
    - :ref:`AdaScale <ptq-adascale>` - A PTQ technique that enhances accuracy by introducing learnable parameters in the weight quantizers and performing Block-wise Knowledge Distillation (BKD) against FP outputs.

#. LPBQ + SequentialMSE
    - :ref:`Low Power Blockwise Quantization (LPBQ) <techniques-lpbq>`  - Applies blockwise quantization (``block_size=64``) for weights on linear layers.
    - :ref:`SequentialMSE <ptq-seq-mse>` - Calibrates layer-by-layer to minimize Mean Square Error (MSE) between quantized and FP outputs.

To maintain accuracy, activations are primarily kept at INT16, with a mixed-precision profile using INT8 activations selectively where feasible—such as for the KV cache.


Workflow Overview
=================

#. Load the HuggingFace model
    - Start by loading the pretrained model using HuggingFace ``transformers`` library.
#. Apply the selected Quantization recipe
    - Use ``aimet-torch`` for `PyTorch <https://pytorch.org/>`_ based workflows or
    - Use ``aimet-onnx`` for `ONNX <https://onnx.ai/>`_ based workflows.
    - .. note::

        The :ref:`SpinQuant <ptq-spinquant>` technique is currently available only in ``aimet-torch``. You can apply SpinQuant on the FP32 model before exporting it to ONNX, and then continue the workflow using ``aimet-onnx``.
#. Compute Activations encodings
    - Both ``aimet-torch`` and ``aimet-onnx`` compute activation encodings using representative data. In this tutorial, we use `WikiText (English) <https://github.com/quic/aimet/blob/develop/GenAITests/shared/helpers/datasets.py>`_ for calibration.
    - For ``aimet-torch`` only:
        - Due to PyTorch limitations, certain functional operations (``torch.nn.functional``) cannot have quantizers inserted. This makes implementing a mixed-precision profile (e.g., KV Cache in INT8) challenging.
        - To address this, include ``aimet-onnx`` evaluation step within the ``aimet-torch`` workflow. ``aimet-onnx`` provides a static graph, ensuring correct quantizer insertion for all activations and delivering a more accurate quantization simulation.
#. Export for deployment
    -   Export the ONNX model along with the encodings file for the on-target inference.


Quick Start
===========

This section provides a quick example of applying a quantization recipe using either ``aimet-torch`` or ``aimet-onnx``.

In this tutorial, we apply the quantization recipe to the `Llama 3.2 1B` model. The steps work for all fine-tuned variants that share the same tokenizer and network architecture.

The example scripts are designed to be `flattened`, so all AIMET API calls and HuggingFace API calls are visible at the top level.

To understand how this works under the hood for PyTorch and ONNX models using the same driver code, refer to the `Generator <https://github.com/quic/aimet/tree/develop/GenAITests#how-it-all-works>`_ class in ``GenAITests``.

Quantize
--------

Example: Apply Recipe 1 (pcq_spinquant_adascale)

Using ``aimet-torch``:

.. code-block:: Python

    python -m Examples.torch.quantize \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --recipe "pcq_spinquant_adascale" \
     --export-path "./torch_pcq" \
     --adascale-num-batches 128 --adascale-num-iterations 2048



Using ``aimet-onnx``:

.. code-block:: Python

    python -m Examples.onnx.quantize \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --recipe "pcq_spinquant_adascale" \
     --export-path "./onnx_pcq" \
     --adascale-num-batches 128 --adascale-num-iterations 2048

Example: Apply Recipe 2 (lpbq_seqmse)

Using ``aimet-torch``:

.. code-block:: Python

    python -m Examples.torch.quantize \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --recipe "lpbq_seqmse" \
     --export-path "./torch_lpbq" \
     --seqmse-num-batches 20


Using ``aimet-onnx``:

.. code-block:: Python

    python -m Examples.onnx.quantize \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --recipe "lpbq_seqmse" \
     --export-path "./onnx_lpbq" \
     --seqmse-num-batches 20


Evaluate
--------

Use the checkpoint generated in the previous step to evaluate the quantized model.

- ONNX evaluation works for models quantized with either ``aimet-torch`` or ``aimet-onnx``.
- PyTorch evaluation works only for models quantized with ``aimet-torch``.

Using ``aimet-onnx``:

.. code-block:: Python

    python -m Examples.onnx.evaluate \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --checkpoint "./torch_lpbq" \
     --eval-ppl

Now, we will go through the performance numbers for the selected LLMs.


Performance Summary
===================

Once the model is quantized, it is essential to evaluate its accuracy to ensure it meets acceptable thresholds. The same evaluation can also be performed on the original (unquantized) model to establish a strong baseline.

We demonstrate quantitative evaluation using two key metrics:

- `Perplexity (PPL) <https://en.wikipedia.org/wiki/Perplexity>`_ on WikiText (English)
- `MMLU <https://huggingface.co/datasets/cais/mmlu>`_

Additionally, we report:

- End-to-end runtime for each quantization recipe
- Peak CUDA memory usage during quantization

The consolidated performance tables summarize results for selected LLM models. You will find numbers for both recipes using ``aimet-torch`` and ``aimet-onnx``.

.. note::

    For models quantized using ``aimet-torch``, we include results from evaluation on ``aimet-onnx``. This ensures accurate activation quantizer placement and mixed-precision simulation (e.g., INT8 KV Cache).

    To avoid confusion, we explicitly report two fields for each result:

    - ``Quantized With`` – the AIMET package used to create the quantized model
    - ``Evaluated On`` – the AIMET package used to measure accuracy and performance

During quantization and evaluation, we use a sequence length of `2048 tokens` (referred to as AR-2048) and the context length of `4096 tokens`.


1. meta-llama/Llama-3.2-1B-Instruct
-----------------------------------
.. include:: models/llama-3.2-1b.rst
    :start-line: 2


2. meta-llama/Llama-3.2-3B-Instruct
-----------------------------------

.. include:: models/llama-3.2-3b.rst
    :start-line: 2


3. Qwen/Qwen2.5-0.5B-Instruct
-----------------------------

.. include:: models/qwen-2.5-0.5b.rst
    :start-line: 2


4. Qwen/Qwen2.5-1.5B-Instruct
-----------------------------

.. include:: models/qwen-2.5-1.5b.rst
    :start-line: 2


5. Qwen/Qwen3-4B
----------------

.. include:: models/qwen-3-4b.rst
    :start-line: 2


6. microsoft/Phi-3.5-mini-instruct
----------------------------------

.. include:: models/phi-3.5-mini.rst
    :start-line: 2


FAQs
====

#. When should I choose ``aimet-torch`` vs ``aimet-onnx``?
    - Choose ``aimet-torch`` when:
        - You want to apply quantization directly on a PyTorch model and keep the workflow within the PyTorch ecosystem.
        - You plan to apply :ref:`Quantization-Aware Training (QAT) <techniques-qat>` or run calibration using PyTorch datasets and dataloaders.
        - You need flexibility for dynamic graph operations.

    - Choose ``aimet-onnx`` when:
        - You need a static graph representation for deployment.
        - You want full quantization coverage, including functional operations that ``aimet-torch`` cannot instrument easily.
        - You are preparing the model for hardware adaptation (e.g. QAIRT) or other runtimes which consume ONNX graphs.

#. When should I choose Recipe 1 vs Recipe 2?
    - Choose Recipe 1: PCQ + SpinQuant + AdaScale
        - Uses Per-channel Quantization (PCQ), which provides good granularity for weights.
        - Performance KPIs (token rate, time-to-first-tokens etc.) are better on the target device.
        - Recommended when you can afford longer calibration time and prioritize throughput over accuracy.
    - Choose Recipe 2: LPBQ + SequentialMSE
        - Uses Blockwise quantization, which provides finer granularity than PCQ.
        - Recommended when the accuracy is the top priority.
        - Trade off: Slight impact on performance KPIs due to INT4 -> INT8 decoding.

#. Can I run the artifacts generated from the recipes as-is on target hardware?
    - No. The generated artifacts from the recipes are not directly compatible with QAIRT and require non-trivial adaptation steps for deployment on target hardware. Refer to the `model adaptation guide <https://github.com/quic/ai-hub-models/blob/main/tutorials/llm/onboarding.md>`_ for details.

#. Why does computing MMLU takes a long time?
    - MMLU evaluation can be slow even on high-end GPUs because it involves thousands of questions across 57 subjects. You can trade off accuracy for speed by reducing the number of samples.

#. Why INT8 KV Cache is not "Good enough" for Qwen 2.5?
    - Qwen 2.5 (0.5B and 1.5B) suffers with INT8 path (with only 256 discrete levels) for KV Cache activations due to wider dynamic range and INT16 offers 65,536 discrete levels which drastically reduces quantization error. So for Qwen 2.5, INT16, which doubles memory compared to INT8, maintains performance much closer to FP32, making it the better choice when quality matters and memory allows.


Contact Us
==========

Please reach out to us if you encounter any issue with this tutorial or applying recipes to similar models.

- `Slack Community <https://qualcomm-ai-hub.slack.com/archives/C08JKBE0UHY>`_
