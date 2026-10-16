.. include:: ../abbreviation.txt

.. _quantization-genai-recipe:


#############################
Quantization recipes for LLMs
#############################


This page shows the recipe used to quantize a set of popular large language models (LLMs) to INT4 weights and INT16 
activations using ``aimet-onnx``. The accuracy results (PPL, MMLU) are shown, followed by the steps that you can use 
to reproduce these results.

Note that the intent here is to show how to quantize the model. To create artifacts that can be directly deployed on 
the target hardware, additional adaptation steps are required. Refer to the 
`model adaptation guide <https://github.com/qualcomm/ai-hub-models/blob/main/tutorials/llm/onboarding.md>`_ for more details.


Accuracy Results
================

We report accuracy using two key metrics, alongside the cost of running each recipe:

- `Perplexity (PPL) <https://en.wikipedia.org/wiki/Perplexity>`_ on WikiText (English)
- `MMLU <https://huggingface.co/datasets/cais/mmlu>`_
- End-to-end runtime for each quantization recipe
- Peak CUDA memory usage during quantization

The FP32 row for each model is the unquantized baseline. The following settings are common
across all models:

- Embedding: INT16
- LM Head weights: INT8
- Calibration: ``num_batches=20``
- SeqMSE: ``num_batches=20``

The KV Cache precision varies per model and is shown on the second line of the Acts column.

AdaScale ``num_batches`` and ``num_iterations`` vary per model and are noted inline in the Recipe column as
``AdaScale (b=<num_batches>, i=<num_iterations>)``.

.. raw:: html

    <table class="perf-table">
      <colgroup>
        <col style="width:20%"><col style="width:8%"><col style="width:11%"><col style="width:40%">
        <col style="width:5%"><col style="width:5%"><col style="width:7%"><col style="width:4%">
      </colgroup>
      <thead>
        <tr>
          <th>Model</th><th>Weights</th><th>Acts</th><th>Recipe</th>
          <th>PPL</th><th>MMLU</th><th>Time (min)</th><th>CUDA (GB)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td rowspan="3"><a href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct">LLaMA 3.2 1B Instruct</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>12.14</td><td>46.06</td><td>&lt;1</td><td>7</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>SpinQuant (r1)<br>AdaScale (b=128, i=2048)</td>
          <td>13.68</td><td>41.82</td><td>113</td><td>22</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT8</td><td>SeqMSE</td>
          <td>13.84</td><td>43.53</td><td>21</td><td>29</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="3"><a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct">LLaMA 3.2 3B Instruct</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>10.13</td><td>60.74</td><td>&lt;1</td><td>14</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>SpinQuant (r1)<br>AdaScale (b=128, i=1024)</td>
          <td>11.14</td><td>57.28</td><td>290</td><td>40</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT8</td><td>SeqMSE</td>
          <td>10.55</td><td>59.29</td><td>73</td><td>47</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="3"><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B">Qwen 2.5 0.5B Instruct</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>13.14</td><td>46.30</td><td>&lt;1</td><td>4</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT16</td><td>SpinQuant (r1)<br>AdaScale (b=128, i=2048)</td>
          <td>13.82</td><td>42.65</td><td>77</td><td>15</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT16</td><td>SeqMSE</td>
          <td>15.30</td><td>43.26</td><td>12</td><td>18</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="3"><a href="https://huggingface.co/Qwen/Qwen2.5-1.5B">Qwen 2.5 1.5B Instruct</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>12.41</td><td>54.65</td><td>&lt;1</td><td>8</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT16</td><td>SpinQuant (r1)<br>AdaScale (b=128, i=1024)</td>
          <td>13.35</td><td>50.27</td><td>134</td><td>23</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT16</td><td>SeqMSE</td>
          <td>14.33</td><td>49.97</td><td>38</td><td>30</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-0.6B">Qwen 3 0.6B</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>19.15</td><td>47.11</td><td>&lt;1</td><td>8</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>SpinQuant (r2 + r3)<br>AdaScale (b=128, i=2048)</td>
          <td>20.67</td><td>43.41</td><td>65</td><td>18</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-1.7B">Qwen 3 1.7B</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>15.63</td><td>59.96</td><td>1</td><td>18</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>AdaScale (b=128, i=2048)</td>
          <td>16.59</td><td>56.65</td><td>105</td><td>25</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-4B">Qwen 3 4B</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>12.41</td><td>70.06</td><td>&lt;1</td><td>17</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>SpinQuant (r1)<br>AdaScale (b=128, i=512)</td>
          <td>13.79</td><td>65.07</td><td>274</td><td>48</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT8</td><td>SeqMSE</td>
          <td>12.77</td><td>65.36</td><td>95</td><td>54</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen 3 8B</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>8.99</td><td>74.96</td><td>2</td><td>34</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>AdaScale (b=128, i=2048)</td>
          <td>9.49</td><td>69.89</td><td>403</td><td>74</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3.5-4B">Qwen 3.5 4B</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>8.88</td><td>72.03</td><td>1</td><td>22</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>AdaScale (b=128, i=2048)</td>
          <td>10.09</td><td>72.67</td><td>221</td><td>52</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="3"><a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct">Phi 3.5 mini Instruct</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>5.77</td><td>68.89</td><td>&lt;1</td><td>17</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>SpinQuant (r1)<br>AdaScale (b=32, i=256)</td>
          <td>6.50</td><td>62.51</td><td>112</td><td>49</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT8</td><td>SeqMSE</td>
          <td>6.41</td><td>63.90</td><td>93</td><td>64</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="2"><a href="https://huggingface.co/google/gemma-3-4b-it">Gemma 3 4B Instruct</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>13.41</td><td>58.41</td><td>&lt;1</td><td>20</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT8</td><td>SpinQuant (r2)<br>SeqMSE</td>
          <td>19.61</td><td>52.79</td><td>105</td><td>68</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="2"><a href="https://huggingface.co/google/gemma-3-1b-it">Gemma 3 1B Instruct</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>23.11</td><td>39.92</td><td>&lt;1</td><td>8</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT8</td><td>SpinQuant (r2)<br>SeqMSE</td>
          <td>26.70</td><td>37.28</td><td>29</td><td>27</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="2"><a href="https://huggingface.co/google/gemma-3-270m-it">Gemma 3 270M Instruct</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>54.88</td><td>26.31</td><td>&lt;1</td><td>4</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>SpinQuant (r2)<br>AdaScale (b=128, i=2048)</td>
          <td>64.79</td><td>26.83</td><td>26</td><td>81</td>
        </tr>
      </tbody>
    </table>


System Requirements
===================

The quantization process requires a machine with:

- Operating System: Linux
- Hardware: CUDA-enabled GPU

GPU Memory Requirements:

- Minimum: 40GB VRAM


Recipes
=======
=======

We present two recipes for INT4 weights, INT16 activations quantization using combinations of :ref:`Post-Training Quantization (PTQ) <featureguide-index>` techniques available in ``aimet-onnx``:

#. PCQ + SpinQuant + AdaScale
    - :ref:`Per-Channel Quantization (PCQ) <techniques-ptq>` - Uses per-output channel scales for weights on linear layers.
    - :ref:`SpinQuant <ptq-spinquant>` - A PTQ technique that improves the accuracy by inserting rotations at specific points in the model to mitigate activation outliers.
    - :ref:`AdaScale <ptq-adascale>` - A PTQ technique that enhances accuracy by introducing learnable parameters in the weight quantizers and performing Block-wise Knowledge Distillation (BKD) against FP outputs.

#. LPBQ + SeqMSE
    - :ref:`Low Power Blockwise Quantization (LPBQ) <techniques-lpbq>`  - Applies blockwise quantization (``block_size=64``) for weights on linear layers.
    - :ref:`SeqMSE <ptq-seq-mse>` - Calibrates layer-by-layer to minimize Mean Square Error (MSE) between quantized and FP outputs.

To maintain accuracy, activations are primarily kept at INT16, with a mixed-precision profile using INT8 activations selectively where feasible—such as for the KV cache.


Workflow Overview
=================

#. Load the HuggingFace model
    - Start by loading the pretrained model using HuggingFace ``transformers`` library.
#. Apply the selected Quantization recipe
    - Use ``aimet-onnx`` for `ONNX <https://onnx.ai/>`_ based workflows.
#. Compute Activations encodings
    - ``aimet-onnx`` computes activation encodings using representative data. In this tutorial, we use `WikiText (English) <https://github.com/qualcomm/aimet/blob/develop/GenAILab/shared/helpers/datasets.py>`_ for calibration.
    - ``aimet-onnx`` provides a static graph, ensuring correct quantizer insertion for all activations (including a mixed-precision profile such as INT8 KV Cache) and delivering an accurate quantization simulation.
#. Export for deployment
    -   Export the ONNX model along with the encodings file for the on-target inference.


Quick Start
===========

This section provides a quick example of applying a quantization recipe using ``aimet-onnx``.

In this tutorial, we apply the quantization recipe to the `Llama 3.2 1B` model. The steps work for all fine-tuned variants that share the same tokenizer and network architecture.

The example scripts are designed to be `flattened`, so all AIMET API calls and HuggingFace API calls are visible at the top level.

To understand how this works under the hood using the same driver code, refer to the `Generator <https://github.com/qualcomm/aimet/tree/develop/GenAILab#how-it-all-works>`_ class in ``GenAILab``.

Quantize
--------

Example: Apply Recipe 1 (pcq_spinquant_adascale)

.. code-block:: Python

    python -m Examples.onnx.quantize \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --recipe "pcq_spinquant_adascale" \
     --export-path "./onnx_pcq" \
     --adascale-num-batches 128 --adascale-num-iterations 2048

Example: Apply Recipe 2 (lpbq_seqmse)

.. code-block:: Python

    python -m Examples.onnx.quantize \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --recipe "lpbq_seqmse" \
     --export-path "./onnx_lpbq" \
     --seqmse-num-batches 20


Evaluate
--------

Use the checkpoint generated in the previous step to evaluate the quantized model.

.. code-block:: Python

    python -m Examples.onnx.evaluate \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --checkpoint "./onnx_lpbq" \
     --eval-ppl


FAQs
====

#. Why does this tutorial use ``aimet-onnx``?
    - ``aimet-onnx`` provides a static graph representation, giving full quantization coverage (including functional operations that ``aimet-torch`` cannot instrument easily) and an accurate mixed-precision simulation (e.g. INT8 KV Cache).
    - It is also the natural starting point for hardware adaptation (e.g. QAIRT) and other runtimes which consume ONNX graphs.
    - You may still prefer ``aimet-torch`` when you want to keep the workflow within the PyTorch ecosystem, apply :ref:`Quantization-Aware Training (QAT) <techniques-qat>`, or calibrate using PyTorch datasets and dataloaders. Equivalent ``aimet-torch`` results are reported on the :ref:`AIMET Torch recipes page <quantization-genai-recipe-torch>`.

#. When should I choose Recipe 1 vs Recipe 2?
    - Choose Recipe 1: PCQ + SpinQuant + AdaScale
        - Uses Per-channel Quantization (PCQ), which provides good granularity for weights.
        - Performance KPIs (token rate, time-to-first-tokens etc.) are better on the target device.
        - Recommended when you can afford longer calibration time and prioritize throughput over accuracy.
    - Choose Recipe 2: LPBQ + SeqMSE
        - Uses Blockwise quantization, which provides finer granularity than PCQ.
        - Recommended when the accuracy is the top priority.
        - Trade off: Slight impact on performance KPIs due to INT4 -> INT8 decoding.

#. Can I run the artifacts generated from the recipes as-is on target hardware?
    - No. The generated artifacts from the recipes are not directly compatible with QAIRT and require non-trivial adaptation steps for deployment on target hardware. Refer to the `model adaptation guide <https://github.com/qualcomm/ai-hub-models/blob/main/tutorials/llm/onboarding.md>`_ for details.

#. Why does computing MMLU takes a long time?
    - MMLU evaluation can be slow even on high-end GPUs because it involves thousands of questions across 57 subjects. You can trade off accuracy for speed by reducing the number of samples.

#. Why INT8 KV Cache is not "Good enough" for Qwen 2.5?
    - Qwen 2.5 (0.5B and 1.5B) suffers with INT8 path (with only 256 discrete levels) for KV Cache activations due to wider dynamic range and INT16 offers 65,536 discrete levels which drastically reduces quantization error. So for Qwen 2.5, INT16, which doubles memory compared to INT8, maintains performance much closer to FP32, making it the better choice when quality matters and memory allows.


AIMET Torch Recipes
===================

This page reports results using ``aimet-onnx``. For completeness, equivalent results obtained by quantizing with ``aimet-torch`` (and evaluating on ``aimet-onnx``) are reported on a separate page:

- :ref:`Quantization recipes for LLMs (AIMET Torch) <quantization-genai-recipe-torch>`


Contact Us
==========

Please reach out to us if you encounter any issue with this tutorial or applying recipes to similar models.

- `Slack Community <https://qualcomm-ai-hub.slack.com/archives/C08JKBE0UHY>`_
