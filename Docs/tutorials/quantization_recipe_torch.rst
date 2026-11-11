.. include:: ../abbreviation.txt

.. _quantization-genai-recipe-torch:


###########################################
Quantization recipes for LLMs (AIMET Torch)
###########################################


This page is a companion to the :ref:`Quantization recipes for LLMs <quantization-genai-recipe>` tutorial.
The primary tutorial reports results using ``aimet-onnx``. For completeness, this page reports the equivalent
results obtained by quantizing with ``aimet-torch``.

The same two reference recipes are applied (INT4 weights, INT16 activations):

#. PCQ + SpinQuant + AdaScale
#. LPBQ + SeqMSE

Refer to the :ref:`primary tutorial <quantization-genai-recipe>` for the recipe descriptions, system
requirements, workflow overview, and FAQs.

.. note::

    Due to PyTorch limitations, certain functional operations (``torch.nn.functional``) cannot have quantizers
    inserted, which makes a mixed-precision profile (e.g., INT8 KV Cache) challenging to apply directly. To
    address this, models quantized with ``aimet-torch`` are evaluated on ``aimet-onnx``, which provides a static
    graph that ensures correct quantizer insertion for all activations and a more accurate quantization
    simulation.


Accuracy Results
================

We report accuracy using two key metrics, alongside the cost of running each recipe:

- `Perplexity (PPL) <https://en.wikipedia.org/wiki/Perplexity>`_ on WikiText (English)
- `MMLU <https://huggingface.co/datasets/cais/mmlu>`_
- End-to-end runtime for each quantization recipe
- Peak CUDA memory usage during quantization

Models are quantized with ``aimet-torch`` and evaluated on ``aimet-onnx``, using a sequence length of
``2048`` tokens (AR-2048) and a context length of ``4096`` tokens.

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
          <td>FP32</td><td>FP32</td><td>—</td><td>12.14</td><td>46.06</td><td>&lt;1</td><td>6</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>SpinQuant<br>AdaScale (b=128, i=2048)</td>
          <td>13.67</td><td>42.25</td><td>151</td><td>21</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT8</td><td>SeqMSE</td>
          <td>14.07</td><td>43.09</td><td>45</td><td>29</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="3"><a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct">LLaMA 3.2 3B Instruct</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>10.13</td><td>60.74</td><td>&lt;1</td><td>14</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>SpinQuant<br>AdaScale (b=128, i=1024)</td>
          <td>11.01</td><td>58.09</td><td>395</td><td>41</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT8</td><td>SeqMSE</td>
          <td>10.69</td><td>59.08</td><td>162</td><td>51</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="3"><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B">Qwen 2.5 0.5B Instruct</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>13.14</td><td>46.30</td><td>&lt;1</td><td>4</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT16</td><td>SpinQuant<br>AdaScale (b=128, i=2048)</td>
          <td>13.89</td><td>44.19</td><td>200</td><td>13</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT16</td><td>SeqMSE</td>
          <td>15.32</td><td>42.33</td><td>23</td><td>14</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="3"><a href="https://huggingface.co/Qwen/Qwen2.5-1.5B">Qwen 2.5 1.5B Instruct</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>12.41</td><td>54.65</td><td>&lt;1</td><td>8</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT16</td><td>SpinQuant<br>AdaScale (b=128, i=1024)</td>
          <td>13.57</td><td>49.81</td><td>183</td><td>23</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT16</td><td>SeqMSE</td>
          <td>14.86</td><td>49.25</td><td>68</td><td>26</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="3"><a href="https://huggingface.co/Qwen/Qwen3-4B">Qwen 3 4B</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>12.41</td><td>70.06</td><td>&lt;1</td><td>17</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>SpinQuant<br>AdaScale (b=128, i=512)</td>
          <td>13.85</td><td>65.07</td><td>402</td><td>48</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT8</td><td>SeqMSE</td>
          <td>13.10</td><td>65.66</td><td>162</td><td>39</td>
        </tr>
      </tbody>
      <tbody>
        <tr>
          <td rowspan="3"><a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct">Phi 3.5 mini instruct</a></td>
          <td>FP32</td><td>FP32</td><td>—</td><td>5.77</td><td>68.89</td><td>&lt;1</td><td>16</td>
        </tr>
        <tr>
          <td>INT4 PCQ</td><td>INT16<br>KV=INT8</td><td>SpinQuant<br>AdaScale (b=32, i=256)</td>
          <td>6.58</td><td>62.62</td><td>257</td><td>48</td>
        </tr>
        <tr>
          <td>INT4 LPBQ</td><td>INT16<br>KV=INT8</td><td>SeqMSE</td>
          <td>6.45</td><td>64.63</td><td>124</td><td>38</td>
        </tr>
      </tbody>
    </table>


Quick Start
===========

Quantize the `Llama 3.2 1B` model using ``aimet-torch``.

Example: Apply Recipe 1 (pcq_spinquant_adascale)

.. code-block:: Python

    python -m Examples.torch.quantize \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --recipe "pcq_spinquant_adascale" \
     --export-path "./torch_pcq" \
     --adascale-num-batches 128 --adascale-num-iterations 2048

Example: Apply Recipe 2 (lpbq_seqmse)

.. code-block:: Python

    python -m Examples.torch.quantize \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --recipe "lpbq_seqmse" \
     --export-path "./torch_lpbq" \
     --seqmse-num-batches 20

Evaluate the checkpoint on ``aimet-onnx``.

.. code-block:: Python

    python -m Examples.onnx.evaluate \
     --model-id "meta-llama/Llama-3.2-1B-Instruct" \
     --checkpoint "./torch_lpbq" \
     --eval-ppl


Contact Us
==========

Please reach out to us if you encounter any issue with this tutorial or applying recipes to similar models.

- `Slack Community <https://qualcomm-ai-hub.slack.com/archives/C08JKBE0UHY>`_
