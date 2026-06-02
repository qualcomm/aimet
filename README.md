
![Qualcomm Innovation Center, Inc.](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/aimet/docs/assets/images/logo-quic-on@h68.png)

[<img src="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/aimet/docs/assets/images/readme/button-overview.png" width="90" height="40">](https://qualcomm.github.io/aimet-pages/releases/latest/index.html)
[<img src="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/aimet/docs/assets/images/readme/button-docs.png" width="90" height="40">](https://qualcomm.github.io/aimet-pages/releases/latest/index.html)
[<img src="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/aimet/docs/assets/images/readme/button-install.png" width="90" height="40">](https://qualcomm.github.io/aimet-pages/releases/latest/overview/install/quick-start.html)
[<img src="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/aimet/docs/assets/images/readme/button-forum.png" width="90" height="40">](https://github.com/quic/aimet/discussions)
[<img src="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/aimet/docs/assets/images/readme/button-slack.png" width="90" height="40">](https://qualcomm-ai-hub.slack.com/archives/C08JKBE0UHY)
[<img src="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/aimet/docs/assets/images/readme/button-new.png" width="90" height="40">](https://qualcomm.github.io/aimet-pages/releases/latest/release_notes.html)

# AI Model Efficiency Toolkit (AIMET)

<a href="https://qualcomm.github.io/aimet-pages/index.html">AIMET</a> is a software toolkit for quantizing trained ML models.

AIMET improves the runtime performance of deep learning models by reducing compute load and memory footprint.
Models quantized with AIMET facilitate its deployment on edge devices like mobile phones or laptops by reducing memory footprint.

AIMET employs post-training and fine-tuning techniques to minimize accuracy loss during quantization and compression.
AIMET supports models from the ONNX and PyTorch frameworks.

![How AIMET works](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/aimet/docs/assets/images/how-it-works.png)

AIMET is designed to work with [PyTorch](https://pytorch.org) and [ONNX](https://onnx.ai) models.

You can find models quantized with AIMET on [Qualcomm AI Hub Models](https://github.com/quic/ai-hub-models) - a collection of optimized and quantized models.

## Why AIMET?

![Benefits of AIMET](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/aimet/docs/assets/images/AImodelEfficency.png)

* **Advanced quantization techniques**: Inference using integer runtimes is significantly faster than using floating-point runtimes. For example, models run 5x-15x faster on the Qualcomm Hexagon DSP than on the Qualcomm Kyro CPU. In addition, 8-bit precision models have a 4x smaller footprint than 32-bit precision models. However, maintaining model accuracy when quantizing ML models is often challenging. AIMET solves this using novel techniques like Data-Free Quantization that provide state-of-the-art INT8 results on several popular models.
* **Supports advanced model compression techniques** that enable models to run faster at inference-time and require less memory
* **AIMET is designed to automate optimization** of neural networks avoiding time-consuming and tedious manual tweaking. AIMET also provides user-friendly APIs that allow users to make calls directly from their [PyTorch](https://pytorch.org) pipelines.

Please visit the [AIMET on Github Pages](https://qualcomm.github.io/aimet-pages/index.html) for more details.

## Quick Start

[aimet-onnx](https://pypi.org/project/aimet-onnx/) and [aimet-torch](https://pypi.org/project/aimet-torch/) is available on PyPI.

Check our [Quick Start](https://qualcomm.github.io/aimet-pages/releases/latest/overview/install/quick-start.html) to get started with latest AIMET package.

### Build from source

To build the latest AIMET code from the source, see [Build, install and run AIMET from source in *Docker* environment](https://qualcomm.github.io/aimet-pages/releases/latest/overview/install/build_from_source.html#build-from-source)


## Supported Features

### Post-Training Quantization(PTQ)

[Check out guide](https://qualcomm.github.io/aimet-pages/releases/latest/techniques/ptq.html) to get started on PTQ technique.

Following table summarizes basic technique such as `Calibration` to advanced techniques such as `SeqMSE` and `Adaptive Rounding(AdaRound)` that you can use with AIMET.

| Technique | ONNX | PyTorch | What does it do? |
| -- | -- | -- | -- |
| [Calibration](https://qualcomm.github.io/aimet-pages/releases/latest/techniques/ptq.html) | ✅ | ✅ | Computes Quantization parameters |
| [AdaRound](https://qualcomm.github.io/aimet-pages/releases/latest/ptq_techniques/adaround.html) | ✅ | ✅ | Rounds quantized weights |
| [SeqMSE](https://qualcomm.github.io/aimet-pages/releases/latest/ptq_techniques/seq_mse.html) | ✅ | ✅ | Optimizes encodings for each layer |
| [BatchNorm Folding](https://qualcomm.github.io/aimet-pages/releases/latest/ptq_techniques/bnf.html) | ✅ | ✅ | Folds batchnorm to bridge the gap between simulation and on-target |
| [Cross Layer Equalization](https://qualcomm.github.io/aimet-pages/releases/latest/ptq_techniques/cle.html) | ✅ | ✅ | Rescales the weight to reduce range imbalance |
| [BatchNorm re-estimation](https://qualcomm.github.io/aimet-pages/releases/latest/ptq_techniques/bn.html) | ✅ | ✅ | Re-estimates batchnorm statistics |
| [AdaScale](https://qualcomm.github.io/aimet-pages/releases/latest/ptq_techniques/adascale.html) | ✅ | ✅ | Learns per-weight scales via blockwise optimization to minimize quantization error |
| [OmniQuant](https://qualcomm.github.io/aimet-pages/releases/latest/ptq_techniques/omniquant.html) | ❌ | ✅ | Optimizes quantized weights |
| [SpinQuant](https://qualcomm.github.io/aimet-pages/releases/latest/ptq_techniques/spinquant.html) | ✅ | ✅ | Reduces activation outliers via Hadamard rotations |

### Quantization Aware Training(QAT)

AIMET supports Quantization Aware Training(QAT) via [aimet-torch](https://pypi.org/project/aimet-torch/).

If you want to use both QAT and some of the advanced [PTQ techniques from AIMET](#post-training-quantizationptq), we recommend the following workflow:

![QAT workflow](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/aimet/docs/assets/images/workflow/qat.png)

Check detailed [QAT guide here](https://qualcomm.github.io/aimet-pages/releases/latest/techniques/qat.html)

### Model Compression

* *Spatial SVD*: Tensor decomposition technique to split a large layer into two smaller ones
* *Channel Pruning*: Removes redundant input channels from a layer and reconstructs layer weights
* *Per-layer compression-ratio selection*: Automatically selects how much to compress each layer in the model

### Visualization

* *Weight ranges*: Inspect visually if a model is a candidate for applying the Cross Layer Equalization technique. And the effect after applying the technique
* *Per-layer compression sensitivity*: Visually get feedback about the sensitivity of any given layer in the model to compression

## Results
AIMET can quantize an existing 32-bit floating-point model to an 8-bit fixed-point model without sacrificing much accuracy and without model fine-tuning.

<h4>DFQ</h4>

The DFQ method applied to several popular networks, such as MobileNet-v2 and ResNet-50, result in less than 0.9% loss in accuracy all the way down to 8-bit quantization, in an automated way without any training data.

<table style="width:50%">
  <tr>
    <th style="width:80px">Models</th>
    <th>FP32</th>
    <th>INT8 Simulation </th>
  </tr>
  <tr>
    <td>MobileNet v2 (top1)</td>
    <td align="center">71.72%</td>
    <td align="center">71.08%</td>
  </tr>
  <tr>
    <td>ResNet 50 (top1)</td>
    <td align="center">76.05%</td>
    <td align="center">75.45%</td>
  </tr>
  <tr>
    <td>DeepLab v3 (mIOU)</td>
    <td align="center">72.65%</td>
    <td align="center">71.91%</td>
  </tr>
</table>
<br>

<h4>AdaRound (Adaptive Rounding)</h4>
<h5>ADAS Object Detect</h5>
<p>For this example ADAS object detection model, which was challenging to quantize to 8-bit precision, AdaRound can recover the accuracy to within 1% of the FP32 accuracy.</p>
<table style="width:50%">
  <tr>
    <th style="width:80px" colspan="15">Configuration</th>
    <th>mAP - Mean Average Precision</th>
  </tr>
  <tr>
    <td colspan="15">FP32</td>
    <td align="center">82.20%</td>
  </tr>
  <tr>
    <td colspan="15">Nearest Rounding (INT8 weights, INT8 acts)</td>
    <td align="center">49.85%</td>
  </tr>
  <tr>
    <td colspan="15">AdaRound (INT8 weights, INT8 acts)</td>
    <td align="center" bgcolor="#add8e6">81.21%</td>
  </tr>
</table>

<h5>DeepLabv3 Semantic Segmentation</h5>
<p>For some models like the DeepLabv3 semantic segmentation model, AdaRound can even quantize the model weights to 4-bit precision without a significant drop in accuracy.</p>
<table style="width:50%">
  <tr>
    <th style="width:80px" colspan="15">Configuration</th>
    <th>mIOU - Mean intersection over union</th>
  </tr>
  <tr>
    <td colspan="15">FP32</td>
    <td align="center">72.94%</td>
  </tr>
  <tr>
    <td colspan="15">Nearest Rounding (INT4 weights, INT8 acts)</td>
    <td align="center">6.09%</td>
  </tr>
  <tr>
    <td colspan="15">AdaRound (INT4 weights, INT8 acts)</td>
    <td align="center" bgcolor="#add8e6">70.86%</td>
  </tr>
</table>
<br>

<h4>Quantization for Recurrent Models</h4>
<p>AIMET supports quantization simulation and quantization-aware training (QAT) for recurrent models (RNN, LSTM, GRU). Using QAT feature in AIMET, a DeepSpeech2 model with bi-directional LSTMs can be quantized to 8-bit precision with minimal drop in accuracy.</p>

<table style="width:50%">
  <tr>
    <th>DeepSpeech2 <br>(using bi-directional LSTMs)</th>
    <th>Word Error Rate</th>
  </tr>
  <tr>
    <td>FP32</td>
    <td align="center">9.92%</td>
  </tr>
  <tr>
    <td>INT8</td>
    <td align="center">10.22%</td>
  </tr>
</table>

<br>

<h4>Model Compression</h4>
<p>AIMET can also significantly compress models. For popular models, such as Resnet-50 and Resnet-18, compression with spatial SVD plus channel pruning achieves 50% MAC (multiply-accumulate) reduction while retaining accuracy within approx. 1% of the original uncompressed model.</p>

<table style="width:50%">
  <tr>
    <th>Models</th>
    <th>Uncompressed model</th>
    <th>50% Compressed model</th>
  </tr>
  <tr>
    <td>ResNet18 (top1)</td>
    <td align="center">69.76%</td>
    <td align="center">68.56%</td>
  </tr>
  <tr>
    <td>ResNet 50 (top1)</td>
    <td align="center">76.05%</td>
    <td align="center">75.75%</td>
  </tr>
</table>

<br>

## Resources
* [Documentation Main Page](https://qualcomm.github.io/aimet-pages/releases/latest/index.html)
* [API Reference](https://qualcomm.github.io/aimet-pages/releases/latest/apiref/index.html)
* [Discussion Forums](https://github.com/quic/aimet/discussions)
* [Slack](https://qualcomm-ai-hub.slack.com/archives/C08JKBE0UHY)
* [Tutorial Videos](https://qualcomm.github.io/aimet-pages/index.html#video)
* [Example Code](Examples/README.md)

## Contributions
Thanks for your interest in contributing to AIMET! Please read our [Contributions Page](CONTRIBUTING.md) for more information on contributing features or bug fixes. We look forward to your participation!

## Team
AIMET aims to be a community-driven project maintained by Qualcomm Innovation Center, Inc.

## License
AIMET is licensed under the BSD 3-clause "New" or "Revised" License. Check out the [LICENSE](LICENSE) for more details.
