
![Qualcomm Innovation Center, Inc.](Docs/images/logo-quic-on@h68.png)

[![AIMET on GitHub Pages](Docs/images/button-overview.png)](https://quic.github.io/aimet-pages/index.html)
[![Documentation](Docs/images/button-docs.png)](https://quic.github.io/aimet-pages/releases/latest/index.html)
[![Install instructions](Docs/images/button-install.png)](#quick-installation)
[![Discussion Forums](Docs/images/button-forums.png)](https://github.com/quic/aimet/discussions)
[![What's New](Docs/images/button-whats-new.png)](#whats-new)

# AI Model Efficiency Toolkit (AIMET)

<a href="https://quic.github.io/aimet-pages/index.html">AIMET</a> is a library that provides advanced model quantization and compression techniques for trained neural network models. It provides features that have been proven to improve run-time performance of deep learning neural network models with lower compute and memory requirements and minimal impact to task accuracy.

![How AIMET works](Docs/images/how-it-works.png)

AIMET is designed to work with [PyTorch](https://pytorch.org), [TensorFlow](https://tensorflow.org) and [ONNX](https://onnx.ai) models.

We also host the [AIMET Model Zoo](https://github.com/quic/aimet-model-zoo) - a collection of popular neural network models optimized for 8-bit inference. We also provide recipes for users to quantize floating point models using AIMET.

## Table of Contents
- [Installation](#quick-installation)
- [Why AIMET?](#why-aimet)
- [Supported features](#supported-features)
- [What's New](#whats-new)  
- [Results](#results)
- [Resources](#resources)
- [Contributions](#contributions)
- [Team](#team)
- [License](#license)

## Installation

To install the latest version of AIMET for supported frameworks and compute platforms, see [Install and run AIMET](https://quic.github.io/aimet-pages/releases/latest/install)

### Building from source

To build the latest AIMET code from the source, see [Build, install and run AIMET from source in *Docker* environment](./packaging/docker_install.md)


## Why AIMET?

![Benefits of AIMET](Docs/images/AImodelEfficency.png)

* **Supports advanced quantization techniques**: Inference using integer runtimes is significantly faster than using floating-point runtimes. For example, models run 5x-15x faster on the Qualcomm Hexagon DSP than on the Qualcomm Kyro CPU. In addition, 8-bit precision models have a 4x smaller footprint than 32-bit precision models. However, maintaining model accuracy when quantizing ML models is often challenging. AIMET solves this using novel techniques like Data-Free Quantization that provide state-of-the-art INT8 results on several popular models.
* **Supports advanced model compression techniques** that enable models to run faster at inference-time and require less memory
* **AIMET is designed to automate optimization** of neural networks avoiding time-consuming and tedious manual tweaking. AIMET also provides user-friendly APIs that allow users to make calls directly from their [TensorFlow](https://tensorflow.org) or [PyTorch](https://pytorch.org) pipelines.

Please visit the [AIMET on Github Pages](https://quic.github.io/aimet-pages/index.html) for more details.

## Supported Features

### Quantization

* *Cross-Layer Equalization*: Equalize weight tensors to reduce amplitude variation across channels
* *Bias Correction*: Corrects shift in layer outputs introduced due to quantization
* *Adaptive Rounding*: Learn the optimal rounding given unlabelled data
* *Quantization Simulation*: Simulate on-target quantized inference accuracy
* *Quantization-aware Training*: Use quantization simulation to train the model further to improve accuracy

### Model Compression

* *Spatial SVD*: Tensor decomposition technique to split a large layer into two smaller ones
* *Channel Pruning*: Removes redundant input channels from a layer and reconstructs layer weights
* *Per-layer compression-ratio selection*: Automatically selects how much to compress each layer in the model

### Visualization

* *Weight ranges*: Inspect visually if a model is a candidate for applying the Cross Layer Equalization technique. And the effect after applying the technique
* *Per-layer compression sensitivity*: Visually get feedback about the sensitivity of any given layer in the model to compression

## What's New
Some recently added features include
* Adaptive Rounding (AdaRound): Learn the optimal rounding given unlabelled data
* Quantization-aware Training (QAT) for recurrent models (including with RNNs, LSTMs and GRUs)

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
* [Documentation Main Page](https://quic.github.io/aimet-pages/releases/latest/index.html)
* [API Reference](https://quic.github.io/aimet-pages/releases/latest/apiref/index.html)
* [Discussion Forums](https://github.com/quic/aimet/discussions)
* [Tutorial Videos](https://quic.github.io/aimet-pages/index.html#video)
* [Example Code](Examples/README.md)

## Contributions
Thanks for your interest in contributing to AIMET! Please read our [Contributions Page](CONTRIBUTING.md) for more information on contributing features or bug fixes. We look forward to your participation!

## Team
AIMET aims to be a community-driven project maintained by Qualcomm Innovation Center, Inc.

## License
AIMET is licensed under the BSD 3-clause "New" or "Revised" License. Check out the [LICENSE](LICENSE) for more details.
