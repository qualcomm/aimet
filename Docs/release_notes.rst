.. include:: abbreviation.txt

.. _rn-index:

#############
Release notes
#############

2.11.0
======
* New Feature
    * PyTorch
        * SpinQuant (experimental) - implement SpinQuant PTQ technique (https://arxiv.org/pdf/2308.13137) for Llama, Qwen2, and Mistral families (R1 rotation w/o optimization) (`7364b37`_)
        * Enable Adascale and Omniquant for Mistral (`d33e98c`_)

    * ONNX 
        * Enable llm_configurator for Llama (Experimental) (`08c17b8`_)
    
* Bug fixes and Improvements
    * Common    
        * Represent LPBQ as DequantizeLinear in onnx QDQ (`a967b8f`_)
        * Add additional sanity checks in LPBQ export logic (`45c2a65`_)
        * Allow negative block axis in LPBQ QDQ export (`6f670a4`_)
        * Add support for enabling param bw=2 in QuantSim (`2d4e0eb`_)
        * Fix tanh output encoding range to [-1, 1] (`3c92bb7`_)
        
    * ONNX 
        * Apply matmul exception rule only for integer quantization (`bb93c76`_)
        * Optimize blockwise min-max encoding analyzer (`4febdd4`_)
        * Remove explicit FP32 model creation inside AdaRound and optimize building sessions during the optimization process (`b1415bd`_)
        * Make Concat output quantizer inherit fixed input range (`50f35dd`_)
        * Enable output quantizers to inherit input encoding when tying encodings (`3750526`_)
        * Fix bug in CLE with bn_conv groups (`654f4b1`_)

    * PyTorch 
        * Guarantee positive scale during aimet-torch QAT (`2ed8305`_)
        * Add secondary progress bars to Adascale and Omniquant (`6c92a97`_)
    
* Documentation Updates
    * Update Quick Start example and PTQ section (`6c9f584`_)
    * Add missing workflow images (`f961ed4`_)

* Known Issues
    * Keras
        * Accuracy drop observed with AIMET Keras for certain models. Fix is planned for the next release.

.. _6c92a97: https://github.com/quic/aimet/commit/6c92a9760fdb0fd1f095acd58935564eab18e69f
.. _6c9f584: https://github.com/quic/aimet/commit/6c9f5848edbbe8bc1a3d87bed2ed0072abda0e9b
.. _f961ed4: https://github.com/quic/aimet/commit/f961ed40f3f0f1c05315b901add3275751aa3afe
.. _2ed8305: https://github.com/quic/aimet/commit/2ed8305190856a81881a590d5f7390e02531d912
.. _a967b8f: https://github.com/quic/aimet/commit/a967b8f0d71abe5d24c0a381abcdda3622982d15
.. _3c92bb7: https://github.com/quic/aimet/commit/3c92bb72683fb6a5ed89142dbeacf9bea901bf67
.. _d33e98c: https://github.com/quic/aimet/commit/d33e98c427f4cdcb19bc6443dec772590d1011a5
.. _08c17b8: https://github.com/quic/aimet/commit/08c17b875cbe6fce0a5d6f2ba75a7ddea508ad0f
.. _2d4e0eb: https://github.com/quic/aimet/commit/2d4e0eb7b235b1ff7c420362037f0292b183dfe1
.. _b1415bd: https://github.com/quic/aimet/commit/b1415bded1d7ba539d7a1f35b04adf7a7ebf17be
.. _45c2a65: https://github.com/quic/aimet/commit/45c2a65e254ee674bfc4c00f4bb5fbe830aa4922
.. _6f670a4: https://github.com/quic/aimet/commit/6f670a41d75fbe4664a24c3d899ab37faac7fbfc
.. _bb93c76: https://github.com/quic/aimet/commit/bb93c765bdcc2f06a4d9fd1a07833bb54e2627a9
.. _50f35dd: https://github.com/quic/aimet/commit/50f35dd933744a2096de22b679e6e4a08ed29cb4
.. _3750526: https://github.com/quic/aimet/commit/3750526bb6c6e339c16773cc1bdc752fffcb9802
.. _654f4b1: https://github.com/quic/aimet/commit/654f4b181bc4825c6122f5191d29cc218996caac
.. _4febdd4: https://github.com/quic/aimet/commit/4febdd4f72a1414c90b37704db220321b8a43d77
.. _7364b37: https://github.com/quic/aimet/commit/7364b37c9ab5cb0f90f02209634c5fc412cce8d8


2.10.0
======

* New Feature
    * Promote to_onnx_qdq to a public API (`f333188`_). Note: This is currently a beta feature

* Bug fixes and Improvements
    * Common
        * Added hover tooltip to plot per layer sensitivity. Changed x-axis to plot layer indices instead of names (`c96894f`_)
    * PyTorch
        * Implement scaling factor in aimet-torch float QDQ (`9b8c655`_)
        * Fix CustomSiLU bug (`499df9f`_)
        * Added extra logic to isolate model outputs from connectedgraph (`4ad0703`_)
        * Always instantiate quantizers with requires_grad=True (`5aac9c5`_)
    * ONNX
        * Allow AdaRound and SeqMSE to take uncalibrated sims(`31ca7fd`_)
        * Modify bias quantizer setting based on weight quantizer (`b47a97e`_)
        * Fix cnt overflow issue (`70029c5`_)
        * Make memory saving optimization default in build_session and _infer_activation_dtypes (`4b94ca9`_)

* Documentation
    * Update SeqMSE feature guide (`fefd504`_)
    * Fix links in example notebooks (`fe66376`_)
    * Modify docs for CLE (`f9d0d6c`_)
    * Edit automatic mixed precision feature guide (`22b5c94`_)
    * Polish BQ user guide (`f547a49`_)
    * Polish QAT user guide (`339a225`_)

.. _c96894f: https://github.com/quic/aimet/commit/c96894f3795e1b0986ba0c2b6f0b04464d003d0f
.. _9b8c655: https://github.com/quic/aimet/commit/9b8c655a6a17cc4339f494f17e063f36aa679383
.. _499df9f: https://github.com/quic/aimet/commit/499df9f24054c291160272d2a4155ad82919d8b7
.. _4ad0703: https://github.com/quic/aimet/commit/4ad0703ba3e6e6dd688831eb6f297f3c735a4e8b
.. _5aac9c5: https://github.com/quic/aimet/commit/5aac9c503961aa832ae1350d3fdbc81fd2c10ff0
.. _31ca7fd: https://github.com/quic/aimet/commit/31ca7fdead574bd8614720bba5a7cae2739c7841
.. _b47a97e: https://github.com/quic/aimet/commit/b47a97eef0b89ea1becea3b4cbca0de018cc113c
.. _f333188: https://github.com/quic/aimet/commit/f3331884a2e7da0dc22770fd1ae792564f0fa094
.. _70029c5: https://github.com/quic/aimet/commit/70029c596cff1d188fcfbc308cc06f99bdff1fdf
.. _4b94ca9: https://github.com/quic/aimet/commit/4b94ca9267cb9513f996fedc350b583e6f28ce30
.. _fefd504: https://github.com/quic/aimet/commit/fefd504c79de738a99b82d051e7b70ffcb195a3e
.. _fe66376: https://github.com/quic/aimet/commit/fe66376f5704b9fa4dc494dd8d22f8a2689fc0c4
.. _f9d0d6c: https://github.com/quic/aimet/commit/f9d0d6cb1719ef8eaf2a51b8c0984c50240f01f6
.. _22b5c94: https://github.com/quic/aimet/commit/22b5c94ecf3f743c3954a44fc93de31aab223a47
.. _f547a49: https://github.com/quic/aimet/commit/f547a49db222011c354ad2df6703e0a60ef5c767
.. _339a225: https://github.com/quic/aimet/commit/339a22514ef0aaa1961f82d4832e07d45817779f


2.9.0
=====

* Bug Fixes and Improvements
    * ONNX
        * Rename QuantizeLinear outputs from <...>_int to <...>_q in onnx QDQ export (`e78dbec`_)
        * Preserve I/O names in onnx QDQ export (`35ad990`_)
        * Allow freezing loaded encodings in load_encodings_to_sim (`911af75`_)
        * Represent activation QDQ with uint in encodings 2.0.0 in onnx QDQ export (`92f63f5`_)
        * Allow aimet-onnx to load partial encodings (`6636515`_)
        * Fix onnx sim.export permanently removing quantizers (`9a2a407`_)
        * Fix onnx QDQ export output name swapping bug (`6d1664c`_)
        * Switch AdaRound API naming to num_iterations (`fea395f`_)
    * PyTorch
        * Add native support for Mistral-0.3 (`db99447`_)
        * AdaScale: Update the learning rates for AdaScale learnable parameters (`7336ead`_)
        * AdaScale: Add LR scheduler and add block input sampling probability (`2f05175`_)
        * AdaScale: Maintain LR per model and fix first sample being used during loss computation(`ac05d10`_)
    * Common
        * Add docs to build aimet from source (`ae981f7`_)

.. _e78dbec: https://github.com/quic/aimet/commit/e78dbecb76f5f278baabb6f32a45de299f03a75a
.. _35ad990: https://github.com/quic/aimet/commit/35ad990c4e476f8ef2b51eecbafba1ff25d439cb
.. _911af75: https://github.com/quic/aimet/commit/911af7587ef111e7d90d66db4988e5df218337ee
.. _92f63f5: https://github.com/quic/aimet/commit/92f63f55127f90a6c939d4e8e7fd65189d741e4f
.. _6636515: https://github.com/quic/aimet/commit/66365155f5f0d5620c1bb84321732099ce1d8719
.. _9a2a407: https://github.com/quic/aimet/commit/9a2a40708a73d105cb56152ece5bd127e0ed9474
.. _6d1664c: https://github.com/quic/aimet/commit/6d1664c110d86c401e9715f92cbad10230f489a0
.. _fea395f: https://github.com/quic/aimet/commit/fea395f750de16147a5ce541f2a9723558f0a710
.. _db99447: https://github.com/quic/aimet/commit/db99447da525b114d081acc81d60dfaa95863e79
.. _7336ead: https://github.com/quic/aimet/commit/7336eadb286592eb5f798a689ee5b6e8b918483f
.. _ae981f7: https://github.com/quic/aimet/commit/ae981f73f91580d26024c652a5bbda4d4d8ff77d
.. _2f05175: https://github.com/quic/aimet/commit/2f0517539ce02bff32c79b82501aca543dbefc33
.. _ac05d10: https://github.com/quic/aimet/commit/ac05d10752c3f5034f475b483f2cf049e23d66f6


2.8.0
=====

* New Features
    * ONNX
        * Update aimet_onnx :func:`QuantizationSimModel.__init__` function signature (`cbe67ae`_)
        * Defined new AdaRound API :func:`aimet_onnx.apply_adaround` (`84edcf5`_)
        * Defined new sequential MSE API :func:`aimet_onnx.apply_seq_mse` (`836ab1e`_)
        * Defined new per-layer sensitivity analysis API :func:`aimet_onnx.analyze_per_layer_sensitivity` (`dc34fa4`_)
        * Allowed onnx :func:`QuantizationSimModel.compute_encodings` to take iterables (`2c8ae88`_)
    * PyTorch
        * Added native support for huggingface Phi-3 (`80cd141`_)

* Bug Fixes and Improvements
    * ONNX
        * Made dynamic weights of Conv, ConvTranspose, Gemm, and MatMul follow the symmetry of static weights (`ce68e75`_)
        * aimet-onnx on PyPI is now compatible with onnxruntime-gpu (`6d3aa97`_)
        * Unpinned onnx version (`abe8782`_)
        * Changed default execution provider to CPUExecutionProvider (`e7d10c7`_)
        * Made QcQuantizeOp's data_type attribute always consistent without additional reconfiguration (`8009871`_)
        * Made delta/offset and min/max always consistent (`88706ef`_)
    * PyTorch
        * Made input quantizers always get enabled whenever the input wasn't already quantized (`a2adae2`_)
        * Deprecated saving PyTorch model object during :func:`QuantizationsimModel.export` (`b5521f3`_)

* Known Issues
  * ONNX
      * Adaround runs over 2x slower with onnxruntime 1.20 or higher. The root cause has been identified, and a fix is in progress

.. _cbe67ae: https://github.com/quic/aimet/commit/cbe67ae291f3519f3207d438450d22964f5a8c0d
.. _84edcf5: https://github.com/quic/aimet/commit/84edcf580ac76afa8d128316e03c7737f2599c2d
.. _836ab1e: https://github.com/quic/aimet/commit/836ab1e56de792569155269dbe3c54d717649468
.. _dc34fa4: https://github.com/quic/aimet/commit/dc34fa46e802cc50bfc16cfbc197e3b56d9d8d9e
.. _2c8ae88: https://github.com/quic/aimet/commit/2c8ae88193da0f6284e5dc416ee6af53a9aea701
.. _80cd141: https://github.com/quic/aimet/commit/80cd14176448e586b7b53e624f1dd38b93e78d24
.. _cbe67ae: https://github.com/quic/aimet/commit/cbe67ae291f3519f3207d438450d22964f5a8c0d
.. _ce68e75: https://github.com/quic/aimet/commit/ce68e75f2d55ad07e918f9b0ffb2dc23893ceaf6
.. _6d3aa97: https://github.com/quic/aimet/commit/6d3aa97195317010fe650df7fe612570b53f1d13
.. _abe8782: https://github.com/quic/aimet/commit/abe87827fa77bc6b850289ae35566e7de437c8d1
.. _e7d10c7: https://github.com/quic/aimet/commit/e7d10c799d29beb2b8b36cd4bce8dcaacd1bd9f7
.. _8009871: https://github.com/quic/aimet/commit/8009871262dc702b277b34ae53f70d760e300736
.. _88706ef: https://github.com/quic/aimet/commit/88706eff5301eeb4274b333efbab140a1bc1b5f5
.. _a2adae2: https://github.com/quic/aimet/commit/a2adae2e9ca7ee261bb03e407da0598715b9f933
.. _a2adae2: https://github.com/quic/aimet/commit/a2adae2e9ca7ee261bb03e407da0598715b9f933
.. _b5521f3: https://github.com/quic/aimet/commit/b5521f3fefc5ee405f0596fcf01be670af81cd4a

2.7.0
=====

* New Features
    * PyTorch
        * OmniQuant (experimental) - implement OmniQuant PTQ technique (https://arxiv.org/pdf/2308.13137) for Llama and Qwen2 model families

* Bug Fixes and Improvements
    * ONNX
        * Remove DlCompression, DlEqualization, OpenCV, zlib dependencies
        * Support loading encodings for missing quantizers
        * Set bitwidth of tensor quantizer while loading encodings
    * PyTorch
        * Remove DlCompression, DlEqualization, OpenCV, zlib dependencies
        * Export encodings for data movement operations in ONNX QDQ export
        * AdaScale (experimental) - support for updating Conv2D layers in blocks
        * AdaScale (experimental) - update API to take num_iterations instead of num_epochs

2.6.0
=====

* New Features
    * ONNX
        * Support for passing onnxruntime EPs directly to :func:`QuantizationSimModel.__init__`
    * PyTorch
        * Support for simulating float8 quantization
        * Experimental: Added :func:`aimet_torch.onnx.export` API for exporting :mod:`QuantizationSimModel` to onnx QDQ graph
        * Added native support for huggingface Llama, Qwen2, and Gemma3 (`1493fe1`_)

* Bug Fixes and Improvements
    * ONNX
        * Reduced CPU and GPU memory usage during sequential MSE
        * Fixed AMP generating incompatible quantizer configurations
        * Fixed AMP errors with dynamic Conv ops
        * Aligned computation of symmetric encodings with :mod:`aimet_torch`
    * PyTorch
        * Fixed AttributeError when catching :func:`torch.onnx.export` failures during QuantSim export
        * Fixed errors being thrown when deepspeed import fails
        * Aligned input and output encodings for Resize layers
        * Added supergroup fusion handling for LeakyRelu layers
        * Docs: Updated LoRA user guide

* Deprecations:
    * ONNX
        * Deprecated `use_cuda`, `device`, `rounding_mode`, and `use_symmetric_encodings` args to :func:`QuantizationSimModel.__init__`

.. _1493fe1: https://github.com/quic/aimet/commit/1493fe1d8e40e5b8d041f11603b2d60cd76d94d3

2.5.0
=====

* New Features
    * ONNX
        * Added a new set_quantizers() API to QuantizationSimModel
    * PyTorch
        * Added new api to fold param quantizers
        * Experimental: AdaScale - a new post-training quantization technique

* Bug Fixes
    * ONNX
        * Cleaned up tempfiles generated by large model export
    * PyTorch
        * Fixed nullptr error in FloatEncoding
        * Checked wrong parameter access only upon AttributeError
        * Changed to import spconv lazily
        * Fixed type error in transformer utils

2.4.0
=====

* New Features
    * ONNX
        * Introduced option to export only encodings
    * Common
        * Added RMSNormalization in default AIMET config

* Bug Fixes
    * ONNX
        * Removed cublas dependency from the libpymo executable
        * Represent y_zero_point as int
        * Represent per-block scale as int
    * PyTorch
        * SeqMSE optimizes nested modules once improving turn-around time
        * CrossLayerEqualization does not replaces ReLU6 with ReLU automatically
        * AMP creates distict quantizer groups for model inputs

2.3.0
=====

* New Features
    * ONNX
        * Upgraded CUDA to 12.1.0
        * Upgraded ONNX-Runtime to 1.19.2
        * Reduced :func:`QuantizationSimModel.export()` time

* Bug Fixes
    * ONNX
        * Fixed bug in :func:`QuantizationSimModel.export()` to export ONNX models with external weights to one file

2.2.0
=====

* New Features
    * PyTorch and ONNX
        * Added "min_max" (`QuantScheme.min_max`) as a new name for "post_training_tf" quant scheme
    * ONNX
        * Introduced supergroup pattern-matching for complicated patterns such as LayerNormalization and RMSNorm
* Bug Fixes
    * PyTorch
        * Restored :mod:`aimet_torch.v1` tf-enhanced behavior
        * Updated Sequential MSE candidate logic to compute encoding candidates. Vectorized blockwise sequential MSE loss calculation for :mod:`nn.Linear`
    * ONNX
        * Fixed bug in :func:`QuantizationSimModel._tie_quantizers()` which propagates encodings to first op of parent ops if parent op is not quantizable

2.1.0
=====

* New Features
    * PyTorch and ONNX
        * AIMET QuantSim by default uses per-channel quantization for weights instead of per-tensor [Breaking change]
        * AIMET QuantSim exports encoding json schema version 1.0.0 by default
    * PyTorch
        * AIMET now quantizes scalar inputs of type :mod:`torch.nn.Parameter` - these were not quantized in prior releases
        * Published recipe for performing LoRA QAT - using LoRA adapters to recover quantized accuracy of the base model. Includes recipes for weight-only (WQ) and weight-and-activation (QWA) QAT

* Bug Fixes
    * PyTorch
        * Fixed a bug that prevented Adaround from caching data samples with PyTorch versions 2.6 and later

2.0.0
=====

* New Features
    * Common
        * Reorganized the documentation to more clearly explain AIMET procedures
        * Redesigned the documentation using the `Furo theme <https://sphinx-themes.readthedocs.io/en/latest/sample-sites/furo/>`_
        * Added post-AIMET procedures on how to take AIMET quantized model to |qnn| and |qai_hub|
    * PyTorch
        * BREAKING CHANGE: :mod:`aimet_torch.v2` has become the default API. All the legacy APIs are migrated to :mod:`aimet_torch.v1` subpackage, for example from :mod:`aimet_torch.qc_quantize_op` to :mod:`aimet_torch.v1.qc_quantize_op`
        * Added Manual Mixed Precision Configurator (Beta) to make it easy to configure a model in Mixed Precision.
    * ONNX
        * Optimized :func:`QuantizationSimModel.__init__` latency
        * Align :mod:`ConnectedGraph` representation with onnx graph

* Bug Fixes
    * ONNX
        * Bug fixes for Adaround
        * Bug fixes for BN fold

* Upgrading
    * PyTorch
        * aimet_torch 2 is fully backward compatible with all the public APIs of aimet_torch 1.x. If you are using low-level components of :class:`QuantizationSimModel`, please see :doc:`Migrate to aimet_torch 2 </apiref/torch/migration_guide>`.

1.35.1
======

* PyTorch
    * Fixed package versioning for compatibility with latest pip version

1.35.0
======

* PyTorch
    * Added support for W16A16 in Autoquant.
* Deprecation Notice
    * Support for Pytorch 1.13 is deprecated. It will be removed in next release.
* ONNX
    * Optimized Memory and Speed utilization (for CPU).

1.34.0
======

* PyTorch
    * Added support for WSL2
    * CUDA version upgraded for Pytorch 2.1
    * Extended QuantAnalyzer functionality for LLM range analysis
* Keras
    * Adds support for certain TFOpLambda layers created by tf functional calls.
* ONNX
    * Upgraded AIMET to support ONNX version 1.16.1 and ONNXRUNTIME version 1.18.1.


1.33.5
======

* PyTorch
    * Various bugfixes/QoL updates for LoRA
    * Updated minimum scale value and registered additional custom quantized ops with QuantSim 2.0

1.33.0
======

* PyTorch
    * Enhancements done in export pipeline for GPU memory optimization with LLMs.
    * [Experimental] Added support for handling of LoRA (via PEFT API) in AIMET. and enabled export of
      required artifacts for QNN.
    * Added examples for training pipeline with for distributed KD-QAT.
    * [Experimental] Added support for block wise quantization (BQ) to support w4fp16 format, and the
      low-power block quantization (LPBQ) to support w4a8 and w4a16 formats. This feature needs
      QuantSim V2.

1.32.0
======

* PyTorch
    * Added MultiGPU support for Adaround.
    * Upgraded AIMET to support PyTorch version 2.1 as a new variant. AIMET with PyTorch version 1.13
      remains the default.
* Keras
    * For models with SeparableConv2D layers, use model_preparer first before applying any quantization
      API.
* Common
    * Upgraded AIMET to support Ubuntu22 and Python3.10 for all AIMET variants.

1.31.0
======

* ONNX
    * Added support for custom ops in QuantSim, CLE, AdaRound and AMP.
    * Added support for Quant Analyzer.
* Keras
    * Added support for unrolled quantized LSTM with only Quantsim in PTQ mode.
    * Fix for ReLU Encoding min going past 0 for QAT.
    * Fixes Input Quantizers for TFOpLambda Layers (kwargs)
    * Fixes logic for placing input quantizers

1.30.0
======

* ONNX
    * Upgraded AIMET to support Onnx version 1.14 and ONNXRUNTIME version 1.15.
    * Added support for AutoQuant.

1.29.0
======

* Keras
    * Fixes issues with TF Op Lambda Layers in Qc Quantize Wrappers call.
* PyTorch
    * [experimental] Support for embedding AIMET encodings within the graph using ONNX quantize/dequantize
      operators. Currently this option is only supported when using 8bit per-tensor quantization.
* ONNX
    * Added support for Adaround.

1.28.0
======

* Keras
    * Added Support for Spatial SVD Compression feature.
    * [experimental] Debugging APIs have been added for dumping intermediate tensor outputs. This data
      can be used with current QNN/SNPE tools for debugging accuracy problems.
* PyTorch
    * Upgraded AIMET Pytorch default version to 1.13. AIMET remains compatible with Pytorch version 1.9.
* ONNX
    * [experimental] Debugging APIs have been added for dumping intermediate tensor outputs. This data
      can be used with current QNN/SNPE tools for debugging accuracy problems.

1.27.0
======

* Keras
    * Update support for TFOpLambda layers in Batch Norm Folding with extra call args/kwargs.
* PyTorch
    * Added AIMET to support PyTorch version 1.13.0. Only ONNX opset 14 is supported for export.
    * [experimental] Debugging APIs have been added for dumping intermediate tensor data. This data can
      be used with current QNN/SNPE tools for debugging accuracy problems. Layer Output Generation API
      gives incorrect tensor data for the layer just before Relu when used for original FP32 model.
    * [experimental] Support for embedding AIMET encodings within the graph using ONNX quantize/dequantize
      operators. Currently this is option is only supported when using 8bit per-tensor quantization.
    * Fixed a bug in AIMET QuantSim for PyTorch models to handle non-contiguous tensors.
* ONNX
    * AIMET support for ONNX 1.11.0 has been added. However there is currently limited op support
      in QNN/SNPE. If the model fails to load please continue to use opset 11 for export.
* TensorFlow
    * [experimental] Debugging APIs have been added for dumping intermediate tensor outputs. This data
      can be used with current QNN/SNPE tools for debugging accuracy problems.

1.26.0
======

* Keras
    * Added a feature called BN Re-estimation that can improve model accuracy after QAT for INT4
      quantization.
    * Updated the AutoQuant feature to automatically choose the optimal calibration scheme, create an
      HTML report on which optimizations were applied.
    * Update to Model Preparer to replace separable conventional with depth wise and point wise conv
      layers.
    * Fixes BN fold implementation to account for a subsequent multi-input layer
    * Fixed a bug where min/max encoding values were not aligned with scale/offset during QAT.
* PyTorch
    * Several bug fixes
* TensorFlow
    * Added a feature called BN Re-estimation that can improve model accuracy after QAT for INT4
      quantization
    * Updated the AutoQuant feature to automatically choose the optimal calibration scheme, create an
      HTML report on which optimizations were applied.
    * Fixed a bug where min/max encoding values were not aligned with scale/offset during QAT.
* Common
    * Documentation updates for taking AIMET models to target.
    * Standalone Batchnorm layers parameter’s conversion such that it will behave as linear/dense layer.
    * [Experimental] Added new Architecture Checker feature to identify and report model architecture
      constructs that are not ideal for quantized runtimes. Users can utilize this information to change
      their model architectures accordingly.

1.25.0
======

* Keras
    * Added QuantAnalyzer feature
    * Adds Batch Normalization folding for Functional Keras Models. This allows the default config files
      to work for super grouping.
    * Resolved an issue with quantizer placement in Sequential blocks in subclassed models
* PyTorch
    * Added AutoQuant V2 which includes advanced features such as out-of-the-box inference, model
      preparer, quant scheme search, improved summary report, etc.
    * Fixes to resolve minor accuracy diffs in the learnedGrid quantizer for per-channel quantization
    * Fixes to improve EfficientNetB4 accuracy w/respect to target
    * Fixed rare case where quantizer may calculate incorrect offset when generating QAT 2.0 learned
      encodings
* TensorFlow
    * Added QuantAnalyzer feature
    * Fixed an accuracy issue due to rare cases where the incorrect BN epsilon was being used
    * Fixed an accuracy issue due to Quantsim export incorrectly recomputing QAT2.0 encodings
* Common
    * Updated AIMET python package version format to support latest pip
    * Fixed an issue where not all inputs might be quantized properly

1.24.0
======

* PyTorch
    * Fixes to resolve minor accuracy diffs in the learnedGrid quantizer for per-channel quantization
    * Added support for AMP 2.0 which enables faster automatic mixed precision
    * Added support for QAT for INT4 quantized models – includes a feature for performing BN Re-estimation
      after QAT
* Keras
    * Added support for AMP 2.0 which enables faster automatic mixed precision
    * Support for basic transformer networks
    * Added support for subclassed models. The current subclassing feature includes support for only a
      single level of subclassing and does not support lambdas.
    * Added QAT per-channel gradient support
    * Minor updates to the quantization configuration
    * Fixed QuantSim bug where layers using dtypes other than float were incorrectly quantized
* TensorFlow
    * Added an additional prelu mapping pattern to ensure proper folding and quantsim node placement
    * Fixed per-channel encoding representation to align with Pytorch and Keras
* Common
    * Export quantsim configuration for configuring downstream target quantization

1.23.0
======

* PyTorch
    * Fixed backward pass of the fake-quantize (QcQuantizeWrapper) nodes to handle symmetric mode
      correctly
    * Per-channel quantization is now enabled on a per-op-type basis
    * Support for recursively excluding module from a root module in QuantSim
    * Support for excluding layers when running model validator and model preparer
    * Reduced memory usage in AdaRound
    * Fixed bugs in AdaRound for per-channel quantization
    * Made ConnectedGraph more robust when identifying custom layers
    * Added jupyter notebook-based examples for the following features
    * AutoQuant: Added support for sparse conv layers in QuantSim (experimental)
* Keras
    * Added support for Keras per-channel quantization
    * Changed interface to CLE to accept a pre-compiled model
    * Added jupyter notebook-based examples for the following features: Transformer quantization
* TensorFlow
    * Fix to avoid unnecessary indexing in AdaRound
* Common
    * TF-enhanced calibration scheme has been accelerated using a custom CUDA kernel. Runs significantly
      faster now.
    * Installation instructions are now combined with rest of the documentation (User-Guide and API docs)

1.22.2
======

* Tensorflow
    * Added support for supergroups : MatMul + Add
    * Added support for TF-Slim BN name with backslash
    * Added support for Depthwise + Conv in CLS

1.22.1
======

* PyTorch
    * Added support for QuantizableMultiHeadAttention for PyTorch nn.transformer layers
    * Support functional conv2d in model preparer
    * Enable qat with multi gpu
    * Optimize forward pass logic of PyTorch QAT 2.0
    * Fix functional depthwise conv support on model preparer
    * Fix bug in model validator to correctly identify functional ops in leaf module
    * Support dynamic functional conv2d in model preparer
    * Added updated default runtime config, also a per-channel one.
    * Include residing module info in model validator
* Keras
    * Support for Keras MultiHeadAttention Layer

1.22.0
======

* PyTorch
    * Support for simulation and QAT for PyTorch transformer models (including support for torch.nn mha and
      encoder layers)

1.21.0
======

* PyTorch
    * PyTorch QuantAnalyzer - Visualize per-layer sensitivity and per-quantizer PDF histograms
    * PyTorch QAT with Range Learning: Added support for Per Channel Quantization
    * PyTorch: Enabled exporting of encodings for multi-output leaf module
* TensorFlow
    * * New feature: TensorFlow AutoQuant - Automatically apply various AIMET post-training quantization techniques
    * Adaround: Added ability to use configuration file in API to adapt to a specific runtime target
    * Adaround: Added Per-Channel Quantization support
    * TensorFlow QuantSim: Added support for FP16 inference and QAT
    * TensorFlow Per Channel Quantization
        * Fixed speed and accuracy issues
        * Fixed zero accuracy for 16-bits per channel quantization
        * Added support for DepthWise Conv2d Op
    * Multiple other bug fixes

1.20.0
======

* PyTorch
    * Propagated encodings for ONNX Ops that were expanded from a single PyTorch Op
* TensorFlow
    * Upgraded AIMET to support TensorFlow version 2.4. AIMET remains compatible with TensorFlow
      version 1.15
* Common
    * Added Jupyter Notebooks for Examples
    * Multiple bug fixes
    * Removed version pinning of many dependent software packages

1.19.1
======

* PyTorch
    * Added CLE support for Conv1d, ConvTranspose1d and Depthwise Separable Conv1d layers
    * Added High-Bias Fold support for Conv1D layer
    * Modified Elementwise Concat Op to support any number of tensors
    * Minor dependency fixes

1.18.0
======

* Common
    * Multiple bug fixes
    * Additional feature examples for PyTorch and TensorFlow

1.17.0
======

* TensorFlow
    * Add Adaround TF feature
* PyTorch
    * Added Examples for Torch quantization, and Channel Pruning & Spatial SVD compression

1.16.2
======

* PyTorch
    * Added a new post-training quantization feature called AdaRound, which stands for AdaptiveRounding
    * Quantization simulation and QAT now also support recurrent layers (RNN, LSTM, GRU)

1.16.1
======

* Added separate packages for CPU and GPU models. This allows users with CPU-only hosts to run AIMET.
* Added separate packages for PyTorch and TensorFlow. Reduces the number of dependencies that users would need to install.

1.16.0
======

* Ported AIMET PyTorch to work with PyTorch ver 1.7.1 with CUDA 11.0
* AIMET PyTorch and AIMET TensorFlow are now available as separate packages
* Version of the AIMET PyTorch and AIMET TensorFlow packages for CPU-only machines are now available

1.13.0
======

* PyTorch
    * Added Adaptive Rounding feature (AdaRound) for PyTorch.
    * Various bug fixes.
