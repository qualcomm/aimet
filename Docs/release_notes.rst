.. include:: abbreviation.txt

.. _rn-index:

#############
Release notes
#############

2.32.1
======

* Bug fixes and Improvements
    * ONNX
        * Fix onnxruntime dispatching to incorrect function overload (`a969ada`_)

.. _a969ada: https://github.com/qualcomm/aimet/commit/a969ada


2.32.0
======

* Bug fixes and Improvements
    * ONNX
        * Add C++ support for bfloat16 quantization (`ca7d3e01b`_)
        * Fix large model support with protobuf 7.x (`9ef22519a`_)
        * Skip QDQ pair scale/zp in duplicate_shared_initializers (`05e8332ce`_)
        * Handle Identity passthrough in duplicate_shared_initializers (`1b27d9841`_)
        * Fix SpinQuant embed_tokens filter to exclude non-embedding Gathers (`81b80411a`_)
        * Inline fused supergroups after encoding propagation (`68fdcb673`_)

    * Torch
        * Disable output quantizers of reused modules before output encoding propagation (`b66b9a1e0`_)
        * Inline Q/DQ nodes statically without re-invoking torch.export (`de79ae495`_)
        * Stop incorrect encoding propagation through non-grid-preserving ops (`66b2834fc`_)

.. _ca7d3e01b: https://github.com/qualcomm/aimet/commit/ca7d3e01b
.. _9ef22519a: https://github.com/qualcomm/aimet/commit/9ef22519a
.. _05e8332ce: https://github.com/qualcomm/aimet/commit/05e8332ce
.. _1b27d9841: https://github.com/qualcomm/aimet/commit/1b27d9841
.. _81b80411a: https://github.com/qualcomm/aimet/commit/81b80411a
.. _b66b9a1e0: https://github.com/qualcomm/aimet/commit/b66b9a1e0
.. _de79ae495: https://github.com/qualcomm/aimet/commit/de79ae495
.. _68fdcb673: https://github.com/qualcomm/aimet/commit/68fdcb673
.. _66b2834fc: https://github.com/qualcomm/aimet/commit/66b2834fc


2.31.0
======

* New Features
    * ONNX
        * Support Qwen 3VL in AdaScale ONNX (`35d2440db`_)

    * Torch
        * Add Gemma 3 support for AdaScale (`a2da0de9c`_)
        * LoRA integration (`0b90d8a4f`_)

* Removed Features
    * Torch
        * Delete AutoQuant (`a2382756e`_)
        * Delete bias correction (`a2382756e`_)
        * Delete quantizable transformer (`a2382756e`_)
        * Delete winnow (`a2382756e`_)

* Bug fixes and Improvements
    * ONNX
        * Fuse supergroups to ONNX function nodes in QuantSim init (`441ac6dc8`_)
        * Enable ONNX initializer deduplication pass in torch>=2.12 (`21dc8e05f`_)
        * Detect post-writing norm incompatibility in ONNX SpinQuant (`85bdbdb88`_)
        * Remove incorrect entries from grid-preserving ops list (`4007d7f3f`_)
        * Set self.session = None to avoid double memory allocation during rebuild session (`18664a4db`_)
        * Give fused supergroup nodes intuitive naming (`a775b6e09`_)

    * Torch
        * Raise ValueError for unsupported architectures in PyTorch SpinQuant (`a96261475`_)

* Documentation
    * Add zero_point_shift to 1.0.0 encoding spec documentation (`094deadfa`_)
    * Add float8/float4 encoding to 2.0.0 spec documentation (`02e75aa95`_)

.. _441ac6dc8: https://github.com/quic/aimet/commit/441ac6dc8
.. _21dc8e05f: https://github.com/quic/aimet/commit/21dc8e05f
.. _35d2440db: https://github.com/quic/aimet/commit/35d2440db
.. _a2da0de9c: https://github.com/quic/aimet/commit/a2da0de9c
.. _0b90d8a4f: https://github.com/quic/aimet/commit/0b90d8a4f
.. _85bdbdb88: https://github.com/quic/aimet/commit/85bdbdb88
.. _4007d7f3f: https://github.com/quic/aimet/commit/4007d7f3f
.. _18664a4db: https://github.com/quic/aimet/commit/18664a4db
.. _a775b6e09: https://github.com/quic/aimet/commit/a775b6e09
.. _a96261475: https://github.com/quic/aimet/commit/a96261475
.. _a2382756e: https://github.com/quic/aimet/commit/a2382756e
.. _094deadfa: https://github.com/quic/aimet/commit/094deadfa
.. _02e75aa95: https://github.com/quic/aimet/commit/02e75aa95


2.30.0
======

* New Features
    * ONNX
        * Extend SpinQuant support for Vision-Language Models (VLM) (`e5cd62847`_)

    * Torch
        * Remove legacy aimet_torch v1. v2 is now the sole API (`4192da749`_, `00e3c7220`_)

* Bug fixes and Improvements
    * ONNX
        * Improve set_and_freeze_param_encodings (`39bf1f69b`_, `d320cae7e`_)
        * Optimize GPU calibration for fp16 models (`d7faee0e9`_)
        * Only save model with external data if necessary (`29fbef23e`_)

    * Common
        * Fix exception rule bug with mixed precision (`c9add2219`_)

* Documentation
    * Add SpinQuant ONNX documentation and examples (`37454b34e`_)
    * Document 2.0.0 encoding specification (`7e54291c3`_)

.. _e5cd62847: https://github.com/quic/aimet/commit/e5cd62847
.. _4192da749: https://github.com/quic/aimet/commit/4192da749
.. _00e3c7220: https://github.com/quic/aimet/commit/00e3c7220
.. _39bf1f69b: https://github.com/quic/aimet/commit/39bf1f69b
.. _d320cae7e: https://github.com/quic/aimet/commit/d320cae7e
.. _d7faee0e9: https://github.com/quic/aimet/commit/d7faee0e9
.. _29fbef23e: https://github.com/quic/aimet/commit/29fbef23e
.. _c9add2219: https://github.com/quic/aimet/commit/c9add2219
.. _37454b34e: https://github.com/quic/aimet/commit/37454b34e
.. _7e54291c3: https://github.com/quic/aimet/commit/7e54291c3


2.29.0
======

* New Features
    * ONNX
        * Add support for Qwen 2.5 VL in aimet-onnx (`f25668610`_)

    * Torch
        * Support OOTB quantization of nn.MultiHeadAttention (`4d19f470f`_)
        * Support OOTB quantization of Qwen 3.5 normalization layers (`01b912f65`_)
        * Support OOTB quantization of InternVL GELU (`c5f65b782`_)

* Bug fixes and Improvements
    * Common
        * Make export_int32_bias default to True if encoding_version >= 2.0.0 (`22876ca30`_)

    * ONNX
        * Optimize QDQ latency for fp16 models (`c817a17a5`_)
        * Support pattern matching LayerNormalization without bias (`84f880aee`_)
        * Make from_onnx_export ignore unloadable encodings by default (`1b5072751`_)
        * Enable loading models with redundant back-to-back QDQ using from_onnx_qdq (`0f2be91bf`_)
        * Skip folding BatchNormalization when the Conv layer has shared weights (`8f552b7a6`_)
        * Fix bug in standalone BatchNormalization fold with shared tensors (`eb7ae4b72`_)

    * Torch
        * Disable activation quantizers for re-used stateless nn.Modules (`8f552b7a6`_)


.. _f25668610: https://github.com/quic/aimet/commit/f25668610
.. _4d19f470f: https://github.com/quic/aimet/commit/4d19f470f
.. _01b912f65: https://github.com/quic/aimet/commit/01b912f65
.. _c5f65b782: https://github.com/quic/aimet/commit/c5f65b782
.. _22876ca30: https://github.com/quic/aimet/commit/22876ca30
.. _c817a17a5: https://github.com/quic/aimet/commit/c817a17a5
.. _84f880aee: https://github.com/quic/aimet/commit/84f880aee
.. _8f552b7a6: https://github.com/quic/aimet/commit/8f552b7a6
.. _1b5072751: https://github.com/quic/aimet/commit/1b5072751
.. _eb7ae4b72: https://github.com/quic/aimet/commit/eb7ae4b72
.. _0f2be91bf: https://github.com/quic/aimet/commit/0f2be91bf
.. _8f552b7a6: https://github.com/quic/aimet/commit/8f552b7a6


2.28.0
======

* New Features
    * Torch
        * Add resumable checkpointing for AdaScale optimization (`20ecb0a`_)

    * Common
        * Migrate pybind11 bindings to Cython using Python's Stable ABI to enable Python-version-independent wheels (`0d6f856`_)

* Bug fixes and Improvements
    * Torch
        * Fix rescale encodings not propagating with shared scale values (`d9f3a90`_)

* Documentation
    * Update docs and examples to use new API for setting lm_head precision (`ac3e11e`_)

.. _20ecb0a: https://github.com/quic/aimet/commit/20ecb0ae89a1261250d5b5a8128802bc8d442bae
.. _0d6f856: https://github.com/quic/aimet/commit/0d6f856cf8b6c1bbb3adc5d8d5e2d0a2c14c72e6
.. _d9f3a90: https://github.com/quic/aimet/commit/d9f3a9094876db4834e9b1359b36627ff3ef6570
.. _ac3e11e: https://github.com/quic/aimet/commit/ac3e11e1c6d835a6f65b83e29fd5e36479a9a750


2.27.0
======

* Bug fixes and Improvements
    * ONNX
        * Add `force_activation_as` option to export APIs to control activation signedness (`3583462`_)

    * Torch
        * Reduce quantize-dequantize latency overhead (`9ca3bf4`_, `525e993`_, `b3de9a2`_)
        * Optimize inference speed for GenAITests models (`cacd5cc`_, `b6ea5bd`_, `30ab60a`_)
        * Allow checkpointing and loading during SeqMSE optimization (`4eb97f0`_)
        * Fix SeqMSE error when model contains unquantized Conv/Linear layers (`3dd4ca9`_)
        * Populate scalar constant Mul/Div output encodings at export (`1228394`_, `169952d`_, `ca2a324`_)
        * Propagate tensor encodings through scalar Mul/Div operations (`54c7462`_, `2cfd07e`_)

    * Common
        * Propagate concat input quantizers to output when possible (`5ee0f13`_)

.. _9ca3bf4: https://github.com/quic/aimet/commit/9ca3bf4cf74bc6b6db2a4d508ee260303a466edf
.. _1228394: https://github.com/quic/aimet/commit/1228394ae2f534890cf29062332062105e6cd9db
.. _cacd5cc: https://github.com/quic/aimet/commit/cacd5cc7a08ba91b359d2fdcab30adaaa9f99df8
.. _525e993: https://github.com/quic/aimet/commit/525e993be3daec18ac7733b6822281bace4d0ca3
.. _4eb97f0: https://github.com/quic/aimet/commit/4eb97f093e37e2d8c71d40a7aa7222a32063b4d7
.. _3dd4ca9: https://github.com/quic/aimet/commit/3dd4ca9f44159b27a9eb3e802dd9fdde5af90e16
.. _b6ea5bd: https://github.com/quic/aimet/commit/b6ea5bd022d4622f77a59694ef5d90fde0f5ac1e
.. _169952d: https://github.com/quic/aimet/commit/169952d8582a289962eb3a168bb5aa70132c86d8
.. _ca2a324: https://github.com/quic/aimet/commit/ca2a324a99724a1222818fe5bb53ea0a47fb3850
.. _30ab60a: https://github.com/quic/aimet/commit/30ab60a7f210a8297e66c743d6f33a92aa186689
.. _b3de9a2: https://github.com/quic/aimet/commit/b3de9a2b3591648e47067ac619ae8ceeb7984fd0
.. _54c7462: https://github.com/quic/aimet/commit/54c746272d29058219eb0b57725de5cacd97a6ba
.. _2cfd07e: https://github.com/quic/aimet/commit/2cfd07edccc6a5715ff99294dbc6057159acdd73
.. _5ee0f13: https://github.com/quic/aimet/commit/5ee0f13d08b00294460566fe7d4bde5244f9c6db
.. _3583462: https://github.com/quic/aimet/commit/35834628c5631853d3d7f36fd32b3ea1dceb8926


2.26.0
======

* Bug fixes and Improvements
    * ONNX
        * Implement onnxscript RMSNorm fusion for improved graph optimization (`68710d9`_)
        * Propagate encoding through Concat during ONNX QDQ export (`4811a34`_)
        * Export scale & offset as initializers instead of Constants in ONNX QDQ export (`ea9a619`_)
        * Fix AdaScale (aimet-onnx) for Qwen3 models (`beac8f8`_)
        * Fix BN fold for YOLO models (`bae9953`_)

    * Torch
        * Support int2 ONNX QDQ export (`5fa79cf`_)
        * Significantly improve ONNX export performance by eliminating O(N^2) iterations and redundant Q/DQ operations (`695465e`_, `cb1f9ae`_, `abe0ef5`_, `c547cfb`_, `310b43d`_, `fb9629d`_, `d54efa0`_)
        * Add native support for Qwen3 MoE models (`389d71f`_, `ab6e810`_)
        * Fix triton kernel bug upon transposed inputs (`a1f6795`_)
        * Fix GPU memory leak in AdaScale optimization loop (`f52f2e2`_)
        * Fix AdaScale device error with caching disabled (`964d11f`_)
        * Work around torch.compile bugs and exclude internal quantization methods from compilation (`b8bcb47`_, `d518f35`_)
        * Fix tie quantizers removing relu encoding constraint (`3cc7252`_)
        * Fail immediately without retrying upon torch.cuda.OutOfMemoryError (`4f84eb1`_)
        * Release blockwise sampler input memory before yielding to reduce memory usage (`ee3d193`_)
        * Add aimet_torch.v1 end-of-life warning (`8fc52c6`_)
        * Use whitelist approach for enabling per-channel quantization in quantsim config (`817d3b1`_)

    * Common
        * Tie concat and interpolation op quantizers by default with safe edge case handling (`5ce7229`_, `5084af3`_)
        * Implement supergroup unrolling without name mangling (`e351112`_)
        * Treat CG_split as grid-preserving op (`738ee26`_)
        * Handle dynamic matmul add in connected graph passes (`3c0de8e`_)

* Documentation
    * Add AdaScale documentation with HuggingFace LLM example (`c403562`_)
    * Update doc code examples to use aimet_torch.onnx.export (`fed2a06`_)

.. _68710d9: https://github.com/quic/aimet/commit/68710d9bf6cc0eeaed90b529671f8873a2f6bc56
.. _5fa79cf: https://github.com/quic/aimet/commit/5fa79cfb394b84a7fc469134efbe322d854b2c3d
.. _4811a34: https://github.com/quic/aimet/commit/4811a34a20d6aa209bd53178fa2cacdd9e147372
.. _ea9a619: https://github.com/quic/aimet/commit/ea9a619539382ad9f02e8d097ad7068abd62f58e
.. _beac8f8: https://github.com/quic/aimet/commit/beac8f80226d57adcad6a5cc7c534afbea707045
.. _bae9953: https://github.com/quic/aimet/commit/bae99538e0043ffadc36a17cb5757687f8979edc
.. _695465e: https://github.com/quic/aimet/commit/695465ed29923ecee3ae54a87a29ffd1c884dadb
.. _cb1f9ae: https://github.com/quic/aimet/commit/cb1f9aee2e5f484c53bc3094a74e5243835c8bc1
.. _abe0ef5: https://github.com/quic/aimet/commit/abe0ef597d70af7429625d0046c82df9cdef3531
.. _c547cfb: https://github.com/quic/aimet/commit/c547cfb4d279055349e7a202a65ea692c71ad8a8
.. _310b43d: https://github.com/quic/aimet/commit/310b43d973a7d06cc63fb632eeebf1c9cbd385d7
.. _fb9629d: https://github.com/quic/aimet/commit/fb9629df4bdd9ccdd939cd93c78778e24165bce2
.. _d54efa0: https://github.com/quic/aimet/commit/d54efa030dcd525cdccd3e9ecd3b71a8d5b01cb5
.. _389d71f: https://github.com/quic/aimet/commit/389d71fc418e213e535e09bc24a413923b933a13
.. _ab6e810: https://github.com/quic/aimet/commit/ab6e81066ff478d0c086ea401791ce0af8252b03
.. _a1f6795: https://github.com/quic/aimet/commit/a1f6795daaafff3adca9f134b643be6b76b74a9a
.. _f52f2e2: https://github.com/quic/aimet/commit/f52f2e247db6cb3da3f77347b8800f0cb5d9fd05
.. _964d11f: https://github.com/quic/aimet/commit/964d11f2ea8833f7185405cfe7a7722189ed1648
.. _b8bcb47: https://github.com/quic/aimet/commit/b8bcb4793869bbe1007bcfdd70c822b2f717072f
.. _d518f35: https://github.com/quic/aimet/commit/d518f35a2873eacdaa941d368c028c403e76ba2d
.. _3cc7252: https://github.com/quic/aimet/commit/3cc72529f747ab3e4355fe8cee60e92cdbd13c4c
.. _4f84eb1: https://github.com/quic/aimet/commit/4f84eb1e481d239edeb42d6c724746374dae3d79
.. _ee3d193: https://github.com/quic/aimet/commit/ee3d1933e80ac37c79cf52b53d5a1528b16987ea
.. _8fc52c6: https://github.com/quic/aimet/commit/8fc52c6c7c710c1c9989e493b0f6d6936239dd0a
.. _dfa20ee: https://github.com/quic/aimet/commit/dfa20ee4df195d853adcfff503de0f27a8eb7208
.. _817d3b1: https://github.com/quic/aimet/commit/817d3b1f8f521750737d974f629d0cf54b29861f
.. _5ce7229: https://github.com/quic/aimet/commit/5ce72290e2e6614e29fe3c8363695d41cb689430
.. _5084af3: https://github.com/quic/aimet/commit/5084af38f40e297a0f73240c8ce8f9bad40880a0
.. _e351112: https://github.com/quic/aimet/commit/e35111218ebad4b907443a0a9fb874df3d1b1dcf
.. _738ee26: https://github.com/quic/aimet/commit/738ee26272a04d6220bb0342a259a8690dc91e28
.. _3c0de8e: https://github.com/quic/aimet/commit/3c0de8e575fc159d4b0130def6f813c0d9f255ce
.. _c403562: https://github.com/quic/aimet/commit/c4035624e65de059e9c11d7e201a1aab7f3b35b8
.. _fed2a06: https://github.com/quic/aimet/commit/fed2a0600c2f45dc9b1bb31d88fc55133d2d0cd3


2.25.1
======
* Bug fixes and Improvements
    * ONNX
        * Fix for encoding propagation for concat layers (`5084af3`_)
    * Torch
        * Fix to reduce GPU RAM usage for AdaScale for Qwen 3 VL model (`ee3d193`_)

.. _5084af3: https://github.com/quic/aimet/commit/5084af38f40e297a0f73240c8ce8f9bad40880a0
.. _ee3d193: https://github.com/quic/aimet/commit/ee3d1933e80ac37c79cf52b53d5a1528b16987ea


2.25.0
======

* Bug fixes and Improvements
    * ONNX
        * Reduced peak CPU memory usage for AdaScale and SeqMSE techniques (`28f89a7`_)
        * Reduced peak CUDA memory usage for AdaScale technique (`a29f44f`_)
        * Added support for Qwen3 VL models in GenAITests  (`c014961`_)
        * ONNX-IR based supergroup pattern detection and replacement (`9972c1b`_)
        * Tie concat and interpolation ops by default (`a8ac6f4`_)

    * Torch
        * Bug fix for onnx qdq export with control flow ops (`ae1abd1`_)
        * Use Triton kernels by default if available (`3adcbee`_)
        * Introduces `block_size` parameter to EncodingAnalyzer (`e250abd`_)
        * Always export encodings as uint (`ae7d5ef`_)
        * float4/8 QDQ export support (`135a0af`_)
        * Support loading zero_point_shift with sim.load_encodings() (`624ba30`_)
        * Support built-in quantization of SyncBatchNorm (`1e8eceb`_)


.. _28f89a7: https://github.com/quic/aimet/commit/28f89a7b4a212d651c62f4a25433ee4a41e25d55
.. _a29f44f: https://github.com/quic/aimet/commit/a29f44ff6fa1c268da37c71ed0bf44014161d43e
.. _c014961: https://github.com/quic/aimet/commit/c0149612d5b2d1ee629f6183a5c023ed9e9f09fd
.. _ae1abd1: https://github.com/quic/aimet/commit/ae1abd1462de8d958fde33df122603300790d31f
.. _3adcbee: https://github.com/quic/aimet/commit/3adcbeeed4b85845eeb523a50cc9e371b03d96fb
.. _9972c1b: https://github.com/quic/aimet/commit/9972c1b5bfabcfa1b1487fecbe2aee16e94ca591
.. _e250abd: https://github.com/quic/aimet/commit/e250abd61d0a95c120aa755e87b51d0270cee95c
.. _ae7d5ef: https://github.com/quic/aimet/commit/ae7d5ef19b3071dde90d5564ae81907c95f17101
.. _a8ac6f4: https://github.com/quic/aimet/commit/a8ac6f412b2266fb4043a4d18db7475ab6289140
.. _135a0af: https://github.com/quic/aimet/commit/135a0afca6250556a363291c1df8c17682645b16
.. _624ba30: https://github.com/quic/aimet/commit/624ba30ac50a36159ea21b90293ddfa25d79fe6b
.. _1e8eceb: https://github.com/quic/aimet/commit/1e8eceb320189fd37dc7f1b75f86a47c9ab8d52c


2.24.0
======

* Bug fixes and Improvements
    * ONNX
        * Add Windows ARM64 wheel build/test support, distribute Windows ARM64 wheel on GitHub releases (`1390b96`_)
        * Add transpose MatMul support in Sequential MSE (`ff7a284`_)

    * Torch
        * Expose block-level AdaScale API (`72246db`_)
        * Improve numerical stability of zero point shifting ([-1.5, -.5, .5, 1.5]) implementation (`489f7df`_)
        * Fix :func:`replace_lora_layers_with_quantizable_layers` to inherit train/eval flag (`af5a82d`_)
        * Fix SpinQuant evaluation by untying lm_head and embed_tokens prior to loading the state_dict. (`47f574d`_)
        * Experimental - Implement Progressive Gradient Scaling (PGS) support for Triton-based quantization kernels (`b58b00b`_)

    * Common
        * Fix TFEnhanced incorrectly producing negative scales when encountering empty (size‑0) inputs (`ea4af6a`_)
        * Unpin numpy dependency (`8a999a1`_)
        * Add an alias for referencing the eNPU configuration file (`b79611c`_)


.. _8a999a1: https://github.com/quic/aimet/commit/8a999a1cf8c42e0a4cdfa9db52e3ea959aa155a4
.. _af5a82d: https://github.com/quic/aimet/commit/af5a82ddacff1f5f01a9e2d3f5450677a359049c
.. _ea4af6a: https://github.com/quic/aimet/commit/ea4af6a99ea61029bfaae48f041299cdbccb2508
.. _489f7df: https://github.com/quic/aimet/commit/489f7df64c73260e83704fd09f0cab63f98f425d
.. _72246db: https://github.com/quic/aimet/commit/72246dbb168559972fd5c9fc3a02d07b70a732d8
.. _47f574d: https://github.com/quic/aimet/commit/47f574de4d2f51d5a6189a3f3de18726d5abcf5a
.. _b58b00b: https://github.com/quic/aimet/commit/b58b00bd35c905ad1499e37f97dea7b0c264dbdd
.. _ff7a284: https://github.com/quic/aimet/commit/ff7a2841935620615ccd52c001886f8ae8e29705
.. _b79611c: https://github.com/quic/aimet/commit/b79611c86b52054a286adca257a2ba6d131ed5b0
.. _1390b96: https://github.com/quic/aimet/commit/1390b96ec4bd014055c4a9c6b116eb9f06a85afd


2.23.0
======

* Bug fixes and Improvements
    * ONNX
        * Disable per-channel quantization for ConvTranspose ops (`9395e32`_)
        * New top level API for configuring parameter quantization type (`a1c197d`_)

    * Torch
        * Enable Torch Dynamo ONNX export (`59e0125`_)

    * Common
        * Enable per-channel matmul quantization in config files (`7137849`_)
        * LLM quantization recipes in docs (`6561f0e`_)
        * Fix CUDA discrepancies against CPU wheel (`01e7422`_)

.. _9395e32: https://github.com/quic/aimet/commit/9395e3243fdfdc245540ef8a39b3058b27404baa
.. _a1c197d: https://github.com/quic/aimet/commit/a1c197d06c72793a4d903fdecdd214389c3f4b3e
.. _59e0125: https://github.com/quic/aimet/commit/59e01254fefdc99394b984ef4c9a8bb907ac4a3c
.. _7137849: https://github.com/quic/aimet/commit/71378496f6a966904938b600d562e7d3eb402b90
.. _6561f0e: https://github.qualcomm.com/qualcomm-ai/aimet/commit/6561f0e06e339596201ea93963f75f187094e79c
.. _01e7422: https://github.com/quic/aimet/commit/01e7422dd0da63f8fcf338dcc1636cf6a0294823


2.22.0
======

* Bug fixes and Improvements
    * ONNX
        * Allow loading 2.0.0 encoding format to sim (`e8cb098`_)
        * Fix Cast unpacking error (`6761a19`_)
        * Enable exporting non-LPBQ encodings with zero_point shift (`7b3cc4c`_)
        * Implement aimet-onnx LPBQEncoding (`5ad7ea6`_)

    * Common
        * Support exporting 1x1 Conv LPBQ to ONNX QDQ (`58ce71d`_)

.. _e8cb098: https://github.com/quic/aimet/commit/e8cb098c19a786ab3cdae693f6cc651d348bc824
.. _6761a19: https://github.com/quic/aimet/commit/6761a1920718cf161a8c5f42614cd5b25ac0be70
.. _7b3cc4c: https://github.com/quic/aimet/commit/7b3cc4c927991e0cdd98df5fa69c4c529147b787
.. _5ad7ea6: https://github.com/quic/aimet/commit/5ad7ea66056a3e243e17f14e2f3daad3c1c517ac
.. _58ce71d: https://github.com/quic/aimet/commit/58ce71de197e681d364a958da9b2d2389dfbf501


2.21.0
======

* Bug fixes and Improvements
    * ONNX
        * Fix IndexError when Conv or Linear layers are reused in the model (`65c4b3b`_)
        * Add optional argument `export_int32_bias` to aimet-onnx export (`3b8e0f0`_)
        * Unpin PyTorch version in aimet-onnx (`d99b6c4`_)
        * Align NaN handling with ORT CPU Execution Provider (`e4c49eb`_)
        * Fix quantization axis handling for transposed MatMul operations (`6ca06d6`_)
    
    * PyTorch
        * Fix quantization logic to enable input quantizers for layers following ignored layers (`80fb4fe`_)

.. _65c4b3b: https://github.com/quic/aimet/commit/65c4b3b9f58cbeddfb41f79db42e79a80d6427df
.. _3b8e0f0: https://github.com/quic/aimet/commit/3b8e0f03f7701003377a65cf5d782143f627db2b
.. _d99b6c4: https://github.com/quic/aimet/commit/d99b6c4a9e54f0f05e2ac2b72487058d7c19fcdc
.. _e4c49eb: https://github.com/quic/aimet/commit/e4c49eb6c5e1b5c666221313981b8fab44f19ea8
.. _6ca06d6: https://github.com/quic/aimet/commit/6ca06d60b82b59bf2f30e90663d8cfdb3777da91
.. _80fb4fe: https://github.com/quic/aimet/commit/80fb4fef91bf81bb5a7356645295168ca1ccef88


2.20.0
======

* Bug fixes and Improvements
    * Common
        * Update supported python version to >=3.10 (`2bc8c94`_)
        * Repackage aimet_common as alias to aimet_onnx.common or aimet_torch.common (`074e85f`_)
        * Remove Pad op from data movement ops (`21cddb6`_)

    * ONNX
        * Export data movement op output encoding in sim.export by default (`550c029`_)
        * Assign generic node names if node name is missing or duplicate (`273dd82`_)
        * Add PyTorch Pad modules to nn.Module -> onnx op mapping (`7e5342b`_)
        * Add LSTM cell state int32 quantization mechanism for LPAI (`3a8659b`_)
        * Support stacked RNN/GRU/LSTM (`552ad83`_)
        * Make exclude/include node argument naming consistent (`ec22d86`_)
        * Implement LPBQ support in aimet-onnx SeqMSE (`495567f`_)
        * Add support for dilation, grouping, stride to Quantized Conv (`f94f3e2`_)
        * Remove block type from adascale config  (`b55b058`_)
        * Skip tying concat encoding if input has multiple consumers (`3136828`_)
        * Tie quantizers upstream first and downstream later (`59aac3e`_)
        * Fix ValidationError in LazyExtractor when external files are missing or inconsistent (`a8f32fc`_)
        * Align torch and onnx GenAI recipes (`7d4659d`_)

    * Torch
        * Use separate input quantizer for each concat input (`755c54a`_)
        * Add predict and fallback later approach for batched matmul in aimet-torch seq mse (`8874173`_)
        * Refactored MMP to not use rounding mode (`fd7e40d`_)
        * Use tuple for strided slice indexing (`4ddbd66`_)
        * Fix symmetry bug in _from_qnn_encoding_dict (`35602ea`_)
        * Align onnx 1.0.0 BQ encoding export ordering with QAIRT expectation (`0182b7a`_)

.. _21cddb6: https://github.com/quic/aimet/commit/21cddb68889e3d01843de8744e8493f6daa3db28
.. _550c029: https://github.com/quic/aimet/commit/550c0291d074626e555db6b6a5fa3239f333787e
.. _273dd82: https://github.com/quic/aimet/commit/273dd8202489205ff39d20d52a227053ee6cd2e6
.. _7e5342b: https://github.com/quic/aimet/commit/7e5342bcf60e6e51467ebf791ab96ac9eadbca65
.. _3a8659b: https://github.com/quic/aimet/commit/3a8659b3b97b7d923d2f32b44b55fae48b7f6ac2
.. _755c54a: https://github.com/quic/aimet/commit/755c54ad7f716f39f7088f995f37f25deedb3520
.. _552ad83: https://github.com/quic/aimet/commit/552ad83f861502f99765358d83ccda252f8a40fa
.. _ec22d86: https://github.com/quic/aimet/commit/ec22d8682ba076eb6e09b29a96cd3aab827e8e2b
.. _495567f: https://github.com/quic/aimet/commit/495567f3bf05d447e76580a1b30aa5aa86ce6c0b
.. _35602ea: https://github.com/quic/aimet/commit/35602eac649576812078a2d93b73d8f319dd25bf
.. _f94f3e2: https://github.com/quic/aimet/commit/f94f3e22d89f71d6dbced1cfa1393dc83c19f1b4
.. _b55b058: https://github.com/quic/aimet/commit/b55b058445fe552986e0ddae5f837570aebae69c
.. _074e85f: https://github.com/quic/aimet/commit/074e85fd15b92c2b65b03059374a5272f07bdeb5
.. _3136828: https://github.com/quic/aimet/commit/3136828f051ff4f1032b9fc8fec2c31e979dc67c
.. _59aac3e: https://github.com/quic/aimet/commit/59aac3e4a65c295d6f25d6fb5cb53b7c0441774f
.. _0182b7a: https://github.com/quic/aimet/commit/0182b7aea8a62647269893a9291fd83cd8959f2c
.. _8874173: https://github.com/quic/aimet/commit/887417350d00f8a505c6c3c1754868cfcdd552f7
.. _a8f32fc: https://github.com/quic/aimet/commit/a8f32fce6188564a4cec2db63085d40b26056534
.. _fd7e40d: https://github.com/quic/aimet/commit/fd7e40dbf43660bfb733ff50fe72ea889f2d8c09
.. _7d4659d: https://github.com/quic/aimet/commit/7d4659dae2541e6a72c6cea5cba3f6dc676601eb
.. _4ddbd66: https://github.com/quic/aimet/commit/4ddbd66aac7e75183cfa219a6da60bb576f0e0e8
.. _2bc8c94: https://github.com/quic/aimet/commit/2bc8c94fcced5ceff790f2c8a0b8347ee42f0be1


2.19.0
======

* New Features

* Bug fixes and Improvements
    * ONNX
        * Make LiteMP API percentage float (`69f96ff`_)
        * Set layernorm int16 weight to symmetric by default (`8560e13`_)
        * Automatically insert data movement op output qdq during to_onnx_qdq (`15c8b9b`_)
        * Create LazyExtractor to handle external data for onnx Extractor utils (`104e7e8`_)
        * Tie input/output encodings across maximum Concat subgraph (`832ea91`_)
        * Tie hidden state quantizers of RNN/GRU/LSTM (`c18fd05`_)

    * Torch
        * Fix histogram observer rebinning logic (`2c88364`_)
        * Fix connectedgraph input ordering for non-trivial layer types (`2b7b548`_)

    * Common
        * Disable per-channel quantization of RNN/GRU/LSTM for all HTP backends (`df8b875`_)

.. _df8b875: https://github.com/quic/aimet/commit/df8b87516dc894baf768377a037944fcdd60f0f6
.. _69f96ff: https://github.com/quic/aimet/commit/69f96ff1af7c69603f325dbdf2a89cf6b22d57a7
.. _8560e13: https://github.com/quic/aimet/commit/8560e136914216a1003bb2888b827daaae490991
.. _15c8b9b: https://github.com/quic/aimet/commit/15c8b9b2672671fd3cff1c92267460340c40db48
.. _2c88364: https://github.com/quic/aimet/commit/2c88364f75f893ec1f798afd7530436685aa5b7a
.. _104e7e8: https://github.com/quic/aimet/commit/104e7e8284393ae383c31e0c2045ab06674fac35
.. _832ea91: https://github.com/quic/aimet/commit/832ea917f48fd1fd70096827c3fca647d7621e2d
.. _c18fd05: https://github.com/quic/aimet/commit/c18fd056674614b44c443679a5620ed5223303d0
.. _2b7b548: https://github.com/quic/aimet/commit/2b7b54816b0f35db5bf09c107075d7b7285f871c


2.18.0
======

* New Features
    * Torch
        * Promoted aimet_torch.onnx.export and QuantizationSimModel.onnx.export as production APIs (`99160d2`_, `e026fd1`_)
        * Added utility functions to exclude some or all unknown nn.Modules from quantization (`5a419f3`_, `501eebd`_)

* Bug fixes and Improvements
    * ONNX
        * Fixed supergroup misidentification bug upon MatMul-MatMul-Add sequence (`ab63866`_)

    * Torch
        * Made compatible with PyTorch 1.13 (`47fae94`_)
        * Made compatible with PyTorch 2.9 (`283ecc1`_)

    * Common
        * Set priority among supergroups (`6676a6c`_)

.. _99160d2: https://github.com/quic/aimet/commit/99160d2
.. _e026fd1: https://github.com/quic/aimet/commit/e026fd1
.. _ab63866: https://github.com/quic/aimet/commit/ab63866
.. _47fae94: https://github.com/quic/aimet/commit/47fae94
.. _283ecc1: https://github.com/quic/aimet/commit/283ecc1
.. _5a419f3: https://github.com/quic/aimet/commit/5a419f3
.. _501eebd: https://github.com/quic/aimet/commit/501eebd
.. _6676a6c: https://github.com/quic/aimet/commit/6676a6c


2.17.0
======

* Bug fixes and Improvements
    * ONNX
        * Optimize SeqMSE latency and CPU memory usage (`434ac6b`_)
        * Support excluding nodes from SeqMSE optimization (`6a37239`_)
        * Support exporting large models (> 2GB) to ONNX QDQ (`b1dafe6`_, `1bf8b82`_)
        * Support exporting float16 ONNX models to ONNX QDQ (`66ccb45`_)
        * Allow disabling MatMul-Add supergroup via config file (`e49660c`_)
        * Fix bug where on-disk tensor data is deleted before InferenceSession (`d57a934`_)

    * Torch
        * Fix sim.export bug when using Python >= 3.12 (`ee949a2`_)
        * Allow export for back-to-back quantizers which share the same encodings (`28a7382`_)
        * Fix numerical issue in FPTQuant (`f0bc6c9`_)

    * Common
        * Remove Conv-Relu supergroup from HTP < V73 config files (`19e5a4e`_)
        * Fix LayerNorm and InstanceNorm weight symmetry in HTP < V73 config files (`eb1ac5c`_, `ce1ea63`_)

.. _434ac6b: https://github.com/quic/aimet/commit/434ac6b8ac5347935a0e3902b2e37e0c49dfe242
.. _b1dafe6: https://github.com/quic/aimet/commit/b1dafe6fa5173fc2247802313224e40013b68822
.. _19e5a4e: https://github.com/quic/aimet/commit/19e5a4ecb3a1e58bcf71f455d7d3855bcc5d86f2
.. _28a7382: https://github.com/quic/aimet/commit/28a73829aee6d77991c100ea4ed9fdeab5fc009c
.. _eb1ac5c: https://github.com/quic/aimet/commit/eb1ac5c36e7dd198d43d4aa450b5933cf94755b4
.. _ce1ea63: https://github.com/quic/aimet/commit/ce1ea63845d0f7b6ddef66ebcc70922cbcad511b
.. _1bf8b82: https://github.com/quic/aimet/commit/1bf8b82fe6c23846d9fe615773797a1df7fb5545
.. _6a37239: https://github.com/quic/aimet/commit/6a37239ffbaf0187ead2ddee205960c403817e17
.. _e49660c: https://github.com/quic/aimet/commit/e49660c87fb3097f247e680c93ddc1c1f62c8871
.. _d57a934: https://github.com/quic/aimet/commit/d57a934cb25539f7f2809c3f5ef8b44e384ef051
.. _ee949a2: https://github.com/quic/aimet/commit/ee949a2d1ad8a64c0de8bfbf34339251f0804294
.. _66ccb45: https://github.com/quic/aimet/commit/66ccb45343abd6a475816b395a1658b6998df202
.. _f0bc6c9: https://github.com/quic/aimet/commit/f0bc6c9b0ae4a45d517c8b96fc032022a07c6217


2.16.0
======

* New Features
    * ONNX
        * Experimental - Added Adascale, a post-training quantization technique (`5e23ceb`_)

* Bug fixes and Improvements
    * ONNX
        * Skip tying Concat input/output quantizers with conflicting encoding constraints (`b924107`_)
        * Small updates to FPT Quant for improved accuracy (`ba10947`_)
        * Implement partial encoding freezing mechanism in aimet-onnx (`658ec3c`_)
        * Add Relu partial encoding constraints to HTP config files (`dc8d978`_)
        * Clear encoding analyzer stats after computing param encodings (`3d4725f`_)
        * Remove wasted computation/memory in FPTQuant local optimizer (`59350af`_)

    * Torch
        * Allow boolean type casting of QuantizedTensors (`7d63e66`_)
        * Implement partial encoding freezing mechanism in aimet-torch (`1b99a39`_)
        * Improve scale post-processing to prevent scale freezing during QAT (`6fe56b0`_)

.. _5e23ceb: https://github.com/quic/aimet/commit/5e23cebea551c074f7a380ef2f385fd95433bb53
.. _b924107: https://github.com/quic/aimet/commit/b9241073256c4a455426451efbc1f3d0672e37b2
.. _ba10947: https://github.com/quic/aimet/commit/ba10947bdbdecdf2980f076560453991c3888e77
.. _658ec3c: https://github.com/quic/aimet/commit/658ec3c20be379b582321171e28f92e8fab1102b
.. _1b99a39: https://github.com/quic/aimet/commit/1b99a39b6c19f6b7fc77c871b5dc232981e6eac9
.. _dc8d978: https://github.com/quic/aimet/commit/dc8d978f672e5a93ecb5c8de64017ccaf949d2bf
.. _3d4725f: https://github.com/quic/aimet/commit/3d4725fc172bffeadd87ee993b7a30e5d51691b2
.. _59350af: https://github.com/quic/aimet/commit/59350afb881678dc0313a7445bb2e61d5b14328b
.. _7d63e66: https://github.com/quic/aimet/commit/7d63e6660050399479c804b474b9cb87c7991fce
.. _6fe56b0: https://github.com/quic/aimet/commit/6fe56b0d94b1de7f659f4e1d08be5847e4313a09


2.15.0
======

* Bug fixes and Improvements
    * ONNX
        * Throws an error on `bfloat16` models (`5181860`_)
        * Added docs and examples for LiteMP (`3d5e0dd`_)
        * Export to QDQ ONNX with pre-quantized constants (`a97354f`_)

    * PyTorch
        * Fix multiple dispatch issue when torch function is called in nested context manager (`6216ca0`_)

    * Keras
        * 2.14.0 is the last release of aimet-tf (`087e9b1`_)

    * Common
        * Added PSNR metrics (`14c8e81`_)

.. _14c8e81: https://github.com/quic/aimet/commit/14c8e81a8309504d564799801cfde102618efc8a
.. _087e9b1: https://github.com/quic/aimet/commit/087e9b1ddefea24f21580256f8b20606b931d74c
.. _5181860: https://github.com/quic/aimet/commit/5181860434e75e67cb161e3c0e9135df10b04507
.. _3d5e0dd: https://github.com/quic/aimet/commit/3d5e0ddd5dab7b0fec2aea5fc20bf21955e918d8
.. _a97354f: https://github.com/quic/aimet/commit/a97354f0f6b72d790a8c03fea1338d7cb5c6aa64
.. _6216ca0: https://github.com/quic/aimet/commit/6216ca03e0d15ce47d5c4273f59d72d6d5ad46dc

2.14.0
======
* New Feature
    * ONNX
        * Add support for FP16 in :class:`QuantizationSimModel` (`2494d90`_)

* Bug fixes and Improvements
    * ONNX
        * Add sequential MSE support for ``onnx >= 1.18.0``. (`754d030`_)
        * Improve histogram granularity during TFE calibration (`91109af`_)
        * Improve runtime for :class:`QuantizationSimModel` creation for large models like LLMs (`f7e700f`_)
        * Improve runtime for setting quantizers in a :class:`QuantizationSimModel` for use cases like tying KV Cache input and output quantizers. (`c0bdb46`_)
        * Add a check for None values in the ``group`` attribute of ``Conv`` layers and fix improper handling of None ``group`` attribute in ``ConvTranspose`` within :func:`fold_all_batch_norms_to_weight` (`374e8db`_)

    * PyTorch
        * Address QAT convergence issue: Add a fix for cases where ``quantizer.min`` becomes equal to ``quantizer.max`` during training, leading to NaN values (`51f8990`_)

    * Keras
        * Fix accuracy drop issue for GPU wheel by excluding ``libpython*.so*`` from the aimet wheel packages (`22cac5c`_)

    * Common
        * Remove ``Conv3d``, ``Conv3dTranspose``, and ``DepthwiseConv`` ops followed by activation from the supergroup until HTP support is available. (`05f6810`_)
        * Fix color theme issue in documentation causing code snippets to render incorrectly (`2c64eac`_)

.. _2494d90: https://github.com/quic/aimet/commit/2494d9048241b0b84e388f69d3c202b2e45285ee
.. _754d030: https://github.com/quic/aimet/commit/754d030838cb676b6f6b08f6e8cc91838bcf8be9
.. _91109af: https://github.com/quic/aimet/commit/91109aff222711a4ce0528e8b2dc2eb2c63cf18d
.. _f7e700f: https://github.com/quic/aimet/commit/f7e700f98973bdf39907482d3092349ceae2047e
.. _c0bdb46: https://github.com/quic/aimet/commit/c0bdb466f0e26b5757f473308af0c41c47a50fb1
.. _374e8db: https://github.com/quic/aimet/commit/374e8dbc344f40cf356d3bae2ede521ad5341622
.. _51f8990: https://github.com/quic/aimet/commit/51f899080ea7260dc2591023a868b30a88ff5fa4
.. _05f6810: https://github.com/quic/aimet/commit/05f6810a5a0e1eef2cfaf6525c572fc02b0174bc
.. _2c64eac: https://github.com/quic/aimet/commit/2c64eac1801724500028e926585c558b936f12ae
.. _22cac5c: https://github.com/quic/aimet/commit/22cac5c7cb69e8ea66f710ab1b967c6f9b44f0f5

2.13.0
======
* Bug fixes and Improvements
    * ONNX
        * Adjust weight scale for int32 bias overflow in W16A16 quantization (`f39c0bf`_)
        * AutoQuant: Remove deprecated feature (`414cdde`_)
        * Support exporting large models in aimet-onnx (`0fe6701`_)
        * AdaRound: Delete deprecated top-level API. (`bfba557`_)
        * AdaRound: Skip optimization if no input to layer (`18dfedc`_)
    
    * PyTorch
        * Enable save_model_as_external_data for sim.onnx.export (`107b339`_)

* Known Issues
    * Keras
        * Accuracy drop observed with AIMET Keras for certain models. Fix is planned for the next release.

.. _f39c0bf: https://github.com/quic/aimet/commit/f39c0bf3e3fc1e21527f9a99c8b8d42e1ebdd277
.. _414cdde: https://github.com/quic/aimet/commit/414cddec3d7317cf7251c0134e6f4b3a15bbcb1e
.. _0fe6701: https://github.com/quic/aimet/commit/0fe67010fd06641c2ef9696f7e36ce87e58456fb
.. _bfba557: https://github.com/quic/aimet/commit/bfba5573007b41935b217e79ada931598b44da19
.. _18dfedc: https://github.com/quic/aimet/commit/18dfedcc3fa0eecfec57b7352c1b467b8c826650
.. _107b339: https://github.com/quic/aimet/commit/107b339fe9cb9ee105e7aa97257751d44d878f34


2.12.0
======
* Bug fixes and Improvements
    * Common
        * Remove data movement ops from config (`ae02aa8`_)

    * ONNX
        * Exclude bias from quantization when weights are not quantized (`62f5879`_)
        * AdaRound: Fix prelu failing in CUDA model (`b2350b2`_)

    * PyTorch
        * Wrap aimet_torch.onnx.export with torch.no_grad (`b73bb71`_)

* Known Issues
    * Keras
        * Accuracy drop observed with AIMET Keras for certain models. Fix is planned for the next release.

.. _62f5879: https://github.com/quic/aimet/commit/62f587909a91d50fe60ae8d453d8e557b6ab67d5
.. _b73bb71: https://github.com/quic/aimet/commit/b73bb7168532469cf0b93e886508fd00bb071fc6
.. _ae02aa8: https://github.com/quic/aimet/commit/ae02aa852aff4f8c9dec651dd13cc7d177904642
.. _b2350b2: https://github.com/quic/aimet/commit/b2350b2f87247a4d879fc83496c7e2d042569917


2.11.0
======
* New Feature
    * PyTorch
        * SpinQuant (experimental) - implement SpinQuant PTQ technique (https://arxiv.org/pdf/2405.16406) for Llama, Qwen2, and Mistral families (R1 rotation w/o optimization) (`7364b37`_)
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
