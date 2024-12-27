.. _apiref-torch-index:

###############
aimet_torch API
###############

.. toctree::
    :hidden:

    Migrate to aimet_torch 2 <migration_guide>
    aimet_torch.quantsim <quantsim>
    aimet_torch.adaround <adaround>
    aimet_torch.nn <nn>
    aimet_torch.quantization <quantization>
    aimet_torch.seq_mse <seq_mse>
    aimet_torch.quantsim.config_utils <lpbq>
    aimet_torch.manual_mixed_precision <mmp>
    aimet_torch.batch_norm_fold <bnf>
    aimet_torch.cross_layer_equalization <cle>
    aimet_torch.model_preparer <model_preparer>
    aimet_torch.model_validator <model_validator>
    aimet_torch.mixed_precision <mp>
    aimet_torch.quant_analyzer <quant_analyzer>
    aimet_torch.autoquant <autoquant>
    aimet_torch.bn_reestimation <bn>
    aimet_torch.visualization_tools <interactive_visualization>
    aimet_torch.layer_output_utils <layer_output_generation>
    aimet_torch.peft <peft_lora>
    aimet_torch.compress <compress>

aimet_torch
===========

.. important::
   :mod:`aimet_torch` package is upgraded to :mod:`aimet_torch.v2` with more
   flexible, extensible, and PyTorch-friendly user interface!

   aimet_torch 2 is fully backward compatible with all the public APIs of aimet_torch 1.x.,
   please see :doc:`Migrate to aimet_torch 2 <migration_guide>`.

- :ref:`aimet_torch.quantsim <apiref-torch-quantsim>`
- :ref:`aimet_torch.nn <apiref-torch-nn>`
- :ref:`aimet_torch.quantization <apiref-torch-quantization>`
- :ref:`aimet_torch.adaround <apiref-torch-adaround>`
- :ref:`aimet_torch.seq_mse <apiref-torch-seq-mse>`
- :ref:`aimet_torch.quantsim.config_utils <apiref-torch-lpbq>`
- :ref:`aimet_torch.batch_norm_fold <apiref-torch-bnf>`
- :ref:`aimet_torch.cross_layer_equalization <apiref-torch-cle>`
- :ref:`aimet_torch.model_preparer <apiref-torch-model-preparer>`
- :ref:`aimet_torch.model_validator <apiref-torch-model-validator>`
- :ref:`aimet_torch.mixed_precision <api-torch-mp>`
- :ref:`aimet_torch.quant_analyzer <apiref-torch-quant-analyzer>`
- :ref:`aimet_torch.autoquant <apiref-torch-autoquant>`
- :ref:`aimet_torch.bn_reestimation <apiref-torch-bn>`
- :ref:`aimet_torch.visualization_tools <api-torch-interactive-visualization>`
- :ref:`aimet_torch.layer_output_utils <apiref-torch-layer-output-generation>`
- :ref:`aimet_torch.peft <apiref-torch-peft-lora>`
- :ref:`aimet_torch.compress <apiref-torch-compress>`

aimet_torch.v1
==============

If you still prefer to use aimet_torch 1.x, your imports should originate from the :mod:`aimet_torch.v1`
namespace.

.. toctree::
    :hidden:

    aimet_torch.v1.quantsim <v1/quantsim>
    aimet_torch.v1.adaround <v1/adaround>
    aimet_torch.v1.seq_mse <v1/seq_mse>
    aimet_torch.v1.quant_analyzer <v1/quant_analyzer>
    aimet_torch.v1.autoquant <v1/autoquant>
    aimet_torch.v1.amp <v1/amp>

- :ref:`aimet_torch.v1.quantsim <apiref-torch-v1-quantsim>`
- :ref:`aimet_torch.v1.adaround <apiref-torch-v1-adaround>`
- :ref:`aimet_torch.v1.seq_mse <apiref-torch-v1-seq-mse>`
- :ref:`aimet_torch.v1.quant_analyzer <apiref-torch-v1-quant-analyzer>`
- :ref:`aimet_torch.v1.autoquant <apiref-torch-v1-autoquant>`
- :ref:`aimet_torch.v1.amp <apiref-torch-v1-amp>`
