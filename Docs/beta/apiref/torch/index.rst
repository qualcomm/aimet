.. _apiref-torch-index:

###############
aimet_torch API
###############

.. toctree::
    :hidden:

    aimet_torch.quantsim <quantsim>
    aimet_torch.adaround <adaround>
    aimet_torch.seq_mse <seq_mse>
    aimet_torch.batch_norm_fold <bnf>
    aimet_torch.cross_layer_equalization <cle>
    aimet_torch.quant_analyzer <quant_analyzer>
    aimet_torch.autoquant <autoquant>
    aimet_torch.bn_reestimation <bn>
    aimet_torch.layer_output_utils <layer_output_generation>
    aimet_torch.compress <compress>

aimet_torch
===========

.. important::

   :mod:`aimet_torch` package is planned to be upgraded to :mod:`aimet_torch.v2` with more flexible, extensible, and PyTorch-friendly user interface! In a future release, the core APIs of :mod:`aimet_torch` will be fully replaced with the equivalents in :mod:`aimet_torch.v2`.

AIMET quantization for PyTorch models provides the following functionality.

- :ref:`aimet_torch.quantsim <apiref-torch-quantsim>`
- :ref:`aimet_torch.adaround <apiref-torch-adaround>`
- :ref:`aimet_torch.seq_mse <apiref-torch-seq-mse>`
- :ref:`aimet_torch.batch_norm_fold <apiref-torch-bnf>`
- :ref:`aimet_torch.cross_layer_equalization <apiref-torch-cle>`
- :ref:`aimet_torch.quant_analyzer <apiref-torch-quant-analyzer>`
- :ref:`aimet_torch.autoquant <apiref-torch-autoquant>`
- :ref:`aimet_torch.bn_reestimation <apiref-torch-bn>`
- :ref:`aimet_torch.layer_output_utils <apiref-torch-layer-output-generation>`
- :ref:`aimet_torch.compress <apiref-torch-compress>`

aimet_torch.v2
==============

Introducing :mod:`aimet_torch.v2`, a future version of :mod:`aimet_torch` with more powerful
quantization features and PyTorch-friendly user interface!

What's New
----------

These are some of the powerful new features and interfaces supported in :mod:`aimet_torch.v2`

- Blockwise quantization (BQ)
- Low power blockwise quantization (LPBQ)
- Dispatching custom quantized kernels

Backwards Compatibility
-----------------------

Good news! :mod:`aimet_torch.v2` is carefully designed to be fully backwards-compatible with all
previous public APIs of :mod:`aimet_torch`. All you need is drop-in replacement of import statements
from :mod:`aimet_torch` to :mod:`aimet_torch.v2` as below!

.. code-block:: diff

   -from aimet_torch.quantsim import QuantizationSimModel
   +from aimet_torch.v2.quantsim import QuantizationSimModel

   -from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
   +from aimet_torch.v2.adaround import Adaround, AdaroundParameters

   -from aimet_torch.seq_mse import apply_seq_mse
   +from aimet_torch.v2.seq_mse import apply_seq_mse

   -from aimet_torch.quant_analyzer import QuantAnalyzer
   +from aimet_torch.v2.quant_analyzer import QuantAnalyzer

All the other APIs that didn't changed in or are orthogonal with :mod:`aimet_torch.v2` will be
still accessible via :mod:`aimet_torch` namespace as before.

.. toctree::
    :hidden:

    aimet_torch.v2 migration guide <v2/migration_guide>
    aimet_torch.v2.nn <v2/nn>
    aimet_torch.v2.quantization <v2/quantization>
    aimet_torch.v2.adaround <v2/adaround>
    aimet_torch.v2.seq_mse <v2/seq_mse>
    aimet_torch.v2.quantsim.config_utils.set_grouped_blockwise_quantization_for_weights <v2/lpbq>
    aimet_torch.v2.quant_analyzer <v2/quant_analyzer>
    aimet_torch.v2.visualization_tools <v2/interactive_visualization>

For more detailed information about how to migrate to :mod:`aimet_torch.v2`,
see :ref:`aimet_torch.v2 migration guide <torch-migration-guide>`

AIMET core APIs for PyTorch framework.

- :ref:`aimet_torch.v2.quantsim <apiref-torch-v2-quantsim>`
- :ref:`aimet_torch.v2.nn <apiref-torch-nn>`
- :ref:`aimet_torch.v2.quantization <apiref-torch-quantization>`
- :ref:`aimet_torch.v2.adaround <apiref-torch-v2-adaround>`
- :ref:`aimet_torch.v2.seq_mse <apiref-torch-v2-seq-mse>`
- :ref:`aimet_torch.v2.quantsim.config_utils.set_grouped_blockwise_quantization_for_weights <apiref-torch-v2-lpbq>`
- :ref:`aimet_torch.v2.quant_analyzer <apiref-torch-v2-quant-analyzer>`
- :ref:`aimet_torch.v2.visualization_tools <api-torch-v2-interactive-visualization>`
