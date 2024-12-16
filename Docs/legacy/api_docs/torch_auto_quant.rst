:orphan:

.. _api-torch-auto-quant:

===========================
AIMET PyTorch AutoQuant API
===========================

User Guide Link
===============
To learn more about this technique, please see :ref:`AutoQuant<ug-auto-quant>`

Examples Notebook Link
======================
For an end-to-end notebook showing how to use PyTorch AutoQuant, please see :doc:`here<../Examples/torch/quantization/autoquant_v2>`.

Top-level API
=============

.. note::

    This module is also available in the experimental :mod:`aimet_torch.v2` namespace with the same top-level API. To
    learn more about the differences between :mod:`aimet_torch` and :mod:`aimet_torch.v2`, please visit the
    :ref:`QuantSim v2 Overview<ug-aimet-torch-v2-overview>`.

.. autoclass:: aimet_torch.auto_quant_v2.AutoQuant
    :members:
    :member-order: bysource

.. autoclass:: aimet_torch.v1.auto_quant.AutoQuant
    :members:

Code Examples
===============
.. literalinclude:: ../torch_code_examples/auto_quant_v2.py
    :language: python
    :lines: 40-
    :emphasize-lines: 74-78,86-

.. note::
   To use :class:`auto_quant.AutoQuant <aimet_torch.v1.auto_quant.AutoQuant>` (will be deprecated), apply the following code changes to step 5 and 7.

.. literalinclude:: ../torch_code_examples/auto_quant.py
    :language: python
    :lines: 113-
    :emphasize-lines: 1-4,12-
