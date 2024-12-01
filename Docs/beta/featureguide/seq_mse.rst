.. _featureguide-seq-mse:

##############
Sequential MSE
##############

Context
=======

Sequential MSE (SeqMSE) is a method that searches for optimal quantization encodings per operation
(i.e. per layer) such that the difference between the original output activation and the
corresponding quantization-aware output activation is minimized.

Since SeqMSE is search-based rather than learning-based, it possesses several advantages:

- It requires only a small amount of calibration data,
- It approximates the global minimum without getting trapped in local minima, and
- It is robust to overfitting.

Workflow
========

tbd

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/seq_mse.rst
            :start-after: # start-after

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../apiref/onnx/seq_mse.rst
           :start-after: # start-after
