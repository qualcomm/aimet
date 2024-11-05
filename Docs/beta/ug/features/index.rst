.. _features-index:

################
Optimization Techniques
################

.. toctree::
    :hidden:

    Quantization aware training (QAT)
    Automatic quantization (AutoQuant) <autoquant>
    Adaptive rounding (Adaround) <adaround>
    Cross-layer equalization (CLE) <cle>
    Batch norm re-estimation (BN) <bn>
    Quantization analyzer (QuantAnalyzer) <quant_analyzer>
    Visualization <visualization>
    Weight singular value decomposition (Weight SVD) <weight_svd>
    Spatial singular value decomposition (Spatial SVD) <spatial_svd>
    Channel pruning (CP) <cp>


Quantization aware training (QAT)
==================

.. grid:: 1

    .. grid-item-card::   Quantization aware training (QAT)
        :link: feature-qat
        :link-type: ref

        QAT fine-tunes the model parameters in the presence of quantization noise.


Automatic quantization (AutoQuant)
==================

.. grid:: 1

    .. grid-item-card::  Automatic quantization (AutoQuant)
        :link: feature-autoquant
        :link-type: ref

        AutoQuant analyzes the model, determines the best sequence of AIMET post-training quantization techniques, and applies these techniques.




Adaptive rounding (Adaround)
==================

.. grid:: 1

    .. grid-item-card::  Adaptive rounding (Adaround)
        :link: feature-adaround
        :link-type: ref

        AdaRound uses training data to improve accuracy over naïve rounding.


Cross-layer equalization (CLE)
==================

.. grid:: 1

    .. grid-item-card::  Cross-layer equalization (CLE)
        :link: feature-cle
        :link-type: ref

        CLE scales the parameter ranges across different channels to increase the range for layers with low range and reduce range for layers with high range, enabling the same quantizaion parameters to be used across all channels.


Batch norm re-estimation (BN)
==================

.. grid:: 1

    .. grid-item-card::  Batch norm re-estimation (BN)
        :link: feature-bn
        :link-type: ref

        BN re-estimated statistics are used to adjust the quantization scale parameters of preceeding Convolution or Linear layers, effectively folding the BN layers.


Quantization analyzer (QuantAnalyzer)
==================

.. grid:: 1

    .. grid-item-card::  Quantization analyzer (QuantAnalzer)
        :link: feature-quant-analyzer
        :link-type: ref

        QuantAnalyzer automatically identify sensitive areas and hotspots in the model.


Visualization
==================

.. grid:: 1

    .. grid-item-card::  Visualization
        :link: feature-visualization
        :link-type: ref

        QuantAnalyzer automatically identify sensitive areas and hotspots in the model.


Weight singular value decomposition (Weight SVD)
==================

.. grid:: 1

    .. grid-item-card::  Weight singular value decomposition (Weight SVD)
        :link: feature-weight-svd
        :link-type: ref

        Weight SVD decomposes one large MAC or memory layer into two smaller layers.


Spatial singular value decomposition (Spatial SVD)
==================

.. grid:: 1

    .. grid-item-card::  Spatial singular value decomposition (Spatial SVD)
        :link: feature-spatial-svd
        :link-type: ref

        Spatial SVD decomposes one large convolution (Conv) MAC or memory layer into two smaller layers.


Channel pruning (CP)
==================

.. grid:: 1

    .. grid-item-card::  Channel pruning (CP)
        :link: feature-cp
        :link-type: ref

        CP removes less-important input channels from 2D convolution layers.
