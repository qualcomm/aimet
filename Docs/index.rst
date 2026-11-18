:template: landing.html
:description: AIMET quantizes trained ONNX and PyTorch models to reduce compute load and memory footprint on edge devices, with minimal accuracy loss.

.. include:: abbreviation.txt

.. title:: AIMET Documentation

.. role:: aimet-accent

.. _top-index:

###########################################################
Quantize trained models. :aimet-accent:`Keep the accuracy.`
###########################################################

.. toctree::
   :hidden:
   :includehidden:
   :maxdepth: 2

   Overview <overview/index>
   Tutorials <tutorials/index>
   Techniques <techniques/index>
   PTQ Techniques <ptq_techniques/index>
   API Reference <apiref/index>
   Release Notes <release_notes>
   External Resources <external/index>
   Glossary <glossary>

.. The landing page layout (two-column hero, card grid) is styled in
   _static/aimet-lumina.css under "Landing page"; the hero animation is
   _static/aimet-landing.js.

.. container:: aimet-hero

   .. container:: aimet-hero-text

      .. rst-class:: aimet-eyebrow

      AI Model Efficiency Toolkit · v\ |version|

      .. rst-class:: aimet-lead

      AIMET reduces compute load and memory footprint so deep learning models run faster on
      edge devices like phones and laptops. Post-training and fine-tuning techniques keep
      accuracy loss to a minimum.

      .. container:: aimet-ctas

         :doc:`Quick start <overview/install/quick-start>`
         :doc:`What is AIMET? <overview/index>`

      .. container:: aimet-frameworks

         Works with ``ONNX`` ``PyTorch``

   .. raw:: html

      <figure class="aimet-quant-viz" aria-label="Animation: 160 MobileNetV2 weights snapping from floating point onto a 16-level int4 quantization grid">
        <div class="aimet-viz-top">
          <span>layer <b>features.15.conv.0.0</b> · ch 301</span>
          <span class="aimet-viz-state" data-aimet-viz="state">fp32</span>
        </div>
        <canvas></canvas>
        <div class="aimet-viz-bottom">
          <span>bitwidth <b data-aimet-viz="bw">32</b></span>
          <span>levels <b data-aimet-viz="levels">&mdash;</b></span>
          <span>scale <b data-aimet-viz="scale">&mdash;</b></span>
          <span>offset <b data-aimet-viz="offset">&mdash;</b></span>
        </div>
      </figure>

.. container:: aimet-docs-head

   .. rubric:: Documentation

   Supports models from the ONNX and PyTorch frameworks.

.. grid:: 1 2 3 3
   :class-container: aimet-cards

   .. grid-item-card:: Overview
      :link: overview/index
      :link-type: doc

      .. rst-class:: aimet-tag

      Start

      What AIMET is and how you can use it.

   .. grid-item-card:: Quick Start Guide
      :link: overview/install/quick-start
      :link-type: doc

      .. rst-class:: aimet-tag

      Start

      Install AIMET and quantize your first model.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      .. rst-class:: aimet-tag

      Learn

      Curated walkthroughs of quantization workflows and other techniques.

   .. grid-item-card:: Example Notebooks
      :link: tutorials/notebooks
      :link-type: doc

      .. rst-class:: aimet-tag

      Learn

      End-to-end quantization and compression examples, downloadable as Jupyter notebooks.

   .. grid-item-card:: Quantization User Guide
      :link: techniques/index
      :link-type: doc

      .. rst-class:: aimet-tag

      Apply

      A high-level explanation of how to use AIMET to optimize a model.

   .. grid-item-card:: PTQ Techniques
      :link: ptq_techniques/index
      :link-type: doc

      .. rst-class:: aimet-tag

      Apply

      How to apply each post-training quantization technique.

   .. grid-item-card:: API Reference
      :link: apiref/index
      :link-type: doc

      .. rst-class:: aimet-tag

      Reference

      A detailed look at the AIMET API.

   .. grid-item-card:: Release Notes
      :link: release_notes
      :link-type: doc

      .. rst-class:: aimet-tag

      Reference

      New features and fixes in this release.

   .. grid-item-card:: Glossary
      :link: glossary
      :link-type: doc

      .. rst-class:: aimet-tag

      Reference

      Terms and acronyms used on this site.
