.. _apiref-torch-peft-lora:

################
aimet_torch.peft
################

This document provides steps for integrating LoRA adapters with AIMET Quantization flow. LoRA adapters
are used to enhance the efficiency of fine-tuning large models with reduced memory usage. We will use
`PEFT <https://huggingface.co/docs/peft/main/en/package_reference/peft_model>`_ library
from HuggingFace to instantiate our model and add adapters to it.

By integrating adapters with AIMET quantization, we can perform similar functionalities as present in PEFT,
for example, changing adapter weights, enabling and disabling adapters. Along with this, we can tweak the quantization
parameters for the adapters alone to get good quantization accuracy.

Terminology
===========

1. Base Model: Model without adapters
2. Concurrency: Multiple adapters can be added to a model. 1 or more adapters which are active/enabled at the same time
make a concurrency.


User flow
=========

This flow encaptures how Lora adapters can be added to a model and quantized using AIMET APIs.
A user can start with a prepared model (using model preparer APIs) or a non-prepared model, for ex, HuggingFace model

Step 0: (Optional, required for taking artifacts to target) Perform Base Model Quantization and export

Step 1: Create a PEFT model with the adapter. Use PEFT APIs from HuggingFace to create a PEFT model. For simplicity, we
are adding the same adapter twice, a user can chose to add adapters with different lora configs.

    >>> from peft import LoraConfig, get_peft_model, PeftMixedModel
    >>> lora_config = LoraConfig(
    >>>    lora_alpha=16,
    >>>    lora_dropout=0.1,
    >>>    r=4,
    >>>    bias="none",
    >>>    target_modules=["linear"])
    >>> peft_model = PeftMixedModel(model, lora_config) ## First adapter gets added with name 'default'
    >>> peft_model.add_adapter('default_new', lora_config) ## Adding second adapter
    >>> peft_model.set_adapter(['default', 'default_new'])

Step 2: To make the model quantization friendly, we replace the PEFT lora layers with quantizable PEFT lora layers

    >>> from aimet_torch.peft import replace_lora_layers_with_quantizable_layers
    >>> replace_lora_layers_with_quantizable_layers(model)

Step 3: Create a QuantSim object & calibrate (steps are not shown below, please refer to quantsim docs for reference)

Step 4: Export

Here we do not show steps for how to compute the encoding. Please refer to Quantization simulation documentation

    >>> sim.export(tmpdir, 'model', dummy_input=dummy_inputs, export_model=False, filename_prefix_encodings='adapter1')

Step 5: For another adapter concurrency we repeat steps 1 to 4

API
===
**The following API can be used to replace PEFT lora layers definition with AIMET lora layers definition**

.. autofunction:: aimet_torch.peft.replace_lora_layers_with_quantizable_layers
