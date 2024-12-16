.. _api-torch-peft-lora:

PEFT LoRA
==========

This document provides steps for integrating LoRA adapters with AIMET Quantization flow. LoRA adapters
are used to enhance the efficiency of fine-tuning large models with reduced memory usage. We will use
:ref:`PEFT<https://huggingface.co/docs/peft/main/en/package_reference/peft_model>` library
from HuggingFace to instantiate our model and add adapters to it.

By integrating adapters with AIMET quantization, we can perform similar functionalities as present in PEFT,
for example, changing adapter weights, enabling and disabling adapters. Along with this, we can tweak the quantization
parameters for the adapters alone to get good quantization accuracy.

User flow
----------

The user can use the following flow to quantize a model with LoRA adapters.

Step 1: Create a PEFT model with one adapter. Use PEFT APIs from HuggingFace to create a PEFT model

    >>> from peft import LoraConfig, get_peft_model
    >>> lora_config = LoraConfig(
    >>>    lora_alpha=16,
    >>>    lora_dropout=0.1,
    >>>    r=4,
    >>>    bias="none",
    >>>    target_modules=["linear"])
    >>> model = get_peft_model(model, lora_config)

Step 2: Replace lora layers with AIMET lora layers. This API helps AIMET quantize the lora layers

    >>> from aimet_torch.peft import replace_lora_layers_with_quantizable_layers
    >>> replace_lora_layers_with_quantizable_layers(model)

Step 3: Track meta data for lora layers such as adapter name, lora layer names & alpha param

    >>> from aimet_torch.peft import track_lora_meta_data
    >>> meta_data = track_lora_meta_data(model, tmp_dir, 'meta_data')
    >>> ## If linear lora layers were replaced with ConvInplaceLinear then
    >>> meta_data = track_lora_meta_data(model, tmp_dir, 'meta_data', ConvInplaceLinear)

Step 4: Create Quantization utilities

    >>> from aimet_torch.peft import PeftQuantUtils
    >>> peft_utils = PeftQuantUtils(meta_data)
    >>> ## If we are using a prepared model, then load name to module dict that gets saved as a json file
    >>> peft_utils = PeftQuantUtils(meta_data, name_to_module_dict)

**Next step will be to Prepare the model and create a QuantSim object (steps are not shown below, please refer to model preparer and
quantsim docs for reference)
Once Sim is created, we can use peft_utils to modify quantization attributes for lora layers in sim**

Step 5: Disable lora adapters. To compute base model encodings without the effect of adapters we need to disable lora adapters.

    >>> peft_utils.disable_lora_adapters(sim)

Step 6: Compute Encodings for sim (Not shown below, refer to quantsim docs) & freeze base model encodings for params.
(The step for computing the encoding for a model is not shows here). Since the base model weights are common across different
adapters, we don't need to recompute the encodings for them. Therefore, to speed up computation we freeze the base model params

    >>> peft_utils.freeze_base_model_param_quantizers(sim)

Step 7: Export base model and encodings

    >>> sim.export(tmpdir, 'model', dummy_input=dummy_inputs, export_model=True, filename_prefix_encodings='base_encodings')

Step 8: Load adapter weights for adapter 1

    >>> peft_utils.enable_adapter_and_load_weights(sim, 'tmpdir/lora_weights_after_adaptation_for_adapter1.safetensor', use_safetensor=True)

Step 9: Configure lora adapter quantizers

    >>> for name, lora_module in peft_utils.get_quantized_lora_layer(sim):
    >>>     ### Change bitwidth
    >>>     lora_module.param_quantizers['weight'].bitwidth = 16
    >>>     ### Change per tensor to per channel
    >>>     lora_module.param_quantizers['weight'] = aimet.quantization.affine.QuantizeDequantize(shape=(1, 1, 1, 1), bitwidth=16, symmetric=True).to(module.weight.device)

Step 10: Compute encodings for model & Export
    Here we do not show steps for how to compute the encoding. Please refer to Quantization simulation documentation
    Note: while exporting the model directory should be the same for base_model export and consecutive exports

    >>> sim.export(tmpdir, 'model', dummy_input=dummy_inputs, export_model=False, filename_prefix_encodings='adapter1')
    >>> peft_utils.export_adapter_weights(sim, tmpdir, 'adapter1_weights')

Step 11: For another adapter with same configration (rank & target module) repeat steps 8-10


Top-level API
-------------


.. autoclass:: aimet_torch.peft.AdapterMetaData

|

**The following API can be used to replace PEFT lora layers definition with AIMET lora layers definition**

.. automethod:: aimet_torch.peft.replace_lora_layers_with_quantizable_layers

|


**The following API can be used to track lora meta data. To be passed to peft utilities**

.. automethod:: aimet_torch.peft.track_lora_meta_data

|

.. autoclass:: aimet_torch.peft.PeftQuantUtils
    :members:
|

