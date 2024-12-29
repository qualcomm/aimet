:orphan:

.. _quantsim-runtime-config:

#####################
Runtime configuration
#####################

Overview
========

You can configure runtime specific settings like enabling the quantizer, choosing between per-channel
and per-tensor quantization, selecting symmetric or asymmetric quantization, and specifying fused
operations. These configurations help you align with the quantization rules of a specific runtime
to simulate its quantization behavior.

Runtime-specific configurations and settings are defined in a JSON configuration file. These settings
are applied when the :class:`QuantizationSimModel` class is instantiated.

.. note::

    Start with the default configuration file ``aimet_common.quantsim_config.default_config_per_channel.json``.
    In most cases, you won't need to make any changes to it.


Configuration file structure
============================

The configuration file contains six main sections, ordered from less- to more specific:

.. image:: ../images/quantsim_config_file.png

Rules defined in a more general section are overridden by subsequent rules defined in a more specific case.
For example, you can specify in "defaults" that no layers be quantized, but then turn on quantization for specific layers in the "op_type" section.

How to modify configuration file
================================

Configure individual sections as described here.

1. defaults
-----------

.. literalinclude:: ../legacy/torch_code_examples/quantsim_config_example.py
   :language: python
   :start-after: # defaults start
   :end-before:  # defaults end

In the defaults section, include an "ops" dictionary and a "params" dictionary (though these dictionaries can be empty).

The ``ops`` dictionary holds settings that apply to all activation quantizers in the model.
The following settings are available:

* is_output_quantized:
    - Optional. If included, must be ``True``.
      Including this setting turns on all output activation quantizers by default.
      If not specified, all activation quantizers are disabled to start.

    - In cases when the runtime quantizes input activations, this is only done for certain op types.
      To configure these settings for specific op types see below.

* is_symmetric:
    - Optional. If included, value is ``True`` or ``False``.

    - ``True`` places all activation quantizers in symmetric mode by default.

    - ``False``, or omitting the parameter, sets all activation quantizers to asymmetric mode by default.

The ``params`` dictionary holds settings that apply to all parameter quantizers in the model.
The following settings are available:

* is_quantized:
    - Optional.  If included, value is ``True`` or ``False``.

    - ``True`` turns on all parameter quantizers by default.

    - ``False``, or omitting the parameter, disables all parameter quantizers by default.

* is_symmetric:
    - Optional. If included, value is ``True`` or ``False``.

    - ``True`` places all parameter quantizers in symmetric mode by default.

    - ``False``, or omitting the parameter, sets all parameter quantizers to asymmetric mode by default.

Outside the ``ops`` and ``params`` dictionaries, the following additional quantizer settings are available:

* strict_symmetric:
    - Optional. If included, value is ``True`` or ``False``.

    - ``True`` causes quantizers configured in symmetric mode to use strict symmetric quantization.

    - ``False``, or omitting the parameter, causes quantizers configured in symmetric mode to not use
      strict symmetric quantization.

* unsigned_symmetric:
    - Optional. If included, value is ``True`` or ``False``.

    - ``True`` causes quantizers configured in symmetric mode use unsigned symmetric quantization when
      available.

    - ``False``, or omitting the parameter, causes quantizers configured in symmetric mode to not use
      unsigned symmetric quantization.

* per_channel_quantization:
    - Optional. If included, value is ``True`` or ``False``.

    - ``True`` causes parameter quantizers to use per-channel quantization rather than per-tensor
      quantization.

    - ``False`` or omitting the parameter, causes parameter quantizers to use per-tensor quantization.

2. params
---------

.. literalinclude:: ../legacy/torch_code_examples/quantsim_config_example.py
   :language: python
   :start-after: # params start
   :end-before:  # params end


In the params section, configure settings for parameters that apply throughout the model.
For example, adding settings for ``weight`` affects all parameters of type ``weight`` in the model.
Supported parameter types include:

* ``weight``
* ``bias``

For each parameter type, the following settings are available:

* is_quantized:
    - Optional. If included, value is ``True`` or ``False``.

    - ``True`` turns on all parameter quantizers of that type.

    - ``False`` disables all parameter quantizers of that type.

Omitting the setting causes the parameter to use the setting specified by the defaults section.

* is_symmetric:
    - Optional. If included, value is ``True`` or ``False``.

    - ``True`` places all parameter quantizers of that type in symmetric mode.

    - ``False`` places all parameter quantizers of that type in asymmetric mode.

    - Omitting the setting causes the parameter to use the setting specified by the defaults section.

#. **op_type**:

.. literalinclude:: ../legacy/torch_code_examples/quantsim_config_example.py
   :language: python
   :start-after: # op_type start
   :end-before:  # op_type end

In the op_type section, configure settings affecting particular op types.
The configuration file supports ONNX op types, and internally maps the type to a PyTorch or TensorFlow op
type depending on which framework is used.

For each op type, the following settings are available:

* is_input_quantized:
    - Optional. If included, must be ``True``.

    - Including this setting turns on input quantization for all ops of this op type.

    - Omitting the setting keeps input quantization disabled for all ops of this op type.

* is_output_quantized:
    - Optional. If included, value is ``True`` or ``False``.

    - ``True`` turns on output quantization for all ops of this op type.

    - ``False`` disables output quantization for all ops of this op type.

    - Omitting the setting causes output quantizers of this op type to fall back to the setting specified
      by the defaults section.

* is_symmetric:
    - Optional. If included, value is ``True`` or ``False``.

    - ``True`` places all quantizers of this op type in symmetric mode.

    - ``False`` places all quantizers of this op type in asymmetric mode.

    - Omitting the setting causes quantizers of this op type to fall back to the setting specified by the defaults section.

* per_channel_quantization:
    - Optional. If included, value is ``True`` or ``False``.

    - ``True`` sets parameter quantizers of this op type to use per-channel quantization rather than
      per-tensor quantization.

    - ``False`` sets parameter quantizers of this op type to use per-tensor quantization.

    - Omitting the setting causes parameter quantizers of this op type to fall back to the setting specified by the defaults section.

For a particular op type, settings for particular parameter types can also be specified.
For example, specifying settings for weight parameters of a Conv op type affects only Conv weights and not weights of Gemm op types.

To specify settings for param types of an op type, include a ``params`` dictionary under the op type.
Settings for this section follow the same convention as settings for parameter types in the ``params``
section, but only affect parameters for this op type.

3. supergroups
--------------

.. literalinclude:: ../legacy/torch_code_examples/quantsim_config_example.py
   :language: python
   :start-after: # supergroups start
   :end-before:  # supergroups end

Supergroups are a sequence of operations that are fused during quantization, meaning no quantization noise
is introduced between members of the supergroup. For example, specifying [``Conv``, ``Relu``] as a
supergroup disables quantization between any adjacent ``Conv`` and ``Relu`` ops in the model.

When searching for supergroups in the model, only sequential groups of ops with no branches in between
are matched with supergroups defined in the list. Using [``Conv``, ``Relu``] as an example, if there were
a ``Conv`` op in the model whose output is used by both a ``Relu`` op and another op, the supergroup would
not include those ``Conv`` and ``Relu`` ops.

To specify supergroups in the config file, add each entry as a list of op type strings.
The configuration file supports ONNX op types, and internally maps the type to a PyTorch or TensorFlow op
type depending on which framework is used.

4. model_input
--------------

.. literalinclude:: ../legacy/torch_code_examples/quantsim_config_example.py
   :language: python
   :start-after: # model_input start
   :end-before:  # model_input end

Use the ``model_input`` section to configure the quantization of inputs to the model.
The following setting is available:

* is_input_quantized:
    - Optional. If included, must be ``True``.

    - Including this setting turns on quantization for input quantizers to the model.

    - Omitting the setting keeps input quantizers at settings resulting from more general configurations.

5. model_output
---------------

.. literalinclude:: ../legacy/torch_code_examples/quantsim_config_example.py
   :language: python
   :start-after: # model_output start
   :end-before:  # model_output end

Use the ``model_output`` section to configure the quantization of outputs of the model.
The following setting is available:

* is_output_quantized:
    - Optional. If included, it must be set to ``True``.

    - Including this setting turns on quantization for output quantizers of the model.

    - Omitting the setting keeps input quantizers at settings resulting from more general configurations.
