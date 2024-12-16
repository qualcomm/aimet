.. _torch-model-guidelines:

########################
PyTorch model guidelines
########################

Model guidelines
================

In order to make full use of AIMET features, there are several guidelines users are encouraged to follow when defining
PyTorch models.

**Model should support conversion to onnx**

The model definition should support conversion to onnx, user could check compatibility of model for onnx conversion as
shown below::

    ...
    model = Model()
    torch.onnx.export(model, <dummy_input>, <onnx_file_name>):

**Model should be jit traceable**

The model definition should be jit traceable, user could check compatibility of model for jit tracing as
shown below::

    ...
    model = Model()
    torch.jit.trace(model, <dummy_input>):

**Define layers as modules instead of using torch.nn.functional equivalents**

When using activation functions and other stateless layers, PyTorch will allow the user to either

- define the layers as modules (instantiated in the constructor and used in the forward pass), or
- use a torch.nn.functional equivalent purely in the forward pass

For AIMET quantization simulation model to add simulation nodes, AIMET requires the former (layers defined as modules).
Changing the model definition to use modules instead of functionals, is mathematically equivalent and does not require
the model to be retrained.

As an example, if the user had::

    def forward(...):
        ...
        x = torch.nn.functional.relu(x)
        ...

Users should instead define their model as::

    def __init__(self,...):
        ...
        self.relu = torch.nn.ReLU()
        ...

    def forward(...):
        ...
        x = self.relu(x)
        ...

This will not be possible in certain cases where operations can only be represented as functionals and not as class
definitions, but should be followed whenever possible.

Also, User can also automate this by using :ref:`Model Preparer API<api-torch-model-preparer>`

**Avoid reuse of class defined modules**

Modules defined in the class definition should only be used once. If any modules are being reused, instead define a new
identical module in the class definition.
For example, if the user had::

    def __init__(self,...):
        ...
        self.relu = torch.nn.ReLU()
        ...

    def forward(...):
        ...
        x = self.relu(x)
        ...
        x2 = self.relu(x2)
        ...

Users should instead define their model as::

    def __init__(self,...):
        ...
        self.relu = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        ...

    def forward(...):
        ...
        x = self.relu(x)
        ...
        x2 = self.relu2(x2)
        ...



**Use only torch.Tensor or tuples of torch.Tensors as model/submodule inputs and outputs**

Modules should use tensor or tuples of tensor for inputs and output in order to support conversion of the model to onnx.
For example, if the user had::

    def __init__(self,...):
    ...
    def forward(self, inputs: Dict[str, torch.Tensor]):
        ...
        x = self.conv1(inputs['image_rgb'])
        rgb_output = self.relu1(x)
        ...
        x = self.conv2(inputs['image_bw'])
        bw_output = self.relu2(x)
        ...
        return { 'rgb': rgb_output, 'bw': bw_output }

Users should instead define their model as::

    def __init__(self,...):
    ...
    def forward(self, image_rgb, image_bw):
        ...
        x = self.conv1(image_rgb)
        rgb_output = self.relu1(x)
        ...
        x = self.conv2(image_bw)
        bw_output = self.relu2(x)
        ...
        return rgb_output, bw_output


Prepare and validate the model
------------------------------

:mod:`aimet_torch` has two utilities to validate/automate model complaince:

* Model validator utility automates checking PyTorch model requirements
* Model preparer utility automates updating model definition to align with requirements

In model prep and validation using :mod:`aimet_torch`, we recommend the following flow:

.. image:: ../../../images/pytorch_model_prep_and_validate.PNG

Use the Model Validator utility to check if the model definition is compliant with AIMET. If validator
checks fail, put Model Preparer in the pipeline and retry Model Validator.

.. note::
    If the model validator keeps producing warnings, update the model definition using either the model
    preparer or by manually modifying it.

Model validator
===============

AIMET provides a model validator utility to help check whether AIMET feature can be applied on a Pytorch model. The
model validator currently checks for the following conditions:

- No modules are reused
- Operations have modules associated with them and are not defined as Functionals (excluding a set of known operations)

In this section, we present models failing the validation checks, and show how to run the model validator, as well as
how to fix the models so the validation checks pass.

**Example 1: Model with reused modules**

We begin with the following model, which contains two relu modules sharing the same module instance.

.. literalinclude:: ../../../torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: ModelWithReusedNodes
   :emphasize-lines: 13, 15

Import the model validator:

.. literalinclude:: ../../../torch_code_examples/model_validator_code_example.py
   :language: python
   :lines: 44

Run the model validator on the model by passing in the model as well as model input:

.. literalinclude:: ../../torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: validate_example_model

For each validation check run on the model, a logger print will appear::

    Utils - INFO - Running validator check <function validate_for_reused_modules at 0x7f127685a598>

If the validation check finds any issues with the model, the log will contain information for how to resolve the model::

    Utils - WARNING - The following modules are used more than once in the model: ['relu1']
    AIMET features are not designed to work with reused modules. Please redefine your model to use distinct modules for
    each instance.

Finally, at the end of the validation, any failing checks will be logged::

    Utils - INFO - The following validator checks failed:
    Utils - INFO -     <function validate_for_reused_modules at 0x7f127685a598>

In this case, the validate_for_reused_modules check informs that the relu1 module is being used multiple times in the
model. We rewrite the model by defining a separate relu instance for each usage:

.. literalinclude:: ../torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: ModelWithoutReusedNodes
   :emphasize-lines: 9, 16

Now, after rerunning the model validator, all checks pass::

    Utils - INFO - Running validator check <function validate_for_reused_modules at 0x7ff577373598>
    Utils - INFO - Running validator check <function validate_for_missing_modules at 0x7ff5703eff28>
    Utils - INFO - All validation checks passed.

**Example 2: Model with functionals**

We start with the following model, which uses a torch linear functional layer in the forward pass:

.. literalinclude:: ../torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: ModelWithFunctionalLinear
   :emphasize-lines: 17

Running the model validator shows the validate_for_missing_modules check failing::

    Utils - INFO - Running validator check <function validate_for_missing_modules at 0x7f9dd9bd90d0>
    Utils - WARNING - Ops with missing modules: ['matmul_8']
    This can be due to several reasons:
    1. There is no mapping for the op in ConnectedGraph.op_type_map. Add a mapping for ConnectedGraph to recognize and
    be able to map the op.
    2. The op is defined as a functional in the forward function, instead of as a class module. Redefine the op as a
    class module if possible. Else, check 3.
    3. This op is one that cannot be defined as a class module, but has not been added to ConnectedGraph.functional_ops.
    Add to continue.
    Utils - INFO - The following validator checks failed:
    Utils - INFO - 	<function validate_for_missing_modules at 0x7f9dd9bd90d0>

The check has identified matmul_8 as an operation with a missing pytorch module. In this case, it is due to reason #2
in the log, in which the layer has been defined as a functional in the forward function. We rewrite the model by
defining the layer as a module instead in order to resolve the issue.

.. literalinclude:: ../torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: ModelWithoutFunctionalLinear
   :emphasize-lines: 10, 20

Model preparer
==============

AIMET PyTorch ModelPreparer API uses new graph transformation feature available in PyTorch 1.9+ version and automates
model definition changes required by user. For example, it changes functionals defined in forward pass to
torch.nn.Module type modules for activation and elementwise functions. Also, when torch.nn.Module type modules are reused,
it unrolls into independent modules.

Users are strongly encouraged to use AIMET PyTorch ModelPreparer API first and then use the returned model as input
to all the AIMET Quantization features.

AIMET PyTorch ModelPreparer API requires minimum PyTorch 1.9 version.

Code Examples
-------------

**Required imports**

.. literalinclude:: ../../../torch_code_examples/model_preparer_code_example.py
   :language: python
   :start-after: # ModelPreparer imports
   :end-before: # End of import statements

**Example 1: Model with Functional relu**

We begin with the following model, which contains two functional relus and relu method inside forward method.

.. literalinclude:: ../../../torch_code_examples/model_preparer_code_example.py
   :language: python
   :pyobject: ModelWithFunctionalReLU
   :emphasize-lines: 11, 12, 14, 15

Run the model preparer API on the model by passing in the model.

.. literalinclude:: ../../../torch_code_examples/model_preparer_code_example.py
   :language: python
   :pyobject: model_preparer_functional_example

After that, we get prepared_model, which is functionally same as original model. User can verify this by comparing
the outputs of both models.

prepared_model should have all three functional relus now converted to :mod:`torch.nn.ReLU` modules which satisfy
model guidelines.


**Example 2: Model with reused torch.nn.ReLU module**

We begin with the following model, which contains torch.nn.ReLU module which is used at multiple instances inside
model forward function.

.. literalinclude:: ../../../torch_code_examples/model_preparer_code_example.py
   :language: python
   :pyobject: ModelWithReusedReLU
   :emphasize-lines: 13, 15, 18, 20

Run the model preparer API on the model by passing in the model.

.. literalinclude:: ../../../torch_code_examples/model_preparer_code_example.py
   :language: python
   :pyobject: model_preparer_reused_example

After that, we get prepared_model, which is functionally same as original model. User can verify this by comparing
the outputs of both models.

prepared_model should have separate independent torch.nn.Module instances which satisfy model guidelines.

**Example 3: Model with elementwise Add**

We begin with the following model, which contains elementwise Add operation inside model forward function.

.. literalinclude:: ../torch_code_examples/model_preparer_code_example.py
   :language: python
   :pyobject: ModelWithElementwiseAddOp
   :emphasize-lines: 10

Run the model preparer API on the model by passing in the model.

.. literalinclude:: ../torch_code_examples/model_preparer_code_example.py
   :language: python
   :pyobject: model_preparer_elementwise_add_example

After that, we get prepared_model, which is functionally same as original model. User can verify this by comparing
the outputs of both models.

Limitations
-----------

.. note::
    Limitations of torch.fx symbolic trace: https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing

**1. Dynamic control flow is not supported by torch.fx**
Loops or if-else statement where condition may depend on some of the input values. It can only trace one execution
path and all the other branches that weren't traced will be ignored. For example, following simple function when traced,
will fail with TraceError saying that 'symbolically traced variables cannot be used as inputs to control flow'::

        def f(x, flag):
            if flag:
                return x
            else:
                return x*2

        torch.fx.symbolic_trace(f) # Fails!
        fx.symbolic_trace(f, concrete_args={'flag': True})

Workarounds for this problem:

- Many cases of dynamic control flow can be simply made to static control flow which is supported by torch.fx
  symbolic tracing. Static control flow is where loops or if-else statements whose value can't change
  across different model forward passes. Such cases can be traced by removing data dependencies on input values by
  passing concrete values to 'concrete_args' to specialize your forward functions.

- In truly dynamic control flow, user should wrap such piece of code at model-level scope using torch.fx.wrap API
  which will preserve it as a node instead of being traced through::

    @torch.fx.wrap
    def custom_function_not_to_be_traced(x, y):
        """ Function which we do not want to be traced, when traced using torch FX API, call to this function will
        be inserted as call_function, and won't be traced through """
        for i in range(2):
            x += x
            y += y
        return x * x + y * y



**2. Non-torch functions which does not use __torch_function__ mechanism is not supported by default in symbolic
tracing.**

Workaround for this problem:

- If we do not want to capture them in symbolic tracing then user should use torch.fx.wrap() API at module-level scope::

        import torch
        import torch.fx
        torch.fx.wrap('len')  # call the API at module-level scope.
        torch.fx.wrap('sqrt') # call the API at module-level scope.

        class ModelWithNonTorchFunction(torch.nn.Module):
            def __init__(self):
                super(ModelWithNonTorchFunction, self).__init__()
                self.conv = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x = self.conv(inputs[0])
                return x / sqrt(len(x))

        model = ModelWithNonTorchFunction().eval()
        model_transformed = prepare_model(model)


**3. Customizing the behavior of tracing by overriding the Tracer.is_leaf_module() API**

In symbolic tracing, leaf modules appears as node rather than being traced through and all the standard torch.nn modules
are default set of leaf modules. But this behavior can be changed by overriding the Tracer.is_leaf_module() API.

AIMET model preparer API exposes 'module_to_exclude' argument which can be used to prevent certain module(s) being
traced through. For example, let's examine following code snippet where we don't want to trace CustomModule further::

        class CustomModule(torch.nn.Module):
            @staticmethod
            def forward(x):
                return x * torch.nn.functional.softplus(x).sigmoid()

        class CustomModel(torch.nn.Module):
            def __init__(self):
                super(CustomModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=2)
                self.custom = CustomModule()

            def forward(self, inputs):
                x = self.conv1(inputs)
                x = self.custom(x)
                return x

        model = CustomModel().eval()
        prepared_model = prepare_model(model, modules_to_exclude=[model.custom])
        print(prepared_model)

In this example, 'self.custom' is preserved as node and not being traced through.

**4. Tensor constructors are not traceable**

For example, let's examine following code snippet::

            def f(x):
                return torch.arange(x.shape[0], device=x.device)

            torch.fx.symbolic_trace(f)

            Error traceback:
                return torch.arange(x.shape[0], device=x.device)
                TypeError: arange() received an invalid combination of arguments - got (Proxy, device=Attribute), but expected one of:
                * (Number end, *, Tensor out, torch.dtype dtype, torch.layout layout, torch.device device, bool pin_memory, bool requires_grad)
                * (Number start, Number end, Number step, *, Tensor out, torch.dtype dtype, torch.layout layout, torch.device device, bool pin_memory, bool requires_grad)

The above snippet is problematic because arguments to torch.arange() are input dependent.
Workaround for this problem:

- use deterministic constructors (hard-coding) so that the value they produce will be embedded as constant in
  the graph::

            def f(x):
                return torch.arange(10, device=torch.device('cpu'))

- Or use torch.fx.wrap API to wrap torch.arange() and call that instead::

        @torch.fx.wrap
        def do_not_trace_me(x):
            return torch.arange(x.shape[0], device=x.device)

        def f(x):
            return do_not_trace_me(x)

        torch.fx.symbolic_trace(f)

API
===

.. autoclass:: aimet_torch.model_validator.model_validator.ModelValidator
    :members:

.. include:: ../../apiref/torch/model_preparer.rst
    :start-after: # start-after
