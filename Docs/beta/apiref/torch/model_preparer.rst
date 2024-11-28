.. _apiref-torch-model-preparer:

##########################
aimet_torch.model_preparer
##########################

AIMET PyTorch ModelPreparer API uses new graph transformation feature available in PyTorch 1.9+ version and automates
model definition changes required by user. For example, it changes functionals defined in forward pass to
torch.nn.Module type modules for activation and elementwise functions. Also, when torch.nn.Module type modules are reused,
it unrolls into independent modules.

Users are strongly encouraged to use AIMET PyTorch ModelPreparer API first and then use the returned model as input
to all the AIMET Quantization features.

AIMET PyTorch ModelPreparer API requires minimum PyTorch 1.9 version.


.. container:: tab-heading
   
   Top-level API

.. autofunction:: aimet_torch.model_preparer.prepare_model


.. container:: tab-heading
   
Limitations of torch.fx symbolic trace API

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
