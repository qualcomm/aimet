:orphan:

############################
Model Guidelines for PyTorch
############################

To implement the Cross Layer Equalization API,  
:code:`aimet_torch.cross_layer_equalization.equalize_model()`, AIMET creates a computing graph to analyze the sequence of operations in the model.

Certain model constructs prevent AIMET from creating and analyzing the computing graph. The following list describes these potential issues and workarounds.

.. note::

    These restrictions are not applicable if you are using the **Primitive APIs**.

**ONNX Export**
    *Description*: Use :code:`torch.onnx.export()` to export your model. Make sure ONNX export passes.
    
    *Workaround*: If ONNX export fails, rewrite the specific layer so that ONNX export passes.

**Slicing operation** 
    *Description*: Some models use :code:`torch.tensor.view()` in the forward function as follows:

    .. code:: python
        
        x = x.view(-1, 1024)

    If the view function is written this way, it causes an issue while creating the computing graph.

    *Workaround*: Rewrite the :code:`x.view()` statement as follows:
    
    .. code:: python
        
        x = x.view(x.size(0), -1)    
                    
**Bilinear, upsample operation**
    *Description*: Some models use the upsample operation in the forward function as:

    .. code:: python

        x =                              
        torch.nn.functional.upsample(x,    
        size=torch.Size([129,129]),        
        mode='bilinear',                   
        align_corners=True) 

    *Workaround*: Set the align_corners parameter to False as follows:

    .. code:: python     

        x =                              
        torch.nn.functional.upsample(x,    
        size=torch.Size([129,129]),        
        mode='bilinear',                   
        align_corners=False) 

**Deconvolution operation**
    *Description*: The dconvolution operation is used in the DeepLabV3 model. This is not supported by AIMET.

    *Workaround*: There is no workaround available at this time. This issue will be addressed in a subsequent AIMET release.                           
