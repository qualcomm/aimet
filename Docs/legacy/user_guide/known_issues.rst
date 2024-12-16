.. _ug-known-issues:

########################
AIMET Known Issues
########################

Known issues and limitations for Qualcomm AI Model Efficiency ToolKit (AIMET)

- AIMET Spatial SVD currently does not support Fully Connected layers
- AIMET Channel Pruning
    - Does not support Conv layers with dilation other than (1,1). Conv layers with dilation other than (1,1) must be added to Channel Pruning Configuration's modules_to_ignore list.
    - Does not support channel pruning of DepthwiseConv2d layers.
    - For TensorFlow, supports only models with "Channels Last" data format
