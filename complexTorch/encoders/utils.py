import torch
import complexTorch.nn as nn

        
def patch_first_conv(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.ComplexConv2d):
            break

    # change input channels for first conv
    module.conv_r.in_channels, module.conv_i.in_channels = in_channels, in_channels
    
    weight_r, weight_i = module.conv_r.weight.detach(), module.conv_i.weight.detach()
    reset = False

    if in_channels == 1:
        weight_r, weight_i = weight_r.sum(1, keepdim=True), weight_i.sum(1, keepdim=True)
    elif in_channels == 2:
        weight_r, weight_i = weight_r[:, :2] * (3.0 / 2.0), weight_i[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight_r = torch.Tensor(
            module.conv_r.out_channels,
            module.conv_r.in_channels // module.conv_r.groups,
            *module.conv_r.kernel_size
        )
        weight_i = torch.Tensor(
            module.conv_i.out_channels,
            module.conv_i.in_channels // module.conv_i.groups,
            *module.conv_i.kernel_size
        )

    module.conv_r.weight = nn.parameter.Parameter(weight_r)
    module.conv_i.weight = nn.parameter.Parameter(weight_i)
    
    if reset:
        module.conv_r.reset_parameters()
        module.conv_i.reset_parameters() 


class EncoderMixin:
    """Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels
    """

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels):
        """Change first convolution chennels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, in_channels=in_channels)
