import complexTorch.nn as nn


class SegmentationHead(nn.ComplexSequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.ComplexConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.ComplexUpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = nn.ComplexReLU()  # Always
        super().__init__(conv2d, upsampling, activation)
