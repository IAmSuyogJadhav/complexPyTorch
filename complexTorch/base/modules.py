from complexTorch import nn


class Conv2dReLU(nn.ComplexSequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

#         if use_batchnorm == "inplace" and InPlaceABN is None:
#             raise RuntimeError(
#                 "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
#                 + "To install see: https://github.com/mapillary/inplace_abn"
#             )

        conv = nn.ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ComplexReLU()

        if use_batchnorm == "inplace":
#             bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
#             relu = nn.Identity()
            pass

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.ComplexBatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)

        
class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
#             self.attention = SCSEModule(**params)
            pass  #TODO: Complex Sigmoid can be implemented, thus SCSEModule can be implemented too. Refer: kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1742-18.pdf
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, xr, xi):
        return self.attention(xr, xi)
