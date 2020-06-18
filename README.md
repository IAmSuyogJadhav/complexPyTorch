# complexPyTorch

A high-level toolbox for using complex valued neural networks in PyTorch. This fork implements a few more layers and some deep learning architectures. 

## Example Usage (Complex UNet)
```python
import torch
import complexTorch

n_samples, channels, h, w = 2, 60, 256, 256 
sample_batch = torch.ones([n_samples, channels, h, w, 2]).double()  # 2 for real and imaginary parts, respectively
real, imag = sample_batch[..., 0].contiguous().to('cuda'), sample_batch[..., 1].contiguous().to('cuda')

model = complexTorch.models.Unet(
    encoder_name='resnet34',
    encoder_depth=5,
    decoder_use_batchnorm=True,
    decoder_channels=(256, 128, 64, 32, 16),
    decoder_attention_type=None,
    in_channels=60,
    upsampling=2,
    classes=1,
    aux_params=None,
)
model = model.double().to('cuda')

# Test Forward pass
out = model(real, imag)  # Returns two vectors, real and imaginary part respectively.
print(out[0].shape)  # [2, 1, 512, 512]  (h, w = 512 here due to the upsampling being set to 2 in the model definition)

# Test Backward pass
lh = out[1].sum() + out[0].sum()
lh.backward()
```

## To-Do
* There is apparently a problem with backpropagation of complex layers ([see the issue](https://github.com/wavefrontshaping/complexPyTorch/issues/3)). The gradients calculated by torch's autograd do not match with what a complex differentiation gives. Need to implement a proper backward function for that purpose.

## Updates

* **18-06-2020**
    - Fixed problems with the UNet architecture. A bug related to CUDNN was encountered. It was fixed in torch 1.5.0. Thus, the project now requires torch >=1.5.0.
* **14-06-2020** 
    - Added Complex U-Net architecture, with ResNet backend. Adapted heavily from [qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch/)
    - Added `nn.ComplexBilinearUpsampling2D` and `nn.Identity` layers.
* **01-06-2020**
    - Added Complex ResNet architecture.

---------

The following description is as-is from the original repository.

## Complex Valued Networks with PyTorch

Artificial neural networks are mainly used for treating data encoded in real values, such as digitized images or sounds. 
In such systems, using complex-valued tensor would be quite useless. 
However, for physic related topics, in particular when dealing with wave propagation, using complex values is interesting as the physics typically has linear, hence more simple, behavior when considering complex fields. 
complexPyTorch is a simple implementation of complex-valued functions and modules using the high-level API of PyTorch. 
Following [[C. Trabelsi et al., International Conference on Learning Representations, (2018)](https://openreview.net/forum?id=H1T2hmZAb)], it allows the following layers and functions to be used with complex values:
* Linear
* Conv2d
* MaxPool2d
* Relu (&#8450;Relu)
* BatchNorm1d (Naive and Covariance approach)
* BatchNorm2d (Naive and Covariance approach)


## Syntax and usage

The syntax is supposed to copy the one of the standard real functions and modules from PyTorch. 
The names are the same as in `nn.modules` and `nn.functional` except that they start with `Complex` for Modules, e.g. `ComplexRelu`, `ComplexMaxPool2d` or `complex_` for functions, e.g. `complex_relu`, `complex_max_pool2d`.
The only usage difference is that the forward function takes two tensors, corresponding to real and imaginary parts, and returns two ones too.

## BatchNorm

For all other layers, using the recommendation of [[C. Trabelsi et al., International Conference on Learning Representations, (2018)](https://openreview.net/forum?id=H1T2hmZAb)], the calculation can be done in a straightforward manner using functions and modules form `nn.modules` and `nn.functional`. 
For instance, the function `complex_relu` in `complexFunctions`, or its associated module `ComplexRelu` in `complexLayers`, simply performs `relu` both on the real and imaginary part and returns the two tensors.
The complex BatchNorm proposed in [[C. Trabelsi et al., International Conference on Learning Representations, (2018)](https://openreview.net/forum?id=H1T2hmZAb)] requires the calculation of the inverse square root of the covariance matrix.
This is implemented in `ComplexbatchNorm1D` and `ComplexbatchNorm2D` but using the high-level PyTorch API, which is quite slow.
The gain of using this approach, however, can be experimentally marginal compared to the naive approach which consists in simply performing the BatchNorm on both the real and imaginary part, which is available using `NaiveComplexbatchNorm1D` or `NaiveComplexbatchNorm2D`.


>  Example code removed as the code has undergone an overhaul in this repo.
        
## Todo
* Script ComplexBatchNorm for improved efficiency ([jit doc](https://pytorch.org/docs/stable/jit.html))
* Add more layers (Conv1D, Upsample, ConvTranspose...)
* Add complex cost functions and usual functions (e.g. Pearson correlation)

## Acknowledgments

I want to thank Piotr Bialecki for his invaluable help on the PyTorch forum
