import torch

class SegmentationModel(torch.nn.Module):

    def initialize(self):
        pass
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)

    def forward(self, xr, xi):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(xr, xi)
        decoder_output = self.decoder(*features)

#         return decoder_output #DEBUG
        masks = self.segmentation_head(*decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, xr, xi):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            xr, xi = self.forward(xr, xi)

        return xr, xi