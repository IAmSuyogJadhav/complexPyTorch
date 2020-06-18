from .resnet import resnet_encoders
from . import utils 

encoders = {}
encoders.update(resnet_encoders)


def get_encoder(name, in_channels=3, depth=5): #, weights=None):
    Encoder = encoders[name]["encoder"]
    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

#     if weights is not None:
#         settings = encoders[name]["pretrained_settings"][weights]
#         encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels)

    return encoder