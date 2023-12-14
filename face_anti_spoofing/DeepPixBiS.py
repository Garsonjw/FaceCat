import torch
from torch import nn
from torchvision import models


class DeepPixBiS(nn.Module):
    """ The class defining Deep Pixelwise Binary Supervision for Face Presentation
    Attack Detection:

    Reference: Anjith George and SÃ©bastien Marcel. "Deep Pixel-wise Binary Supervision for 
    Face Presentation Attack Detection." In 2019 International Conference on Biometrics (ICB).IEEE, 2019.

    Attributes
    ----------
    pretrained: bool
        If set to `True` uses the pretrained DenseNet model as the base. If set to `False`, the network
        will be trained from scratch. 
        default: True      
    """

    def __init__(self, pretrained=True):

        """ Init function

        Parameters
        ----------
        pretrained: bool
            If set to `True` uses the pretrained densenet model as the base. Else, it uses the default network
            default: True
        """
        super(DeepPixBiS, self).__init__()

        dense = models.densenet161(pretrained=pretrained)

        features = list(dense.features.children())

        self.enc = nn.Sequential(*features[0:8])

        self.dec = nn.Conv2d(384, 1, kernel_size=1, padding=0)

        self.linear = nn.Linear(7 * 7, 1)

    def forward(self, x):
        """ Propagate data through the network

        Parameters
        ----------
        img: :py:class:`torch.Tensor` 
          The data to forward through the network. Expects RGB image of size 3x224x224

        Returns
        -------
        dec: :py:class:`torch.Tensor` 
            Binary map of size 1x14x14
        op: :py:class:`torch.Tensor`
            Final binary score.  

        """
        enc = self.enc(x)

        dec = self.dec(enc)

        dec = nn.Sigmoid()(dec)

        dec_flat = dec.view(-1, 7 * 7)

        op = self.linear(dec_flat)

        op = nn.Sigmoid()(op)

        return dec, op
