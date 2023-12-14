import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

import pkg_resources 
from urllib import request

from .utils import MaxFeatureMap
from .utils import group
from .utils import resblock

import logging
logger = logging.getLogger("bob.pad.face")



class MCCNNCenterLoss(nn.Module):
  """ The class defining the MCCNN

  This class implements the MCCNN for multi-channel PAD
  
  Attributes
  ----------
  num_channels: int
    The number of channels present in the input
  lcnn_layers: list
  	The adaptable layers present in the base LightCNN model
  module_dict: dict
  	A dictionary containing module names and `torch.nn.Module` elements as key, value pairs.
  layer_dict: :py:class:`torch.nn.ModuleDict`
  	Pytorch class containing the modules as a dictionary. 
  light_cnn_model_file: str
  	Absolute path to the pretrained LightCNN model file. 
  url: str
  	The path to download the pretrained LightCNN model from.
  
  """
  def __init__(self, block=resblock, layers=[1, 2, 3, 4], num_channels=3, verbosity_level=2, use_sigmoid=True):
    """ Init function

    Parameters
    ----------

    num_channels: int
      The number of channels present in the input
    use_sigmoid: bool
      Whether to use sigmoid in eval phase. If set to `False` do not use
      sigmoid in eval phase. Training phase is not affected.
    verbosity_level: int
      Verbosity level.
    
    """
    super(MCCNNCenterLoss, self).__init__()

    self.num_channels=num_channels
    self.use_sigmoid=use_sigmoid

    self.lcnn_layers=['conv1','block1','group1','block2', 'group2','block3','group3','block4','group4','fc']

    logger.setLevel(verbosity_level)


    self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    # newly added FC layers

    self.linear1fc=nn.Linear(256*num_channels,10)
    self.linear2fc=nn.Linear(10,1)

    # add modules 

    module_dict={}

    for i in range(self.num_channels):

      m_dict={}

      m_dict['conv1']  = MaxFeatureMap(1, 48, 5, 1, 2)
      m_dict['block1'] = self._make_layer(block, layers[0], 48, 48)
      m_dict['group1'] = group(48, 96, 3, 1, 1)
      m_dict['block2'] = self._make_layer(block, layers[1], 96, 96)
      m_dict['group2'] = group(96, 192, 3, 1, 1)
      m_dict['block3'] = self._make_layer(block, layers[2], 192, 192)
      m_dict['group3'] = group(192, 128, 3, 1, 1)
      m_dict['block4'] = self._make_layer(block, layers[3], 128, 128)
      m_dict['group4'] = group(128, 128, 3, 1, 1)
      m_dict['fc']   = MaxFeatureMap(8*8*128, 256, type=0)

      # ch_0_should be the anchor 

      for layer in self.lcnn_layers:

        layer_name="ch_{}_".format(i)+layer

        module_dict[layer_name] = m_dict[layer]

    self.layer_dict = nn.ModuleDict(module_dict)



    # check for pretrained model 

    light_cnn_model_file = '/home/kangcaixin/chenjiawei/ddpm-segmentation/face_anti_spoofing/MCCNN_BCE_OCCL_GMM/LightCNN_29Layers_checkpoint.pth.tar'
    logger.info("Light_cnn_model_file path: {}".format(light_cnn_model_file))

    ## Loding the pretrained model for ch_0

    self.load_state_dict(self.get_model_state_dict(light_cnn_model_file),strict=False)

    # copy over the weights to all other layers

    for layer in self.lcnn_layers:

      for i in range(1, self.num_channels): # except for 0 th channel

        self.layer_dict["ch_{}_".format(i)+layer].load_state_dict(self.layer_dict["ch_0_"+layer].state_dict())  

            
  def _make_layer(self, block, num_blocks, in_channels, out_channels):
    """ makes multiple copies of the same base module

    Parameters
    ----------
    block: :py:class:`torch.nn.Module`
      The base block to replicate
    num_blocks: int
      Number of copies of the block to be made
    in_channels: int
      Number of input channels for a block
    out_channels: int
      Number of output channels for a block
    """
    layers = []
    for i in range(0, num_blocks):
        layers.append(block(in_channels, out_channels))
    return nn.Sequential(*layers)

  def forward(self, img):
    """ Propagate data through the network

    Parameters
    ----------
    img: :py:class:`torch.Tensor` 
      The data to forward through the network. Image of size num_channelsx128x128

    Returns
    -------
    output: :py:class:`torch.Tensor` 
      score 

    """
    
    embeddings=[]

    for i in range(self.num_channels):

      x=img[:,i,:,:].unsqueeze(1) # the image for the specific channel

      x = self.layer_dict["ch_{}_".format(i)+"conv1"](x)
      x = self.pool1(x)

      x = self.layer_dict["ch_{}_".format(i)+"block1"](x)
      x = self.layer_dict["ch_{}_".format(i)+"group1"](x)
      x = self.pool2(x)

      x = self.layer_dict["ch_{}_".format(i)+"block2"](x)
      x = self.layer_dict["ch_{}_".format(i)+"group2"](x)
      x = self.pool3(x)

      x = self.layer_dict["ch_{}_".format(i)+"block3"](x)
      x = self.layer_dict["ch_{}_".format(i)+"group3"](x)
      x = self.layer_dict["ch_{}_".format(i)+"block4"](x)
      x = self.layer_dict["ch_{}_".format(i)+"group4"](x)
      x = self.pool4(x)

      x = x.view(x.size(0), -1)

      fc = self.layer_dict["ch_{}_".format(i)+"fc"](x)

      fc = F.dropout(fc, training=self.training)

      embeddings.append(fc)

    merged = torch.cat(embeddings, 1)

    output = self.linear1fc(merged)

    embedding = nn.Sigmoid()(output)

    output = self.linear2fc(embedding)

    if  self.training or self.use_sigmoid:

      output=nn.Sigmoid()(output)

    return embedding, output

  @staticmethod
  def get_mccnnpath():

    import pkg_resources
    return pkg_resources.resource_filename('bob.learn.pytorch', 'models')


  def get_model_state_dict(self,pretrained_model_path):


    """ The class to load pretrained LightCNN model

    Attributes
    ----------
    pretrained_model_path: str
      Absolute path to the LightCNN model file

    new_state_dict: dict
      Dictionary with LightCNN weights

    """

    checkpoint = torch.load(pretrained_model_path,map_location=lambda storage,loc:storage)
    start_epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      name = 'layer_dict.ch_0_'+k[7:] # remove `module.`
      new_state_dict[name] = v
    # load params
    return new_state_dict