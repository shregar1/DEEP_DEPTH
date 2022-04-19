import torch
import torch.nn as nn
from fastai.layers import ConvLayer
from models.dense_net import DenseNets
from models.modules.unet import BottleNeck, UnetBlock


class DenseUnet(nn.Module):
  def __init__(self,encoder_arch,input_channels):
    super(DenseUnet,self).__init__()
    self.densenet_encoder = DenseNets(input_channels=input_channels,
                                      encoder_arch=encoder_arch)
    self.densenet_encoder.load_state_dict(torch.load("/content/drive/MyDrive/DensetNet_169_encoder_model.pth"))
    self.bottleneck = BottleNeck(in_channels=self.densenet_encoder.in_channels)
    self.conv1 = nn.Conv2d(in_channels=self.densenet_encoder.in_channels+512,
                          out_channels=self.densenet_encoder.in_channels+512,kernel_size=3,
                          stride=1,padding=1)
    self.conv2 = nn.Conv2d(in_channels=self.densenet_encoder.in_channels+512+self.densenet_encoder.skip4_channels,
                          out_channels=self.densenet_encoder.in_channels+512,kernel_size=3,
                          stride=1,padding=1)
    
    self.in_channels = self.densenet_encoder.in_channels+512
    self.unet_block1 = UnetBlock(in_channels=self.densenet_encoder.in_channels+512,
                                 skip_channels=self.densenet_encoder.skip3_channels)
    
    self.in_channels = (self.in_channels+self.densenet_encoder.skip3_channels)//4
    self.unet_block2 = UnetBlock(in_channels=self.in_channels,
                                 skip_channels=self.densenet_encoder.skip2_channels)
    
    self.in_channels = (self.in_channels+self.densenet_encoder.skip2_channels)//4
    self.unet_block3 = UnetBlock(in_channels=self.in_channels,
                                 skip_channels=self.densenet_encoder.skip1_channels)
    
    self.in_channels = (self.in_channels+self.densenet_encoder.skip1_channels)//4
    self.unet_block4 = UnetBlock(in_channels=self.in_channels,
                                 skip_channels=self.densenet_encoder.skip0_channels)
    
    self.in_channels = (self.in_channels+self.densenet_encoder.skip0_channels)//4
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    self.upsample_conv = ConvLayer(self.in_channels,64,ks=3,stride=1,padding=None,
                                act_cls=None ,bias=None)
    self.out_conv = nn.Conv2d(in_channels=3+64,out_channels=1,kernel_size=1,
                              stride=1,padding=0)
    
  def forward(self,input_image):
    x0,x1,x2,x3,x4,x = self.densenet_encoder(input_image)
    x_ = self.bottleneck(x)
    x_ = torch.cat([x,x_],dim=1)
    x_ = self.conv1(x_)
    x_ = torch.cat([x4,x_],dim=1)
    x_ = self.conv2(x_)
    x_ = self.unet_block1(x3,x_)
    x_ = self.unet_block2(x2,x_)
    x_ = self.unet_block3(x1,x_)
    x_ = self.unet_block4(x0,x_)
    x_ = self.upsample(x_)
    x_ = self.upsample_conv(x_)
    x_ = torch.cat([input_image,x_],dim=1)
    out = self.out_conv(x_)
    return out
    