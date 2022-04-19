import torch
import torch.nn as nn
from models.modules.attention_net import AttentionUnit

class BottleNeck(nn.Module):
  def __init__(self,in_channels):
    super(BottleNeck,self).__init__()
    self.bottle_conv1 = nn.Conv2d(in_channels=in_channels,out_channels=1024,
                          kernel_size=3,stride=1,padding=1)
    self.bottle_conv2=nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=1)

  def forward(self,x):
    x=self.bottle_conv1(x)
    x=self.bottle_conv2(x)
    return x

class UnetBlock(nn.Module):
  def __init__(self,in_channels,skip_channels):
    super(UnetBlock,self).__init__()
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    self.attention = AttentionUnit(skip_channels)
    self.conv1 = nn.Conv2d(in_channels=in_channels+skip_channels,
                          out_channels=(in_channels+skip_channels)//2,
                          kernel_size=3,stride=1,
                          padding=1)
    self.conv2 = nn.Conv2d(in_channels=(in_channels+skip_channels)//2,
                           out_channels=(in_channels+skip_channels)//4,
                           kernel_size=3,stride=1,
                           padding=1)
    self.leaky_relu = nn.LeakyReLU(0.2)

  def forward(self,skip,x):
    x = self.upsample(x)
    skip = self.attention(skip) 
    x = torch.cat([skip,x],dim=1)
    x = self.conv1(x)
    x = self.leaky_relu(x)
    x = self.conv2(x)
    x = self.leaky_relu(x)
    return x
