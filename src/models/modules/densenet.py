import torch
import torch.nn as nn
from collections import OrderedDict

class _DenseLayer(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(_DenseLayer,self).__init__()
    self.norm1 = nn.BatchNorm2d(num_features=in_channels)
    self.relu1 = nn.ReLU()
    self.conv1 = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels*4,
                          kernel_size=1,stride=1,
                          padding=0,bias=False)
    self.norm2 = nn.BatchNorm2d(num_features=out_channels*4)
    self.relu2 = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels=out_channels*4,
                          out_channels=out_channels,
                          kernel_size=3,stride=1,
                          padding=1,bias=False)
  
  def forward(self,x):
      y = self.norm1(x)
      y = self.relu1(y)
      y = self.conv1(y)
      y = self.norm2(y)
      y = self.relu2(y)
      y = self.conv2(y)
      x = torch.cat([x,y],dim=1)
      return x

class _DenseBlock(nn.Module):
  def __init__(self,in_channels,out_channels,num_repeat):
    super(_DenseBlock, self).__init__()
    self.denseblock = self.dense_block(in_channels,out_channels,num_repeat)
  
  def dense_block(self,in_channels,out_channels,num_repeat):
    dense_layers = []
    for i in range(1,num_repeat+1):
      dense_layers.append(("denselayer"+str(i),_DenseLayer(in_channels=in_channels,
                                    out_channels=out_channels)))
      in_channels+=out_channels
    return nn.Sequential(OrderedDict(dense_layers))

  def forward(self,x):
    return self.denseblock(x)

class _Transition(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(_Transition,self).__init__()
    self.norm = nn.BatchNorm2d(num_features=in_channels)
    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,stride=1,
                          padding=0,bias=False)
    self.pool = nn.AvgPool2d(kernel_size=2,stride=2)

  def forward(self,x):
    x=self.norm(x)
    x=self.relu(x)
    x=self.conv(x)
    x=self.pool(x)
    return x

class Global():
  def __init__(self):
    self.last_out_channels=None
global_ = Global()

class Computation():

  @classmethod
  def cal_channels(cls,in_channels,out_channels,num_repeat):
    for i in range(num_repeat):
      in_channels += out_channels
    return in_channels
  
