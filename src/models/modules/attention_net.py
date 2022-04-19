import torch
import torch.nn as nn

class AttentionUnit(nn.Module):
    def __init__(self,in_channels):
        super(AttentionUnit,self).__init__()
        self.relu=nn.LeakyReLU()
        self.conv=nn.Conv2d(in_channels,1,kernel_size=1,stride=1,padding=0)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
      #x_=torch.add(x,g)
      x_=self.relu(x)
      x_=self.conv(x)
      x_=self.sigmoid(x)
      x=torch.mul(x,x_)
      return x