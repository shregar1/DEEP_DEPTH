import torch.nn as nn
from models.modules.densenet import (_DenseBlock, _Transition, Computation)


class DenseNets(nn.Module):
  def __init__(self,input_channels,encoder_arch):
    super(DenseNets,self).__init__()
    self.arch_parameters=self.find_arch_parameters(encoder_arch)
    self.in_channels = self.arch_parameters["initial_channels"]
    self.intermediate_channels = self.arch_parameters["intermediate_channels"]
    self.layers_lengths = self.arch_parameters["layers_length"]

    self.conv0 = nn.Conv2d(in_channels=input_channels,
                          out_channels=self.in_channels,kernel_size=7,
                          stride=2,padding=3,bias=False)
    self.norm0 = nn.BatchNorm2d(num_features=self.in_channels)
    self.relu0 = nn.ReLU()
    self.skip0_channels = self.in_channels

    self.maxpool0 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    self.skip1_channels = self.in_channels

    self.denseblock1 = _DenseBlock(in_channels=self.in_channels,
                                   out_channels=self.intermediate_channels,
                                   num_repeat=self.layers_lengths[0])
    self.in_channels = Computation.cal_channels(in_channels=self.in_channels,
                                                out_channels=self.intermediate_channels,
                                                num_repeat=self.layers_lengths[0])
    self.transition1 = _Transition(in_channels=self.in_channels,
                                   out_channels=self.in_channels//2)
    self.skip2_channels = self.in_channels//2
    
    self.denseblock2 = _DenseBlock(in_channels=self.in_channels//2,
                                   out_channels=self.intermediate_channels,
                                   num_repeat=self.layers_lengths[1])
    self.in_channels = Computation.cal_channels(in_channels=self.in_channels//2,
                                                out_channels=self.intermediate_channels,
                                                num_repeat=self.layers_lengths[1])
    self.transition2 = _Transition(in_channels=self.in_channels,
                                   out_channels=self.in_channels//2)
    self.skip3_channels = self.in_channels//2
    
    self.denseblock3 = _DenseBlock(in_channels=self.in_channels//2,
                                   out_channels=self.intermediate_channels,
                                   num_repeat=self.layers_lengths[2])
    self.in_channels = Computation.cal_channels(in_channels=self.in_channels//2,
                                                out_channels=self.intermediate_channels,
                                                num_repeat=self.layers_lengths[2])
    self.transition3 = _Transition(in_channels=self.in_channels,
                                   out_channels=self.in_channels//2)
    self.skip4_channels = self.in_channels//2
    
    self.denseblock4 = _DenseBlock(in_channels=self.in_channels//2,
                                   out_channels=self.intermediate_channels,
                                   num_repeat=self.layers_lengths[3])
    self.in_channels = Computation.cal_channels(in_channels=self.in_channels//2,
                                                out_channels=self.intermediate_channels,
                                                num_repeat=self.layers_lengths[3])
    
    self.norm5 = nn.BatchNorm2d(num_features=self.in_channels)

  def find_arch_parameters(self,encoder_arch):
    if encoder_arch == "densenet121":
      return {"initial_channels":64,
              "intermediate_channels":32,
              "layers_length":[6,12,24,16]
             }     
    elif encoder_arch == "densenet161":
      return {"initial_channels":96,
              "intermediate_channels":48,
              "layers_length":[6,12,36,24]
             }   
    elif encoder_arch == "densenet169":
      return {"initial_channels":64,
              "intermediate_channels":32,
              "layers_length":[6,12,32,32]
             }
    elif encoder_arch == "densenet201":
      return {"initial_channels":64,
              "intermediate_channels":32,
              "layers_length":[6,12,48,32]
             }
    elif encoder_arch == "densenet264":
      return {"initial_channels":64,
              "intermediate_channels":32,
              "layers_length":[6,12,64,48]
             }
    
  def forward(self,x):
    x = self.conv0(x)
    x = self.norm0(x)
    x0 = self.relu0(x)
    x1 = self.maxpool0(x0)
    x = self.denseblock1(x1)
    x2 = self.transition1(x)
    x = self.denseblock2(x2)
    x3 = self.transition2(x)
    x = self.denseblock3(x3)
    x4 = self.transition3(x)
    x = self.denseblock4(x4)
    x = self.norm5(x)
    return x0,x1,x2,x3,x4,x