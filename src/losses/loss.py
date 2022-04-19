import torch
import torch.nn as nn
from kornia import losses
import torch.nn.functional as F

class Gradient_Net(nn.Module):
  def __init__(self):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)
    kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)
    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

  def forward(self, x):
    grad_x = F.conv2d(x, self.weight_x)
    grad_y = F.conv2d(x, self.weight_y)
    return grad_x,grad_y

class Loss:
    
    @classmethod
    def mse_loss(cls,prediction, target):
        loss = F.mse_loss(prediction, target)
        return loss
    
    @classmethod
    def edge_loss(cls,y_pred,y_true):
      gradients = Gradient_Net()
      dy_true, dx_true = gradients(y_true)
      dy_pred, dx_pred = gradients(y_pred)
      l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))
      return l_edges
  
    @classmethod
    def depth_loss(cls,y_pred,y_true):
      loss = nn.L1Loss()
      l_depth = loss(y_pred,y_true)
      return l_depth
  
    @classmethod
    def ssim_loss(cls,y_pred,y_true):
      ssim_loss = losses.ssim_loss(y_pred, y_true, 5,10.0)
      return ssim_loss
  
    @classmethod
    def total_loss(cls,y_pred,y_true):
      edge_loss = cls.edge_loss(y_pred,y_true)
      depth_loss = cls.depth_loss(y_pred,y_true)
      ssim_loss = cls.ssim_loss(y_pred,y_true)
      mse_loss = cls.mse_loss(y_pred,y_true)
      total_loss = 1.0*edge_loss  + 0.1*depth_loss  + ssim_loss
      return total_loss
