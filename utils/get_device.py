import torch
import torch.nn as nn


class GetDevice(nn.Module):
  def __init__(self):
    super().__init__()
    self.d = torch.randn(1)
  
  @property
  def device(self):
    return self.d.device
  