from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
from typing_extensions import override
from utils.get_device import GetDevice

class Transform(ABC, nn.Module, GetDevice):
  def __init__(self):
    ABC.__init__(self)
    nn.Module.__init__(self)
    GetDevice.__init__(self)

  @torch.no_grad()
  def __call__(self, x):
    return super().__call__(x)

  def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
    return self.apply(x)

  @abstractmethod
  def apply(self, x: Dict[str, Any]) -> Dict[str, Any]:
    pass

class ProbTransform(Transform):
  def __init__(self, p: float = 0.5):
    super().__init__()
    assert (0 <= p) and (p <= 1)
    self.p = p

  def is_first_apply(self) -> bool:
    return not hasattr("keys")

  def should_apply(self) -> bool:
    return self.is_first_apply() or torch.rand(1).item() < self.p

  @override
  def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
    if self.should_apply():
      x = self.apply(x)

      # handle add new key
      if self.is_first_apply():
        self.keys = set(x.keys())
      
      for k in self.keys:
        if k not in x:
          x[k] = None
      
      return x
      
class ProbBatchTransform(ProbTransform):
  def __init__(self, p: float = 0.5):
    super().__init__(p)
  
  def get_bs(self, x: Dict[str, Any]) -> int:
    first = next(iter(x.values()))
    return len(first)
  
  def get_applied_indices(self, bs: int):
    applied_mask = torch.rand(bs, device=self.device) < self.p

    if self.is_first_apply():
      applied_mask[0] = True

    applied_indices = applied_mask.nonzero(as_tuple=True)[0]
    return applied_indices

  def get_applied_samples(self, x: Dict[str, Any]):
    bs = self.get_bs(x)
    applied_indices = self.get_applied_indices(bs)
    applied_samples = {
      k: v[applied_indices]
      for k, v in x.items()
    }
    return applied_indices, applied_samples

  @override
  def forward(self, x):
    applied_indices, applied_samples = self.get_applied_samples(x)
    if len(applied_indices) == 0:
      return x
    
    applied_samples = self.apply(applied_samples)

    # handle add new key
    if self.is_first_apply():
      self.keys = applied_samples.keys()
    
    for k in self.keys:
      if k not in x:
        x[k] = None


    for k in x.keys():
      if torch.is_tensor(x[k]):
        x[k][applied_indices] = applied_samples[k]
      elif isinstance(x[k], list):
        for idx, val in zip(applied_indices, applied_samples[k]):
          x[k][idx] = val

    return x

class SequentialTransform(Transform):
  def __init__(self, *transforms: Sequence[Transform]) -> None:
    super().__init__()
    self.transforms = nn.ModuleList(transforms)
  
  @override
  def apply(self, x: Dict[str, Any]) -> Dict[str, Any]:
    for transform in self.transforms:
      x = transform(x)
    return x
  

    


