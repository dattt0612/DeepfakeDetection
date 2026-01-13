from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
from typing_extensions import override
from utils.get_device import GetDevice

class Transform(ABC, nn.Module):
  def __init__(self):
    ABC.__init__(self)
    nn.Module.__init__(self)
    self.register_buffer("_device_tracker", torch.empty(0))
    
  @property
  def device(self):
    return self._device_tracker.device

  @torch.no_grad()
  def __call__(self, data):
    return super().__call__(data)

  def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
    return self.apply(data)

  @abstractmethod
  def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
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
  def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
    if self.should_apply():
      data = self.apply(data)

      # handle add new key
      if self.is_first_apply():
        self.keys = set(data.keys())
      
      for k in self.keys:
        if k not in data:
          data[k] = None
      
      return data
      
class ProbBatchTransform(ProbTransform):
  def __init__(self, p: float = 0.5):
    super().__init__(p)
  
  def get_bs(self, data: Dict[str, Any]) -> int:
    first = next(iter(data.values()))
    return len(first)
  
  def get_applied_indices(self, bs: int):
    applied_mask = torch.rand(bs, device=self.device) < self.p

    if self.is_first_apply():
      applied_mask[0] = True

    applied_indices = applied_mask.nonzero(as_tuple=True)[0]
    return applied_indices

  def get_applied_samples(self, data: Dict[str, Any]):
    bs = self.get_bs(data)
    applied_indices = self.get_applied_indices(bs)
    applied_samples = {
      k: v[applied_indices]
      for k, v in data.items()
    }
    return applied_indices, applied_samples

  @override
  def forward(self, data):
    applied_indices, applied_samples = self.get_applied_samples(data)
    if len(applied_indices) == 0:
      return data
    
    applied_samples = self.apply(applied_samples)

    # handle add new key
    if self.is_first_apply():
      self.keys = applied_samples.keys()
    
    for k in self.keys:
      if k not in data:
        data[k] = None


    for k in data.keys():
      if torch.is_tensor(data[k]):
        data[k][applied_indices] = applied_samples[k]
      elif isinstance(data[k], list):
        for idx, val in zip(applied_indices, applied_samples[k]):
          data[k][idx] = val

    return data

class SequentialTransform(Transform):
  def __init__(self, *transforms: Sequence[Transform]) -> None:
    super().__init__()
    self.transforms = nn.ModuleList(transforms)
  
  @override
  def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
    for transform in self.transforms:
      data = transform(data)
    return data
  

    


