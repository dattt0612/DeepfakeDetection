from torch.utils.data import Dataset
from transforms.base_transform import Transform


from typing import List, TypeVar, Callable, Dict, Any

Key = TypeVar("Key")

class BaseDataset(Dataset):
  def __init__(
    self,
    keys: List[Key],
    loaddata_fn: Callable[[Key], Dict[str, Any]],
    transform: Transform
  ):
    super().__init__()
    self.keys = keys
    self.loaddata_fn = loaddata_fn
    self.transform = transform
  
  def __len__(self) -> int:
    return len(self.keys)

  def __getitem__(self, index) -> Dict[str, Any]:
    key = self.keys[index]
    data: Dict[str, Any] = self.loaddata_fn(key)    

    if self.transform is not None:
      data = self.transform(data)
    
    return data