import torch
import numpy as np
from glob import glob
from datetime import datetime

def to_tensor(x: np.ndarray):
  if len(x.shape) > 1:
    tensor = torch.from_numpy(x)
    return tensor.view(tensor.shape[0], 1, -1)
  else:
    return torch.from_numpy(x)

def get_file_path(pattern: str):
  """
  Find the path of the most recent file in a local directory using a given pattern.
  """
  paths = glob(pathname=pattern)
  if paths == []:
    raise RuntimeError
  return paths[-1]

def timestamp():
  return datetime.now().strftime('%d%m%y%H%M%S')