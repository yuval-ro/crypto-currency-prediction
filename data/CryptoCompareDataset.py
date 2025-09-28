import torch
import numpy as np
import sklearn.preprocessing as pp
from typing import Literal
from torch.utils.data import DataLoader, TensorDataset
import utils

class CryptoCompareDataset:
  """
  Stores dataset as `x, y` and exposes a conveniant API for data manipulation and scaling.
  """
  def __init__(
    self,
    symbol: str,
    to: str,
    x: np.ndarray,
    y: np.ndarray,
    which: Literal['neutral', 'train', 'test'] = 'neutral'
  ) -> None:
    """
    :param symbol: Cryptocurrency, e.g. BTC, ETH, ...
    :param to: Traditional currency, e.g. USD, EUR, ...
    :param x: Features
    :param y: Targets
    :param which: Type of dataset, defaults to 'neutral'
    """
    self.symbol = symbol
    self.to = to
    self.x = x
    self.y = y
    self.which = which
    # Use separate scalers for X and Y
    self.x_scaler = pp.MinMaxScaler()
    self.y_scaler = pp.MinMaxScaler()

  @property
  def x_size(self):
    return int(1 if len(self.x.shape) == 1 else self.x.shape[1])

  @property
  def y_size(self):
    return int(1 if len(self.y.shape) == 1 else self.y.shape[1])

  @property
  def x_scaled(self):
    # Only pass X to fit_transform
    x_scaled = self.x_scaler.fit_transform(self.x)
    return utils.to_tensor(x_scaled).float()

  @property
  def y_scaled(self):
    # Reshape y to 2D if needed, then scale
    y_reshaped = self.y.reshape(-1, 1) if len(self.y.shape) == 1 else self.y
    y_scaled = self.y_scaler.fit_transform(y_reshaped)
    return utils.to_tensor(y_scaled).float()

  def get_dataloader(self, batch_size: int):
    # Get scaled tensors
    x_tensor = self.x_scaled
    y_tensor = self.y_scaled
    
    # Squeeze y if it has an extra dimension
    if len(y_tensor.shape) == 3:
      y_tensor = y_tensor.squeeze(2)
    elif len(y_tensor.shape) == 2 and y_tensor.shape[1] == 1:
      y_tensor = y_tensor.squeeze(1)
    
    tds = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
      tds,
      batch_size,
      shuffle=(self.which == 'train')
    )

  def scale_x(self, x: np.ndarray):
    # Removed unused y parameter
    return self.x_scaler.transform(x)

  def scale_y(self, y: np.ndarray):
    y_reshaped = y.reshape(-1, 1) if len(y.shape) == 1 else y
    return self.y_scaler.transform(y_reshaped)

  def descale_x(self, x_scaled: torch.Tensor):
    # Convert to numpy and ensure 2D
    if isinstance(x_scaled, torch.Tensor):
      x_scaled = x_scaled.detach().cpu().numpy()
    if len(x_scaled.shape) == 3:
      x_scaled = x_scaled.squeeze(2)
    return self.x_scaler.inverse_transform(x_scaled)

  def descale_y(self, y_scaled: torch.Tensor):
    # Convert to numpy and ensure 2D
    if isinstance(y_scaled, torch.Tensor):
      y_scaled = y_scaled.detach().cpu().numpy()
    if len(y_scaled.shape) == 1:
      y_scaled = y_scaled.reshape(-1, 1)
    elif len(y_scaled.shape) == 3:
      y_scaled = y_scaled.squeeze(2)
    return self.y_scaler.inverse_transform(y_scaled)

  def descale_tensor(self, tensor: torch.Tensor, is_x: bool = True):
    # Specify which scaler to use
    if isinstance(tensor, torch.Tensor):
      arr = tensor.detach().cpu().numpy()
    else:
      arr = tensor
    
    if len(arr.shape) == 1:
      arr = arr.reshape(-1, 1)
    elif len(arr.shape) == 3:
      arr = arr.squeeze(2)
    
    scaler = self.x_scaler if is_x else self.y_scaler
    return scaler.inverse_transform(arr)

  def __repr__(self) -> str:
    return '\n'.join([
      'CryptoCompareDataset(',
      f'symbol={self.symbol}',
      f'to={self.to}',
      f'features={self.x}',
      f'target={self.y}',
      ')',
    ])

  def __eq__(self, other) -> bool:
    if not isinstance(other, CryptoCompareDataset):
      raise TypeError
    return all([
      self.symbol == other.symbol,
      self.to == other.to
    ])