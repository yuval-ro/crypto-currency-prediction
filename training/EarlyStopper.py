from typing import Literal

class EarlyStopper:
  patience: int  # Maximum number of epochs to pass before stopping
  delta: float  # Minimum threshold value required to be considered as an improvement
  best_loss: float
  counter: int

  def __init__(self, tolerance: Literal['low', 'med', 'high']):
    if tolerance == 'low':
      self.patience, self.delta = 8, 0.0100
    elif tolerance == 'med':
      self.patience, self.delta = 16, 0.0050
    else:  # tolerance == 'high'
      self.patience, self.delta = 24, 0.0025
    self.best_loss = float('inf')
    self.counter = 0

  def __call__(self, loss: float) -> bool:
    if loss < (self.best_loss - self.delta):
      self.best_loss = loss
      self.counter = 0
    else:
      self.counter += 1
      if self.counter == self.patience:
        return True  # trigger early stop
    return False
