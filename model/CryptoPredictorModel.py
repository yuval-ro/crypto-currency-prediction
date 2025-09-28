import torch
from utils import timestamp, get_file_path
from data import CryptoCompareDataset

class CryptoPredictorModel(torch.nn.Module):
  """
  Hybrid model for predicting a time-series, comprised of LSTM and Linear Regression. 
  ([Source](https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/))
  """
  def __init__(self, dataset: CryptoCompareDataset):
    super(CryptoPredictorModel, self).__init__()
    self.in_features = dataset.x_size
    self.out_features = dataset.y_size
    self.HIDDEN_SIZE = 256
    self.NUM_LAYERS = 2
    # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    self.lstm = torch.nn.LSTM(
      self.in_features,
      self.HIDDEN_SIZE,
      self.NUM_LAYERS,
      batch_first=True
    )
    self.linear = torch.nn.Linear(
      in_features=self.HIDDEN_SIZE,
      out_features=self.out_features
    )

  def forward(self, x: torch.Tensor):
    h0 = torch.zeros(self.NUM_LAYERS, x.size(0), self.HIDDEN_SIZE).to(x.device)
    c0 = torch.zeros(self.NUM_LAYERS, x.size(0), self.HIDDEN_SIZE).to(x.device)
    out, _ = self.lstm(x, (h0, c0))
    out = out[:, -1, :]  # Get the last output from LSTM sequence
    out = self.linear(out)
    return out

  def predict(self, x: torch.Tensor):
    self.eval()
    with torch.no_grad():
      pred = self.forward(x)
    return pred

  def load(self):
    """Load a state dict from a local `.pth` file into the model.
    """
    pth_file = get_file_path('cpm_*.pth')
    self.load_state_dict(torch.load(pth_file))

  def save(self) -> None:
    """Save current model state dict as a local `.pth` file."""
    torch.save(
      obj=self.state_dict(),
      f=f'cpm_{timestamp()}.pth'
    )
