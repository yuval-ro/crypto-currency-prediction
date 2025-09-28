import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .Hyperparams import Hyperparams
from model import Model
from data import Dataset

class Session:
  """
  Store training session data such as calculated loss and hyperparameters used.
  """
  def __init__(self,
     model: Model,
     hp: Hyperparams,
     train_ds: Dataset,
     valid_ds: Dataset
   ):
    self.model = model
    self.hp = hp
    self.train_ds = train_ds
    self.valid_ds = valid_ds
    self.train_loss = []
    self.valid_loss = []

  def plot(self, save_path=None):
    df = pd.DataFrame({
      'Training':   self.train_loss,
      'Validation': self.valid_loss,
    })
    df.plot(y=['Training', 'Validation'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    if save_path:
      plt.savefig(save_path)
      print(f'Plot saved to {save_path}')
    else:
      plt.show()
    plt.close()

class ModelTrainer:
  def __init__(self, model: Model):
    self.initial_model = model
    self.sessions: list[Session] = []

  def train(self,
    train_ds: Dataset,
    valid_ds: Dataset,
    hp: Hyperparams,
    verbose: bool = False,
    plot: bool = False,
    plot_path: str = None
  ):
    """
    Perform a single training session of this object's most recent model; Save the trained model.
    """
    sesh = Session(self.model, hp, train_ds, valid_ds)

    for epoch in range(sesh.hp.epochs):
      self._train(sesh)
      self._validate(sesh)

      if verbose:
        line = '{0:<8}{1:<24}{2:<24}'
        if epoch == 0:
          print(line.format('epoch', 'training loss', 'validation loss'))
        print(line.format((epoch + 1),
              sesh.train_loss[-1], sesh.valid_loss[-1]))

      if hp.has_stopper:
        if hp.stopper(loss=sesh.train_loss[-1]):  # early stopping triggered
          if verbose:
            print('early stopper triggered!')
          break

    self.sessions.append(sesh)
    if plot:
      sesh.plot(save_path=plot_path)

  def _train(self, sesh: Session):
    optimizer= sesh.hp.optimizer(sesh.model.parameters())
    dataloader = sesh.train_ds.get_dataloader(sesh.hp.batch_size)

    # Training loop
    sesh.model.train()
    loss_per_batch = []
    for inputs, labels in dataloader:
      optimizer.zero_grad()
      outputs = sesh.model.forward(inputs)
      loss = sesh.hp.loss_fn(input=outputs, target=labels)
      loss.backward()
      optimizer.step()
      loss_per_batch.append(loss.item())

    # Save progress
    sesh.train_loss.append(np.mean(loss_per_batch))

  def _validate(self, sesh: Session):
    loss_per_batch = []
    dataloader = sesh.valid_ds.get_dataloader(sesh.hp.batch_size)

    # Evaluation
    sesh.model.eval()
    with torch.no_grad():
      for inputs, labels in dataloader:
        outputs = sesh.model.forward(inputs)
        loss = sesh.hp.loss_fn(input=outputs, target=labels)
        loss_per_batch.append(loss.item())

    sesh.valid_loss.append(np.mean(loss_per_batch))

  def test(self, test_ds: Dataset, plot_image_path=None):
    """
    Test the prediction capabilities of the current model; Plot the results.
    """
    dataloader = test_ds.get_dataloader(1)

    self.model.eval()
    with torch.no_grad():
      actual_values = []
      predicted_values = []

      for inputs, labels in dataloader:
        predicted_scaled = self.model.predict(inputs)
        actual_scaled = labels  # Assuming labels are already in the original scale

        predicted = test_ds.descale_y(predicted_scaled)
        actual = test_ds.descale_y(actual_scaled)

        actual_values.append(actual.item())
        predicted_values.append(predicted.item())

      plt.figure(figsize=(10, 6))
      plt.plot(actual_values, label=f'Actual Price')
      plt.plot(predicted_values, label=f'Predicted Price')
      plt.xlabel('Day')
      plt.ylabel(f'Price ({test_ds.to.upper()})')
      plt.title(f'{test_ds.symbol.upper()} to {test_ds.to.upper()}')

      plt.legend()
      plt.tight_layout()
      if plot_image_path:
        plt.savefig(plot_image_path)
        print(f'Plot saved to {plot_image_path}')
      else:
        plt.show()
      plt.close()

  @property
  def model(self) -> Model:
    # Fixed: Return the model from the last session, or the initial model
    if self.sessions:
      return self.sessions[-1].model
    return self.initial_model