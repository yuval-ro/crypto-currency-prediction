from training import Hyperparams, ModelTrainer, optimizers, loss
from data import DatasetGenerator
from model import Model

if __name__ == '__main__':
  # Obtain the data, then split it
  generator = DatasetGenerator('btc', 'usd')
  train_dataset, valid_dataset, test_dataset = generator.split()
  
  # Instanciate the model
  model = Model(generator.dataset)
  
  # Define training hyper parameters
  hp = Hyperparams(
    loss_fn=loss.mse_loss,
    epochs=100,
    batch_size=64,
    optimizer=optimizers.Adam,
    learn_rate=1e-2, # NOTE learning rate applies only to training and not to validation; May result in inflated training loss
    reg_factor=1e-3, 
    tolerance='high'
  )

  # Instanciate the trainer, then use it to train the model
  trainer = ModelTrainer(model)
  trainer.train(
    train_dataset,
    valid_dataset,
    hp,
    verbose=True,
    plot=True
  )
  
  # Test the model after training it
  trainer.test(test_dataset, 'pred.png')
  
  # Save the model after training
  trainer.model.save()
