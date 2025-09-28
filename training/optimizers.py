# https://pytorch.org/docs/stable/optim.html

from torch.optim import (
  Adadelta,
  Adagrad,
  Adam,
  AdamW,
  SparseAdam, # Lazy version of Adam algorithm suitable for sparse tensors.
  Adamax, # Adamax algorithm (a variant of Adam based on infinity norm).
  ASGD, # Averaged Stochastic Gradient Descent.
  LBFGS, # L-BFGS algorithm, heavily inspired by minFunc.
  NAdam,
  RAdam,
  RMSprop,
  Rprop, # Resilient backpropagation algorithm.
  SGD # Stochastic Gradient Descent
)