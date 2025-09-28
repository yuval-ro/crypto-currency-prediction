# https://pytorch.org/docs/stable/nn.functional.html

from torch.nn.functional import (
  binary_cross_entropy,
  binary_cross_entropy_with_logits,
  poisson_nll_loss,
  cosine_embedding_loss,
  cross_entropy,
  ctc_loss,
  gaussian_nll_loss,
  hinge_embedding_loss,
  kl_div,
  l1_loss,
  mse_loss,
  margin_ranking_loss,
  multilabel_margin_loss,
  multilabel_soft_margin_loss,
  multi_margin_loss,
  nll_loss,
  huber_loss,
  smooth_l1_loss,
  soft_margin_loss,
  triplet_margin_loss,
  triplet_margin_with_distance_loss
)