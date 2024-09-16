import torch as th
import numpy as np

def gather_nd(params, indices, batch_dim=1):
  """
   Source: https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445/37 - Kulbear Ji Yang

   A PyTorch porting of tensorflow.gather_nd
   This implementation can handle leading batch dimensions in params, see below for detailed explanation.

   The majority of this implementation is from Michael Jungo @ https://stackoverflow.com/a/61810047/6670143
   I just ported it compatible to leading batch dimension.

   Args:
   params: a tensor of dimension [b1, ..., bn, g1, ..., gm, c].
   indices: a tensor of dimension [b1, ..., bn, x, m]
   batch_dim: indicate how many batch dimension you have, in the above example, batch_dim = n.

   Returns:
   gathered: a tensor of dimension [b1, ..., bn, x, c].
   """
  batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
  batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
  c_dim = params.size()[-1]  # c
  grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
  n_indices = indices.size(-2)  # x
  n_pos = indices.size(-1)  # m

  # reshape leadning batch dims to a single batch dim
  params = params.reshape(batch_size, *grid_dims, c_dim)
  indices = indices.reshape(batch_size, n_indices, n_pos)

  # build gather indices
  # gather for each of the data point in this "batch"
  batch_enumeration = th.arange(batch_size).unsqueeze(1)
  gather_dims = [indices[:, :, i] for i in range(len(grid_dims))]
  gather_dims.insert(0, batch_enumeration)
  gathered = params[gather_dims]

  # reshape back to the shape with leading batch dims
  gathered = gathered.reshape(*batch_dims, n_indices, c_dim)
  return gathered

def is_finite(div):
  return not(th.any(th.isnan(div)) or th.any(th.isinf(div)))

def kl_div(x,y):
  """
  (Created according to the function in ssd-games of the same name)
  Calculate KL divergence between two distributions.
  :param x: A distribution
  :param y: A distribution
  :return: The KL-divergence between x and y. Returns zeros if the KL-divergence contains NaN
        or Infinity.
  """
  dist_x = th.distributions.categorical.Categorical(x)
  dist_y = th.distributions.categorical.Categorical(y)
  res = th.distributions.kl.kl_divergence(dist_x, dist_y)

  # Don't return nans or infs
  is_finite = th.all(th.isfinite(res))

  if not is_finite:
    res = th.zeros(th.Tensor.shape(res))

  return res
