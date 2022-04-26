import numpy as np
from jx.types import Axes, StaxLayer, NdArray
from typing import Sized, Union, List, Optional, Sequence, Iterable, Tuple
import functools
import operator

def _get_ndim(x: Union[int, Sized, NdArray]) -> int:
  """Get number of dimensions given number of dimensions / shape / array."""
  if hasattr(x, 'ndim'):
    n = x.ndim
  elif hasattr(x, '__len__'):
    n = len(x)
  elif isinstance(x, int):
    n = x
  else:
    raise TypeError(x, type(x))
  return n

def canonicalize_axis(axis: Axes,
                      x: Union[int, Sized, NdArray]) -> List[int]:
  """Converts axis into a sorted non-negative list.
  Args:
    axis: input axis.
    x: array / shape / number of dimensions.
  Returns:
    A sorted list of integer axes.
  """
  axis = [axis] if isinstance(axis, int) else list(axis)
  n = _get_ndim(x)
  return list(set(np.arange(n)[axis]))


def size_at(x: Union[NdArray, Sequence[int]],
            axes: Optional[Iterable[int]] = None) -> int:
  if hasattr(x, 'shape'):
    x = x.shape

  if axes is None:
    axes = range(len(x))

  return functools.reduce(operator.mul, [x[a] for a in axes], 1)

def mean_and_var(
    x: Optional[NdArray],
    axis: Optional[Axes] = None,
    dtype: Optional[np.dtype] = None,
    out: Optional[None] = None,
    ddof: int = 0,
    keepdims: bool = False,
    mask: Optional[NdArray] = None,
    get_var: bool = False
) -> Tuple[Optional[NdArray], Optional[NdArray]]:
  """`np.mean` and `np.var` taking the `mask` information into account."""
  var = None
  if x is None:
    return x, var

  if mask is None:
    mean = np.mean(x, axis, dtype, out, keepdims)
    if get_var:
      var = np.var(x, axis, dtype, out, ddof, keepdims)

  else:
    axis = tuple(canonicalize_axis(axis, x))
    size = size_at(x, axis)
    mask = np.broadcast_to(mask, x.shape)
    mask_size = np.count_nonzero(mask, axis)
    for i in axis:
      mask_size = np.expand_dims(mask_size, i)
    size -= mask_size
    size = np.maximum(size, 1)

    mean = np.sum(x, axis=axis, keepdims=True) / size
    if not keepdims:
      mean = np.squeeze(mean, axis)

    if get_var:
      var = np.sum((x - mean)**2, axis=axis, keepdims=True) / (size - ddof)
      if not keepdims:
        var = np.squeeze(var, axis)

  return mean, var


def LayerNorm(
    axis: Axes = -1,
    eps: float = 1e-12) -> StaxLayer:
  """Layer normalisation.
  Args:
    axis:
      dimensions over which to normalize.
    eps:
      (small) positive constant to be added to the variance estimates in order
      to prevent division by zero.
    batch_axis:
      batch dimension. Defaults to `0`, the leading axis.
    channel_axis:
      channel / feature dimension. Defaults to `-1`, the trailing axis. For
      `kernel_fn`, channel size is considered to be infinite.
  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def init_fn(_, input_shape):
    return input_shape, ()

  def apply_fn(_, inputs, mask=None, **kwargs):
    _axis = canonicalize_axis(axis, inputs)
    mean, var = mean_and_var(inputs, _axis, keepdims=True, mask=mask,
                             get_var=True)
    return (inputs - mean) / np.sqrt(eps + var)

  return init_fn, apply_fn


