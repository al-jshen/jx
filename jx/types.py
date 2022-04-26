from typing import Any, Union, Sequence, TypeVar, Tuple, List
from typing_extensions import Protocol
from jax._src.prng import PRNGKeyArray
import numpy as np
import jax.numpy as jnp

Pytree = Any

PRNGKey = PRNGKeyArray

Axes = Union[int, Sequence[int]]

NdArray = Union[np.ndarray, jnp.ndarray]

T = TypeVar('T')

Shapes = Union[List[T], Tuple[T, ...], T]

class InitFn(Protocol):
  """A type alias for initialization functions.
  Initialization functions construct parameters for neural networks given a
  random key and an input shape. Specifically, they produce a tuple giving the
  output shape and a PyTree of parameters.
  """

  def __call__(
      self,
      rng: PRNGKey,
      input_shape: Shapes,
      **kwargs
  ) -> Tuple[Shapes, Pytree]:
    ...


class ApplyFn(Protocol):
  """A type alias for apply functions.
  Apply functions do computations with finite-width neural networks. They are
  functions that take a PyTree of parameters and an array of inputs and produce
  an array of outputs.
  """

  def __call__(
      self,
      params: Pytree,
      inputs: Pytree,
      *args,
      **kwargs
  ) -> Pytree:
    ...


StaxLayer = Tuple[InitFn, ApplyFn]
