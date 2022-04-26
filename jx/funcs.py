import jax
import jax.numpy as jnp
from jx.types import NdArray


def stlogit(x: NdArray, a: float = 0., b: float = 1.):
    """
    Scaled and translated log-odds transform.

    Parameters:
    -----------
    x: float
        The value to transform.

    a: float
        The lower bound of the transformed range.

    b: float
        The upper bound of the transformed range.

    Returns:
    --------
    float
        The transformed value.
    """
    u = (x - a) / (b - a)
    return jnp.log(u / (1 - u))


def stexpit(x: NdArray, a: float = 0., b: float = 1.):
    """
    Scaled and translated inverse log-odds transform.

    Parameters:
    -----------
    x: float
        The value to transform.

    a: float
        The lower bound of the transformed range.

    b: float
        The upper bound of the transformed range.

    Returns:
    --------
    float
        The transformed value.
    """
    inv_logit_y = 1. / (1. + jnp.exp(-x))
    return a + (b - a) * inv_logit_y


