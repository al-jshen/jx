import distrax
import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp
import jax.scipy.stats as stats
import optax
from jax import random
from functools import partial
from typing import Sequence, Mapping, Any

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


def diag_gaussian_sample(rng, mean, log_std):
    # Take a single sample from a diagonal multivariate Gaussian.
    return mean + jnp.exp(log_std) * random.normal(rng, mean.shape)


def diag_gaussian_logpdf(x, mean, log_std):
    # Evaluate a single point on a diagonal multivariate Gaussian.
    return jnp.sum(jax.vmap(stats.norm.logpdf)(x, mean, jnp.exp(log_std)))


def elbo(logprob, rng, mean, log_std):
    # Single-sample Monte Carlo estimate of the variational lower bound.
    sample = diag_gaussian_sample(rng, mean, log_std)
    return logprob(sample) - diag_gaussian_logpdf(sample, mean, log_std)


def batch_elbo(logprob, rng, params, num_samples):
    # Average over a batch of random samples.
    rngs = random.split(rng, num_samples)
    vectorized_elbo = jax.vmap(partial(elbo, logprob), in_axes=(0, None, None))
    return jnp.mean(vectorized_elbo(rngs, *params))


def make_conditioner(
    event_shape: Sequence[int], hidden_sizes: Sequence[int], num_bijector_params: int
) -> hk.Sequential:
    """Creates an MLP conditioner for each layer of the flow."""
    return hk.Sequential(
        [
            hk.Flatten(preserve_dims=-len(event_shape)),
            hk.nets.MLP(hidden_sizes, activate_final=True),
            # We initialize this linear layer to zero so that the flow is initialized
            # to the identity function.
            hk.Linear(
                np.prod(event_shape) * num_bijector_params,
                w_init=jnp.zeros,
                b_init=jnp.zeros,
            ),
            hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
        ]
    )


def make_flow_model(
    event_shape: Sequence[int],
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
) -> distrax.Transformed:
    """Creates the flow model."""
    # Alternating binary mask.
    mask = jnp.arange(0, np.prod(event_shape)) % 2
    mask = jnp.reshape(mask, event_shape)
    mask = mask.astype(bool)

    def bijector_fn(params: Array):
        return distrax.RationalQuadraticSpline(params, range_min=0.0, range_max=1.0)

    # Number of parameters for the rational-quadratic spline:
    # - `num_bins` bin widths
    # - `num_bins` bin heights
    # - `num_bins + 1` knot slopes
    # for a total of `3 * num_bins + 1` parameters.
    num_bijector_params = 3 * num_bins + 1

    layers = []
    for _ in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=make_conditioner(
                event_shape, hidden_sizes, num_bijector_params
            ),
        )
        layers.append(layer)
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)

    # We invert the flow so that the `forward` method is called with `log_prob`.
    flow = distrax.Inverse(distrax.Chain(layers))
    base_distribution = distrax.Independent(
        distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
        reinterpreted_batch_ndims=len(event_shape),
    )

    return distrax.Transformed(base_distribution, flow)
