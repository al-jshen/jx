from functools import partial
from typing import Optional

import jax
import jax.nn as nn
import jax.numpy as jnp

from jx.types import PRNGKey


@jax.jit
def scaled_dot_product_attention(
    Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray, mask=None
) -> jnp.ndarray:
    """
    Compute scaled dot-product attention following Eq. 1 in Vaswani et al.
    (2017)

    - q shape: (..., seq_len_q, depth)
    - k shape: (..., seq_len_k, depth)
    - v shape: (..., seq_len_v, depth_v)
    """
    d_k = K.shape[-1]

    # (..., seq_len_q, seq_len_k)
    scores_scaled = jnp.einsum("...qd,...kd->...qk", Q, K) / jnp.sqrt(d_k)

    if mask is not None:
        scores_scaled += mask * -1e10

    # (..., seq_len_q, seq_len_k)
    weights = nn.softmax(scores_scaled, axis=-1)
    # (..., seq_len_q, depth_v)
    return weights @ V


class MultiheadAttention:
    def __init__(
        self,
        num_heads: int,
        model_size: int,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        key: Optional[PRNGKey] = None,
    ) -> None:
        assert model_size % num_heads == 0
        self.num_heads = num_heads
        self.model_size = model_size
        self.key_size = key_size or model_size // num_heads
        self.value_size = value_size or self.key_size
        self.key = key
        if self.key is None:
            self.key = jax.random.PRNGKey(0)

        assert self.value_size is not None

        k1, k2, k3, k4 = jax.random.split(self.key, 4)
        self.wq = Linear(self.model_size, self.model_size, key=k1)
        self.wk = Linear(self.model_size, self.model_size, key=k2)
        self.wv = Linear(self.model_size, self.model_size, key=k3)
        self.wo = Linear(self.model_size, self.model_size, key=k4)

    @partial(jax.jit, static_argnums=(0, 2))
    def split_heads(self, x: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        return x.reshape(batch_size, -1, self.num_heads, self.key_size).transpose(
            0, 2, 1, 3
        )

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self, Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray, mask=None
    ) -> jnp.ndarray:
        assert Q.shape[0] == K.shape[0] == V.shape[0]
        batch_size = Q.shape[0]

        q = self.wq(Q)
        k = self.wk(K)
        v = self.wv(V)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.model_size)

        return self.wo(attn)
