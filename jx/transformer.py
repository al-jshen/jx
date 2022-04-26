from typing import Optional

import jax
import jax.nn as nn
import jax.numpy as jnp

from jx import attention
from jx.types import PRNGKey


class EncoderLayer:
    def __init__(
        self,
        model_size: int,
        num_heads: int,
        feedforward_size: int,
        dropout_rate=0.5,
        key: Optional[PRNGKey] = None,
    ):
        self.model_size = model_size
        self.hidden_size = feedforward_size
        self.key = key
        if self.key is None:
            self.key = jax.random.PRNGKey(0)

        k1, k2, k3, k4, k5 = jax.random.split(self.key, 5)
        self.mha = attention.MultiheadAttention(
            num_heads=num_heads,
            model_size=model_size,
            key=k1,
        )
        self.ffn = Sequential(
            [
                Dense(model_size, feedforward_size, activation=nn.relu, key=k2),
                Dense(feedforward_size, model_size, key=k3),
            ]
        )
        self.layernorm1 = LayerNorm(model_size, eps=1e-6)
        self.layernorm2 = LayerNorm(model_size, eps=1e-6)

        self.drop1 = Dropout(dropout_rate, key=k4)
        self.drop2 = Dropout(dropout_rate, key=k5)

    def __call__(self, x: jnp.ndarray, training=False, mask=None) -> jnp.ndarray:
        attn = self.mha(x, x, x, mask)
        attn = self.drop1(attn, training)
        attn = self.layernorm1(attn + x)

        ffn = self.ffn(attn)
        ffn = self.drop2(ffn, training)
        return self.layernorm2(ffn + attn)


def make_padding_mask(x: jnp.ndarray) -> jnp.ndarray:
    return (x == 0).astype(jnp.float32).reshape(x.shape[0], 1, 1, -1)


def make_look_ahead_mask(size: int) -> jnp.ndarray:
    return -jnp.tri(size) + 1.0


def make_positional_encodings(n: int, d: int) -> jnp.ndarray:
    div_term = jnp.exp(jnp.arange(0, d, 2) * -(jnp.log(10000.0) / d))
    position = jnp.arange(n).reshape(-1, 1)
    pos_div = position * div_term
    pe = jnp.zeros((n, d))
    pe = pe.at[:, 0::2].set(jnp.sin(pos_div))
    pe = pe.at[:, 1::2].set(jnp.cos(pos_div))
    return pe


class DecoderLayer:
    def __init__(
        self,
        model_size: int,
        num_heads: int,
        feedforward_size: int,
        key: Optional[PRNGKey] = None,
    ):
        self.model_size = model_size
        self.hidden_size = feedforward_size
        self.key = key
        if self.key is None:
            self.key = jax.random.PRNGKey(0)

        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(self.key, 7)
        self.mha1 = attention.MultiheadAttention(
            num_heads=num_heads,
            model_size=model_size,
            key=k1,
        )
        self.mha2 = attention.MultiheadAttention(
            num_heads=num_heads,
            model_size=model_size,
            key=k2,
        )
        self.ffn = Sequential(
            [
                Dense(model_size, feedforward_size, activation=nn.relu, key=k3),
                Dense(feedforward_size, model_size, key=k4),
            ]
        )
        self.layernorm1 = LayerNorm(model_size, eps=1e-6)
        self.layernorm2 = LayerNorm(model_size, eps=1e-6)
        self.layernorm3 = LayerNorm(model_size, eps=1e-6)

        self.drop1 = Dropout(0.1, key=k5)
        self.drop2 = Dropout(0.1, key=k6)
        self.drop3 = Dropout(0.1, key=k7)

    def __call__(
        self,
        x: jnp.ndarray,
        encoder_output: jnp.ndarray,
        training: bool = False,
        look_ahead_mask=None,
        padding_mask=None,
    ) -> jnp.ndarray:
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.drop1(attn1, training)
        attn1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(attn1, encoder_output, encoder_output, padding_mask)
        attn2 = self.drop2(attn2, training)
        attn2 = self.layernorm2(attn2 + attn1)

        ffn = self.ffn(attn2)
        ffn = self.drop3(ffn, training)
        return self.layernorm3(ffn + attn2)


class Encoder:
    def __init__(
        self,
        model_size: int,
        num_layers: int,
        num_heads: int,
        feedforward_size: int,
        input_vocab_size: int,
        maximum_position_encoding: int,
        dropout_rate: float = 0.5,
        key: Optional[PRNGKey] = None,
    ):
        self.model_size = model_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feedforward_size = feedforward_size
        self.key = key
        if self.key is None:
            self.key = jax.random.PRNGKey(0)

        self.embedding = Embedding(input_vocab_size, model_size)
        self.position_encoding = make_positional_encodings(
            maximum_position_encoding, model_size
        )

        self.encoding_layers = [
            EncoderLayer(
                model_size=model_size,
                num_heads=num_heads,
                feedforward_size=feedforward_size,
                dropout_rate=dropout_rate,
                key=k,
            )
            for k in jax.random.split(self.key, num_layers)
        ]

        self.dropout = Dropout(dropout_rate, key=jax.random.split(self.key)[1])

    def __call__(self, x: jnp.ndarray, training=False, mask=None) -> jnp.ndarray:
        seq_len = x.shape[1]
        x = self.embedding(x)
        x = x * jnp.sqrt(self.model_size)
        x = x + self.position_encoding[:seq_len, :]
        x = self.dropout(x, training)

        for layer in self.encoding_layers:
            x = layer(x, training, mask)

        return x


class Decoder:
    def __init__(
        self,
        model_size: int,
        num_layers: int,
        num_heads: int,
        feedforward_size: int,
        target_vocab_size: int,
        maximum_position_encoding: int,
        dropout_rate: float = 0.5,
        key: Optional[PRNGKey] = None,
    ):
        self.model_size = model_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key = key
        if self.key is None:
            self.key = jax.random.PRNGKey(0)

        self.embedding = Embedding(target_vocab_size, model_size)
        self.position_encoding = make_positional_encodings(
            maximum_position_encoding, model_size
        )

        self.decoding_layers = [
            DecoderLayer(
                model_size=model_size,
                num_heads=num_heads,
                feedforward_size=feedforward_size,
                key=k,
            )
            for k in jax.random.split(self.key, num_layers)
        ]

        self.dropout = Dropout(dropout_rate, key=jax.random.split(self.key)[1])

    def __call__(
        self,
        x: jnp.ndarray,
        encoder_output: jnp.ndarray,
        training=False,
        look_ahead_mask=None,
        padding_mask=None,
    ) -> jnp.ndarray:
        seq_len = x.shape[1]
        x = self.embedding(x)
        x = x * jnp.sqrt(self.model_size)
        x = x + self.position_encoding[:seq_len, :]
        x = self.dropout(x, training)

        for layer in self.decoding_layers:
            x = layer(x, encoder_output, training, look_ahead_mask, padding_mask)

        return x


class Transformer:
    def __init__(
        self,
        model_size: int,
        num_layers: int,
        num_heads: int,
        feedforward_size: int,
        input_vocab_size: int,
        target_vocab_size: int,
        input_position_encoding: int,
        target_position_encoding: int,
        dropout_rate: float = 0.5,
        key: Optional[PRNGKey] = None,
    ):
        self.key = key
        if self.key is None:
            self.key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(self.key, 3)
        self.encoder = Encoder(
            model_size=model_size,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            input_vocab_size=input_vocab_size,
            maximum_position_encoding=input_position_encoding,
            dropout_rate=dropout_rate,
            key=k1,
        )
        self.decoder = Decoder(
            model_size=model_size,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            target_vocab_size=target_vocab_size,
            maximum_position_encoding=target_position_encoding,
            dropout_rate=dropout_rate,
            key=k2,
        )
        self.final = Dense(model_size, target_vocab_size, activation=nn.relu, key=k3)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, training=False) -> jnp.ndarray:
        encoding_padding_mask = make_padding_mask(x)
        decoding_padding_mask = make_padding_mask(x)
        look_ahead_mask = make_look_ahead_mask(y.shape[1])
        decoder_target_padding_mask = make_padding_mask(y)
        look_ahead_mask = jnp.maximum(decoder_target_padding_mask, look_ahead_mask)

        encoder_output = self.encoder(x, training, encoding_padding_mask)
        decoder_output = self.decoder(
            y, encoder_output, training, look_ahead_mask, decoding_padding_mask
        )
        output = self.final(decoder_output)
        return nn.softmax(output, axis=-1)
