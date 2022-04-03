import jax.random as random

__all__ = ['next_key']

JX_GLOBAL_KEY = random.PRNGKey(0)

def next_key():
    global JX_GLOBAL_KEY
    JX_GLOBAL_KEY, key = random.split(JX_GLOBAL_KEY)
    return key
