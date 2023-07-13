### making data

import jax
from jax import numpy as jnp, random
from jax.random import PRNGKey
from numpyro.distributions import Bernoulli, BernoulliLogits
from numpyro import sample


def make_data_noconfounding(
    n=100, px=0.5, gt0=0.0, gxt=0.0, b0=0.0, bt=1.0, bx=1.0, btx=0.0
):
    """simulate data with no unobserved confounding, x = feature, t = treatment, y = outcome

    :n: numer of observations
    :px: marginal probability of x
    :gt0: intercept log odds of treatment
    :gxt: log odds ratio of x -> t
    :b0: intercept log odds of survival
    :bt: log odds ratio of t -> y
    :bx: log odds ratio of t -> y
    :btx: log odds interaction term between t and x for y
    :returns: dict with samples

    """
    
    x = sample("x", Bernoulli(px), sample_shape=(int(n),))
    t = sample("t", BernoulliLogits(gt0 + x * gxt))
    tx = t*x
    eta = b0 + bt * t + bx*x + btx*tx
    y = sample("y", BernoulliLogits(eta))
    # add intercept
    x0 = jnp.ones(y.shape)
    X = jnp.stack([x0, x, t, tx], axis=-1)

    return (X, y)

