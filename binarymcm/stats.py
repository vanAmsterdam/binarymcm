"""
statistical functions as utility functions
"""

from jax import numpy as jnp, vmap
from jax.scipy.special import expit as sigmoid, logit
from jaxopt import Bisection
from jaxopt.objective import binary_logreg


# define some basic functions for logistic regression
loglik_yp = lambda y, p: y * jnp.log(p) + (1 - y) * jnp.log(1-p)
loglik_yeta  = lambda y, eta: loglik_yp(y,sigmoid(eta))
expected_loglik_qp = lambda p,p_hat: p * jnp.log(p_hat) + (1 - p) * jnp.log(1 - p_hat)
expected_loglik_qeta = lambda p,eta_hat: expected_loglik_qp(p, sigmoid(eta_hat))
expected_nll_qp = lambda p, p_hat: -1*expected_loglik_qp(p, p_hat)
expected_nll_qeta = lambda p, eta_hat: -1*expected_loglik_qeta(p, eta_hat)
odds = lambda p: p / (1-p)
invodds = lambda x: x / (1+x)


def calculate_ps(pt, px, pu_t0, pu_t1):
    """calculate joint probabilities from marginals and conditionals
    assumption: x \indep t, x \indep u

    :pt: TODO
    :px: TODO
    :pu_t0: TODO
    :pu_t1: TODO
    :returns: p[t,x,u] = p(t=t,x=x,u=u)

    """
    p000 = (1-pt)*(1-px)*(1-pu_t0)
    p001 = (1-pt)*(1-px)*pu_t0
    p010 = (1-pt)*px*(1-pu_t0)
    p011 = (1-pt)*px*pu_t0
    p100 = pt*(1-px)*(1-pu_t1)
    p101 = pt*(1-px)*pu_t1
    p110 = pt*px*(1-pu_t1)
    p111 = pt*px*pu_t1

    ps = jnp.array([[
            [p000,p001], # t,x,u
            [p010,p011]
            ],[
            [p100,p101],
            [p110,p111]
            ]])

    return ps


def solve_qs_from_or(q_marginal, odds_ratio, q_x):
    """given marginal q(y=1) and odds ratio OR(y|x=1, y|x=0), solve
    q(y|x=0), q(y|x=1)

    :q_marginal: marginal probability q(y=1)
    :odds_ratio: odds ratio OR(y|x=1,y|x=0)
    :q_x: marginal of x q(x=0)
    :returns: q(y|x=0), q(y|x=1)
    """
    a = odds_ratio * q_x - q_x - odds_ratio + 1
    b = -1 - q_marginal + q_marginal * odds_ratio + q_x - odds_ratio * q_x
    c = q_marginal
    q0_minus = (-b - jnp.sqrt(b**2 - 4*a*c)) / (2*a)
    q0_plus = (-b + jnp.sqrt(b**2 - 4*a*c)) / (2*a)

    q0 = jnp.where(jnp.logical_and(0 < q0_minus, q0_minus < 1.), q0_minus, q0_plus)
    q0 = jnp.where(odds_ratio == 1, q_marginal, q0)

    odds1 = odds(q0)*odds_ratio
    q1 = invodds(odds1)
    return q0, q1


def get_q3s_from_betas_and_gammas(betas, gamma, q_u):
    """get 8 'unmarginalized' probabilities q(y|t,x,u)
    given betas (marginalized versions) and q_u (marginal of u)

    :betas: b0, bx, bt, btx
    :gamma: g(t=0,x=0), g(t=0,x=1), g(t=1,x=0), g(t=1,x=1) (odds ratios OR(q(y=1|t,x,u=1),q(y=1|t,x,u=0)), or scalar if g00=g01=g10=g11=g
    :q_x: q(u=1)
    :returns: a0, at, ax, au, ...
    """
    b0, bx, bt, btx = betas
    qtx = parameters_to_probabilities2(b0=b0, bt=bt, bx=bx, btx=btx)

    # if gamma is calar repeat it
    gammas = jnp.where(gamma.size==4, gamma, jnp.ones(4) * gamma)
    q000, q001 = solve_qs_from_or(qtx[0,0], gammas[0], q_u)
    q010, q011 = solve_qs_from_or(qtx[0,1], gammas[1], q_u)
    q100, q101 = solve_qs_from_or(qtx[1,0], gammas[2], q_u)
    q110, q111 = solve_qs_from_or(qtx[1,1], gammas[3], q_u)

    qs = jnp.array([[
            [q000,q001],
            [q010,q010]
            ],[
            [q100,q101],
            [q110,q110]
            ]])

    return qs


def offset_logreg3(param, data, offset):
    """
    logistic regression where one parameter is fixed to a pre-specified 'offset' value
    NOTE: last column of X MUST be where the offset is applied
    """
    param2 = jnp.append(param, offset)
    return binery_logreg(param2, data)


def marginalized_or_from_probabilities(px, qs):
    """
    calculate marginalized odds-ratio from probabilities

    :p: marginal probability of x (which is marginalized out)
    :qs: Pr(y=1|t=t,x=x)=qs[t,x]

    returns: logit(Pr(y=1|t=1)) - logit(Pr(y=1|t=0))
    """
    q00 = qs[0,0]
    q01 = qs[0,1]
    q10 = qs[1,0]
    q11 = qs[1,1]

    # marginalize over x:
    q0 = (1 - px) * q00 + px * q01
    q1 = (1 - px) * q10 + px * q11

    return logit(q1) - logit(q0)




def marginalized_or_from_parameters(px, b0, bx, bt, btx, **kwargs):
    """
    calculate marginalized odds ratio from parameters
    """
    # Pr(y=1|t,x)
    qs = parameters_to_probabilities2(b0=b0,bt=bt,bx=bx,btx=btx)

    return marginalized_or_from_probabilities(px, qs)





def marginalize_odds_ratio4(W, x):
    """
    given parameter estimates W and feature x, return marginalized odds ratio using empirical distribution
    includes interaction term
    """
    # copy of X with treatment 0 and 1 (treatment is last column of X)
    X = x.reshape(-1, 1)
    N = X.shape[0]
    intercept = jnp.ones((N, 1))

    # fixed treatment terms
    t0 = jnp.zeros((N, 1))
    t1 = jnp.ones((N, 1))

    # interaction terms
    tx0 = X * t0
    tx1 = X * t1

    # design matrices
    X0 = jnp.hstack([intercept, X, t0, tx0])
    X1 = jnp.hstack([intercept, X, t1, tx1])

    # predict outcomes
    p0_hats = sigmoid(jnp.dot(X0, W))
    p1_hats = sigmoid(jnp.dot(X1, W))

    # marginalize predictions
    p0 = jnp.mean(p0_hats)
    p1 = jnp.mean(p1_hats)

    return logit(p1) - logit(p0)


def marginalize_odds_ratio3(W, x):
    """
    given parameter estimates W and feature x, return marginalized odds ratio using empirical distribution
    has no interaction term
    """
    # copy of X with treatment 0 and 1 (treatment is last column of X)
    X = x.reshape(-1, 1)
    N = X.shape[0]
    intercept = jnp.ones((N, 1))

    # fixed treatment terms
    t0 = jnp.zeros((N, 1))
    t1 = jnp.ones((N, 1))

    # design matrices
    X0 = jnp.hstack([intercept, X, t0])
    X1 = jnp.hstack([intercept, X, t1])

    # predict outcomes
    p0_hats = sigmoid(jnp.dot(X0, W))
    p1_hats = sigmoid(jnp.dot(X1, W))

    # marginalize predictions
    p0 = jnp.mean(p0_hats)
    p1 = jnp.mean(p1_hats)

    return logit(p1) - logit(p0)

def H4_expectation(param, px, gamma, **kwargs):
    """
    constraint function for 4-parameter model, not using data but data generating mechanism
    """
    gamma_hat = marginalized_or_from_parameters(px, *param)
    return gamma_hat - gamma


def H3_expectation(param, px, gamma, **kwargs):
    """
    constraint function for 3-parameter model (no interaction term), not using data but data generating mechanism
    """
    param4 = jnp.append(param, jnp.zeros(1))
    return H4_expectation(param4, px, gamma, **kwargs)



def H3(params, data, gamma):
    """
    constraint function for 3-parameter model (no interaction term)
    """
    Xmat, _ = data
    # x covariate is the second column in data
    x = jnp.take(Xmat, 1, 1)
    gamma_hat = marginalize_odds_ratio3(params, x)
    return gamma_hat - gamma


def H4(params, data, gamma):
    """
    constraint function for 3-parameter model (no interaction term)
    """
    Xmat, _ = data
    # x covariate is the second column in data
    x = jnp.take(Xmat, 1, 1)
    gamma_hat = marginalize_odds_ratio4(params, x)
    return gamma_hat - gamma


def make_H(sim_prms, interaction_term=True):
    """
    create constraint function
    """
    gamma_t = marginalized_or_from_parameters(**sim_prms)

    if interaction_term:
        marginalizer = marginalize_odds_ratio4
    else:
        marginalizer = marginalize_odds_ratio3

    def H(params, data):
        Xmat, _ = data
        x = Xmat[:, 1]
        gamma_hat = marginalizer(params, x)
        return gamma_hat - gamma_t

    return H


def calculate_deltas(b0, bx, bt, btx):
    """
    given parameter estimates, calculate estimated differences in outcome under treatments
    """
    p00 = sigmoid(b0)
    p01 = sigmoid(b0 + bx)
    p10 = sigmoid(b0 + bt)
    p11 = sigmoid(b0 + bt + bx + btx)

    # deltas
    delta0 = p10 - p00
    delta1 = p11 - p01

    return delta0, delta1

def calculate_pehe2(param, sim_prms):
    """
    given parameters for data generating mechanism and estimated parameters,
    calculate precision in heterogenenous treatment effect estimation (squared), pehe
    """
    delta0, delta1 = calculate_deltas(**{k: v for k, v in sim_prms.items() if k in ['b0', 'bx', 'bt', 'btx']})
    delta0_hat, delta1_hat = calculate_deltas(*param)

    # pehe2
    px = sim_prms['px']
    pehe2 = (1-px) * (delta0 - delta0_hat) ** 2 + px * (delta1 - delta1_hat) ** 2

    return pehe2


def parameters_to_probabilities2(b0, bt, bx, btx, **kwargs):
    """
    given parameter vector (b0, bt, bx, btx),
    return a 2-dim array of probabilities qy[t,x]
    """
    eta00 = b0
    eta01 = b0+bx
    eta10 = b0+bt
    eta11 = b0+bt+bx+btx
    etas = jnp.array([
            [eta00, eta01], # t=0,x=0
            [eta10, eta11], # t=0,x=1
            ])

    return sigmoid(etas)


def parameters_to_probabilities3(a0, at, ax, au, atx, atu, axu, atxu):
    """
    given parameter vector (a0, at, ax, au, atx, atu, axu, atxu),
    return a 3-dim array of probabilities qy[t,x,u]
    """
    eta000 = a0
    eta001 = a0+au
    eta010 = a0+ax
    eta011 = a0+ax+au
    eta100 = a0+at
    eta101 = a0+at+au+atu
    eta110 = a0+at+ax+atx
    eta111 = a0+at+ax+au+atx+atu+axu+atxu
    etas = jnp.array([[
            [eta000, eta001], # t=0,x=0
            [eta010, eta011], # t=0,x=1
            ],[
            [eta100, eta101], # t=1,x=0
            [eta110, eta111], # t=1,x=1
            ]])

    return sigmoid(etas)



def marginalized_offset_criterion(atx, pu, a0, at, ax, au, atu, axu, atxu):
    """
    given parameters of a distribution, check if the offset criterion holds
    """
    # get beta parameters by marginalizing out u
    betas = alphas_to_betas(pu_t0=pu, pu_t1=pu, a0=a0, at=at, ax=ax, au=au, atx=atx, atu=atu, axu=axu, atxu=atxu)

    # the last parameter is the interaction parameter and this should be zero
    return betas[-1]

    
def find_atx(pu, a0, at, ax, au, atu, axu, atxu):
    """
    given parameters of a distribution, find an interaction term to make the marginalized model (marginalized over u) 'linear logistic', meaning no interaction between t and x
    """
    crit = lambda atx: marginalized_offset_criterion(pu, a0, at, ax, au, atx, atu, axu, atxu)

    # first find the correct limits to search between
    pows = jnp.array((6, 5, 4, 3, 2, 1, .5, .1, 0))
    uppers = 10**pows
    lowers = -uppers

    def _check_lims(lower, upper):
        # check if the limits have equal signs
        # they may not have due to sigmoid function saturating at high values and getting rounding errors that dont lead to opposite signs
        vl = crit(lower)
        vu = crit(upper)
        return (1 - jnp.sign(vl) * jnp.sign(vu)) / 2


    lim_ok = vmap(_check_lims)(lowers, uppers)
    lim_idx = jnp.min(jnp.where(lim_ok)[0])

    bisec = Bisection(crit, lower = lowers[lim_idx], upper=uppers[lim_idx])
    
    return bisec.run().params

def marginalize_u_from_probabilities(pu_t0, pu_t1, q3s):
    """
    given parameters, return probabilities for u marginalized out
    assumes u indep x | t
    return: array, q[t,x] = p(y|t,x)
    """
    # qtxu = p(y|t,x,u)
    q000 = q3s[0,0,0]
    q001 = q3s[0,0,1]
    q010 = q3s[0,1,0]
    q011 = q3s[0,1,1]
    q100 = q3s[1,0,0]
    q101 = q3s[1,0,1]
    q110 = q3s[1,1,0]
    q111 = q3s[1,1,1]

    # marginalize out for t=0
    q00 = (1-pu_t0) * q000 + pu_t0 * q001
    q01 = (1-pu_t0) * q010 + pu_t0 * q011
    # marginalize out for t=1
    q10 = (1-pu_t1) * q100 + pu_t1 * q101
    q11 = (1-pu_t1) * q110 + pu_t1 * q111

    # setup q2
    q2 = jnp.array([
            [q00, q01], # p(y|t=0,x=0), p(y|t=0,x=1)
            [q10, q11], # p(y|t=1,x=0), p(y|t=1,x=1)
            ])

    return q2


def marginalize_u_from_parameters(pu_t0, pu_t1, a0, at, ax, au, atx, atu, axu, atxu):
    """
    given parameters, return probabilities for u marginalized out
    assumes u indep x | t
    return: array, q[t,x] = p(y|t,x)
    """
    # get probabilities
    q3s = parameters_to_probabilities3(a0, at, ax, au, atx, atu, axu, atxu)

    return marginalize_u_from_probabilities(pu_t0, pu_t1, q3s)


def probabilities_to_parameters2(qs):
    """from probabilities p(y|t,x) = q[t,x] go to betas such that
    p(y|t,x) = sigmoid(b0 + bt t + bx x + btx tx)

    :qs: TODO
    :returns: TODO

    """
    b0 = logit(qs[0,0])
    bx = logit(qs[0,1]) - logit(qs[0,0])
    bt = logit(qs[1,0]) - logit(qs[0,0])
    btxu = logit(qs[1,1]) - (b0 + bx + bt)
    return b0, bx, bt, btxu


def alphas_to_betas(pu_t0, pu_t1, a0, at, ax, au, atx, atu, axu, atxu):
    """marginalize logistic regression parameters p(y|t,x,u) = sigmoid(a0 + at*t + ...)
    to logistic regression parameters p(y|t,x) = (1-pu_t0)p(y|t,x,0) + pu_t0 p(y|t,x,1)
    assumes u indep x | t

    :pu_t0: TODO
    :pu_t1: TODO
    :a0: TODO
    :at: TODO
    :ax: TODO
    :au: TODO
    :atx: TODO
    :atu: TODO
    :axu: TODO
    :atxu: TODO
    :returns: TODO

    """
    qs = marginalize_u_from_parameters(pu_t0, pu_t1, a0, at, ax, au, atx, atu, axu, atxu)
    return probabilities_to_parameters2(qs)

def L_logistic_3param(param, qs, pt, px, **kwargs):
    """
    get expected log likelihood given parameters and qs for model y|t,x1
    assumption: t \indep x

    :param: three parameter model: b0, bx, bt (no interaction between t and x)
    :qs: outcome probabilities p(y|t,x) = qs[t,x]
    :pt: marginal probability of t
    :px: marginal probability of x
    :**kwargs: these may go into the optinal constraint but are ignored here
    """
    param4 = jnp.append(param, jnp.zeros(1))

    return L_logistic_4param(param4, qs, pt, px, **kwargs)


def L_logistic_4param(param, qs, pt, px, **kwargs):
    """
    get expected log likelihood given parameters and qs for model y|t,x1
    assumption: t \indep x

    :param: four parameter model: b0, bx, bt, btx
    :qs: outcome probabilities p(y|t,x) = qs[t,x]
    :pt: marginal probability of t
    :px: marginal probability of x
    :**kwargs: these may go into the optinal constraint but are ignored here
    """
    b0, bx, bt, btx = param

    eta00 = b0
    eta01 = b0+bx
    eta10 = b0+bt
    eta11 = b0+bt+bx+btx

    ll_x0 = (1-pt)*expected_nll_qeta(qs[0,0], eta00) + pt * expected_nll_qeta(qs[1,0], eta10)
    ll_x1 = (1-pt)*expected_nll_qeta(qs[0,1], eta01) + pt * expected_nll_qeta(qs[1,1], eta11)

    return (1-px)*ll_x0 + px*ll_x1


def L_logistic_txu(qhats, qs, ps, **kwargs):
    """
    get expected log likelihood given parameters and qs for model y|t,x,u
    assumption: t \indep x, u \indep x

    :qhats: estimated outcome probabilities p(y|t,x,u) = qhats[t,x,u]
    :qs: actual outcome probabilities \Pr(y|t,x,u) = qs[t,x,u]
    :ps: probabilities \Pr(t=t,x=x,u=u) = ps[t,x,u]
    :**kwargs: these may go into the optinal constraint but are ignored here
    """
    lls = vmap(lambda p, q, q_hat: p * expected_nll_qp(q, q_hat))(ps, qs, qhats)
    return jnp.sum(lls)


def L_logistic_4param_txu(param, qs, ps, **kwargs):
    b0, bx, bt, btx = param
    qhats_tx = parameters_to_probabilities2(b0=b0, bx=bx, bt=bt, btx=btx)
    # these models dont condition on u so copy for u on new axis
    qhats_txu = jnp.stack([qhats_tx, qhats_tx], axis=-1)
    return L_logistic_txu(qhats_txu, qs, ps)


def L_logistic_3param_txu(param, qs, ps, **kwargs):
    param2 = jnp.append(param, jnp.zeros(1))
    return L_logistic_4param_txu(param2, qs, ps, **kwargs)


def L_offset_logreg3(param, qs, pt, px, offset):
    """
    expecte log like for offset model
    """
    param2 = jnp.append(param, offset)
    return L_logistic_3param(param2, qs, pt, px)


def L_offset_logreg3_txu(param, qs, ps, offset, **kwrags):
    """
    expected log like for offset 2-parameter offset model, outcomes conditional on t,x,u
    """
    param2 = jnp.append(param, offset)
    return L_logistic_3param_txu(param2, qs, ps)





