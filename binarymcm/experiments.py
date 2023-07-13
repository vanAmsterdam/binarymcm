import jax
# jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp, random, vmap, lax
from jax.random import PRNGKey
from jax.scipy.special import logit
import numpy as np
from scipy.special import logit as scipy_logit
from numpyro.handlers import seed
from jaxopt import LBFGS
from jaxopt.objective import binary_logreg

from binarymcm.minlag import MinimumLagrangian
from binarymcm.makedata import make_data_noconfounding
from binarymcm.stats import *

import pandas as pd
import itertools

from pathlib import Path

tol = 1e-4

from functools import wraps
import time

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('experiment', type=str, default='efficiency', help='name of argument to run')
parser.add_argument('--p000', type=float, default=None, help='Pr(Y=1|t=0,x=0,u=0), used in full grid experiment')
parser.add_argument('--enforce_offset', type=int, default=None, help='enforce_offset, used in full grid experiment')
parser.add_argument('--eargs', nargs='*', help='args that go into the experiment')
parser.add_argument('--gpu', type=int, help='gpu index to use')


def experiment_full(setting_grid):
    """
    experiment on effect of confounding and non-collapsibility
    """
    @jax.jit
    def run_setting(setting_prms):
        ## initialize optimizers
        # initialize atx finder
        bisec = Bisection(marginalized_offset_criterion, lower=-10., upper=10., check_bracket=False)

        # initialize_parameters (3 for constrained offset)
        w_init2 = jnp.zeros(2)
        w_init3 = jnp.zeros(3)
        w_init4 = jnp.zeros(4)

        # offset model
        lbfgs_offset = LBFGS(L_offset_logreg3_txu, tol=1e-4)

        # constrained offset approach
        lagopt3 = MinimumLagrangian(L_logistic_3param_txu, H3_expectation, LBFGS, tol=1e-4, maxiter=6, unroll=True,
                                   inner_kwargs=dict(maxiter=20, unroll=True))
        lagopt4 = MinimumLagrangian(L_logistic_4param_txu, H4_expectation, LBFGS, tol=1e-4, maxiter=6, unroll=True,
                                   inner_kwargs=dict(maxiter=20, unroll=True))
        # NOTE: found on effect of unrolling vs not on computation time for 20e3 experiments
        
        # fetcher functions for atx (offset or dummy)
        def fetch_atx_dummy(pu, a0,at,ax,au,atx,atu,axu,atxu):
            return atx
        def fetch_atx_offset(pu, a0,at,ax,au,atx,atu,axu,atxu):
            atx_offset = bisec.run(0.0, pu=pu,a0=a0,at=at,ax=ax,au=au,atu=atu,axu=axu,atxu=atxu).params
            return atx_offset

        # unpack setting parameters
        a0 = setting_prms['a0']
        at = setting_prms['at']
        ax = setting_prms['ax']
        au = setting_prms['au']
        atx = setting_prms['atx']
        atu = setting_prms['atu']
        axu = setting_prms['axu']
        atxu = setting_prms['atxu']

        # get ors
        or_uy = jnp.exp(au)
        or_ut = jnp.exp(setting_prms['gamma_tu'])

        # marginals
        px = setting_prms['px']
        pu = setting_prms['pu']
        pt = setting_prms['pt']

        # flags
        enforce_offset = setting_prms['enforce_offset']
        # done unpacking

        # find the atx such that the offset assumption is satifsied (or not)
        atx = lax.cond(enforce_offset, fetch_atx_offset, fetch_atx_dummy,
                       pu,a0,at,ax,au,atx,atu,axu,atxu)
                       

        # calculate marginal odds ratio as from rct
        # calculate the u-conditional probabilities based on gamma_u (odds(y|t,x,u=1) / odds(y|t,x,u=0))
        q3 = parameters_to_probabilities3(a0=a0,at=at,ax=ax,au=au,atx=atx,atu=atu,axu=axu,atxu=atxu)

        # get p(u=1|t=0) and p(u=1|t=1) from p(u=1), odds(u|t=1) / odds(u|t=0) = gamma, p(t=1)
        pu_t0, pu_t1 = solve_qs_from_or(pu, or_ut, pt)

        # interventional bts; NOTE USE MARGINAL PU and not pu_t here as would be used in RCT
        b0, bx, bt, btx = alphas_to_betas(pu, pu, a0=a0,at=at,ax=ax,au=au,atx=atx,atu=atu,axu=axu,atxu=atxu)
        beta_dict = dict(b0=b0, bx=bx, bt=bt, btx=btx, px=px)
        gt = marginalized_or_from_parameters(px=px, b0=b0, bx=bx, bt=bt, btx=btx)

        # calculate the probability of p(t=t,x=x,u=u)
        ## obs, where u not indep t
        ps_obs = calculate_ps(pt, px, pu_t0, pu_t1)
        ## rct, where u indep t
        ps_rct = calculate_ps(pt, px, pu, pu)

        # get the results
        ## offset
        w_offset, _ = lbfgs_offset.run(w_init2, qs=q3, ps=ps_obs, offset=gt)
        ## run constrained estimator
        w_constrained3, _ = lagopt3.run(w_init3, qs=q3, ps=ps_obs, px=px, gamma=gt)
        w_constrained4, _ = lagopt4.run(w_init4, qs=q3, ps=ps_obs, px=px, gamma=gt)

        # get the ATE pehe
        delta0, delta1 = calculate_deltas(b0=b0, bx=bx, bt=bt, btx=btx)
        delta_ate = (1-px)*delta0 + px*delta1
        pehe2_ate = (1-px)*(delta0 - delta_ate)**2 + px * (delta1 - delta_ate) ** 2

        # calculate pehe2s
        ## add back in the interaction parameter for calculating pehe2
        w_offset2 = jnp.append(w_offset, jnp.array((gt, 0.0)))
        w_constrained32 = jnp.append(w_constrained3, jnp.array(0.0))

        pehe2_offset = calculate_pehe2(w_offset2, beta_dict)
        pehe2_constrained3 = calculate_pehe2(w_constrained32, beta_dict)
        pehe2_constrained4 = calculate_pehe2(w_constrained4, beta_dict)

        # calculate expected log likelihoods of marginalized models
        ## full model
        ll_full = L_logistic_txu(q3, q3, ps_rct)

        ## marginalize over u (u -> y effect measure)
        qs_tx = marginalize_u_from_probabilities(pu, pu, q3)
        qhats_tx = jnp.stack([qs_tx, qs_tx], axis=-1)
        ll_tx = L_logistic_txu(qhats_tx, q3, ps_rct)

        ## marginalize over x (for x -> y effect measure)
        qs_tu = (1-px) * jnp.take(q3, 0, 1) + px * jnp.take(q3, 1, 1)
        qhats_tu = jnp.stack([qs_tu, qs_tu], axis=-1)
        ll_tu = L_logistic_txu(qhats_tu, q3, ps_rct)

        ## calculate min / max probs (to see if it crosses the symmetry point of the sigmoid which can give the ATE baseline an 'artificially' good CATE because CATE is constant though bx non-zero due to symmetry
        qs2 = parameters_to_probabilities2(b0=b0, bx=bx, bt=bt, btx=btx)
        q2_min = jnp.min(qs2)
        q2_max = jnp.max(qs2)


        ## prep return value
        lls = (ll_full, ll_tx, ll_tu)
        pehe2s = (pehe2_ate, pehe2_offset, pehe2_constrained3, pehe2_constrained4)
        q_lims = (q2_min, q2_max)
        # iter_counters = (state_constrained.iter_num, jnp.max(state_constrained.inner_iter_num))


        return pehe2s, lls, q_lims, btx

    # run this over all settings
    def runner(carry, setting_prms):
        return carry, run_setting(setting_prms)
    results = vmap(run_setting)(setting_grid)
    # _, results = lax.scan(runner, None, setting_grid)

    return results


def make_grid_full(p000=None, enforce_offset=None):
    """
    make parameter grid for full experiment (confounding and collapsibility) using alphas
    """
    # define sequences for alphas and gammas
    pos_num = 5
    pos_seq = np.linspace(np.log(1), np.log(5), num=pos_num)
    posneg_seq = np.linspace(np.log(1/5), np.log(5), num=pos_num+pos_num+1)

    # sequences for marginals
    mseq = np.array([.2, .5])
    # mseq = jnp.array([.5])
    # a0 = logit(jnp.array([0.025, .15, .5, .85, .975]))
    if p000 is not None:
        a0 = scipy_logit(np.array([float(p000)]))
    else:
        a0 = scipy_logit(np.array([0.025, .15, .5]))

    at = pos_seq[1:]
    ax = pos_seq
    au = pos_seq
    # atx = np.array([0])
    atx = pos_seq
    atu = pos_seq
    axu = pos_seq
    atxu = pos_seq

    # t -> u
    gamma_tu = posneg_seq

    # marginals
    pu = mseq
    px = mseq
    pt = mseq

    # flags
    if enforce_offset is None:
        enforce_offset = np.array([True, False])
    else: 
        enforce_offset = np.array([bool(enforce_offset)])

    setting_combinations = np.asarray(list(itertools.product(
            a0, at, ax, au, atx, atu, axu, atxu,
            gamma_tu,
            pu, px, pt,
            enforce_offset
            )))
    
    grid_df = pd.DataFrame(setting_combinations, columns=[
            'a0', 'at', 'ax', 'au', 'atx', 'atu', 'axu', 'atxu',
            'gamma_tu',
            'pu', 'px', 'pt',
            'enforce_offset'
            ])
    # remove combinations where enforce_offset = 1 and atx != 0 as this overrides atx
    grid_df = grid_df.astype(np.float32)
    grid_df = grid_df.loc[(grid_df.enforce_offset==0.0)|(grid_df.atx==0.0)]

    grid = grid_df.to_dict(orient='list')
    grid = {k: np.asarray(v) for k, v in grid.items()}

    return grid_df, grid


def run_grid_full(p000=None, enforce_offset=None):
    outdir = Path('..') / 'results' / 'fullgrid'
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=False)

    # make filename, first check existence of other experiment results
    resfiles = list(outdir.glob("settingdf*"))
    resfile_counters = [x.name.replace("settingdf","").replace(".csv", "") for x in resfiles]
    resfile_counters = [int(x) for x in resfile_counters]
    if len(resfile_counters) > 0:
        resfile_counter = int(max(resfile_counters) + 1)
    else:
        resfile_counter = int(0)
    outname = f"settingdf{resfile_counter}.csv"
    outpath = outdir / outname
    assert not outpath.exists()
    # make the file to make sure no parallel process will write to the same output file
    outpath.touch()

    # get the grid and save
    setting_df, setting_grid = make_grid_full(p000, enforce_offset)
    # to jax backend device
    setting_grid = {k: jnp.asarray(v) for k, v in setting_grid.items()}

    print(f"starting experiment over {setting_df.shape[0]} settings, outname: {outname}")

    result = experiment_full(setting_grid)
    pehes, lls, q_lims, btx = result
    pehe2_ate, pehe2_offset, pehe2_constrained3, pehe2_constrained4 = pehes
    ll_full, ll_tx, ll_tu = lls
    q_min, q_max = q_lims

    setting_df['ate'] = pehe2_ate
    setting_df['offset'] = pehe2_offset
    setting_df['constrained3'] = pehe2_constrained3
    setting_df['constrained4'] = pehe2_constrained4
    setting_df['ll_full'] = ll_full
    setting_df['ll_tx'] = ll_tx
    setting_df['ll_tu'] = ll_tu
    setting_df['q_min'] = q_min
    setting_df['q_max'] = q_max
    setting_df['btx'] = btx


    setting_df.to_csv(outpath, index_label='settingidx')


def experiment_confounding(setting_grid, enforce_offset=True):
    """
    experiment on effect of confounding in absence of non-collapsibility
    """
    # initialize atx finder
    bisec = Bisection(marginalized_offset_criterion, lower=-10., upper=10., check_bracket=False)

    # initialize_parameters (2 parameters for offset, 3 for constrained offset, 4 for fully obs)
    w_init2 = jnp.zeros(2)
    w_init3 = jnp.zeros(3)
    w_init4 = jnp.zeros(4)

    # offset model
    lbfgs_offset = LBFGS(L_offset_logreg3_txu, tol=1e-6, maxiter=500)

    # constrained offset approach
    lagopt = MinimumLagrangian(L_logistic_3param_txu, H3_expectation, LBFGS, tol=1e-6, maxiter=10)

    # fully observational approach
    lbfgs_full = LBFGS(L_logistic_4param_txu, tol=1e-6, maxiter=500)
    
    # fetcher functions for atx (offset or dummy)
    def fetch_atx_dummy(pu, a0,at,ax,au,atx,atu,axu,atxu):
        return atx
    def fetch_atx_offset(pu, a0,at,ax,au,atx,atu,axu,atxu):
        atx_offset = bisec.run(0.0, pu=pu,a0=a0,at=at,ax=ax,au=au,atu=atu,axu=axu,atxu=atxu).params
        return atx_offset

    def run_setting(setting_prms):
        # unpack setting parameters
        a0 = setting_prms['a0']
        at = setting_prms['at']
        ax = setting_prms['ax']
        au = setting_prms['au']
        atx = setting_prms['atx']
        atu = setting_prms['atu']
        axu = setting_prms['axu']
        atxu = setting_prms['atxu']

        # get ors
        or_uy = jnp.exp(au)
        or_ut = jnp.exp(au * setting_prms['gamma_ut_sign'])

        # marginals
        px = setting_prms['px']
        pu = setting_prms['pu']
        pt = setting_prms['pt']

        # find the atx such that the offset assumption is satifsied (or not)
        atx = lax.cond(enforce_offset, fetch_atx_offset, fetch_atx_dummy,
                       pu,a0,at,ax,au,atx,atu,axu,atxu)

        # calculate marginal odds ratio as from rct
        # calculate the u-conditional probabilities based on gamma_u (odds(y|t,x,u=1) / odds(y|t,x,u=0))
        q3 = parameters_to_probabilities3(a0=a0,at=at,ax=ax,au=au,atx=atx,atu=atu,axu=axu,atxu=atxu)

        # get p(u=1|t=0) and p(u=1|t=1) from p(u=1), odds(u|t=1) / odds(u|t=0) = gamma, p(t=1)
        pu_t0, pu_t1 = solve_qs_from_or(pu, or_ut, pt)

        # interventional betas; NOTE USE MARGINAL PU and not pu_t here as would be used in RCT
        b0, bx, bt, btx = alphas_to_betas(pu, pu, a0=a0,at=at,ax=ax,au=au,atx=atx,atu=atu,axu=axu,atxu=atxu)
        beta_dict = dict(b0=b0, bx=bx, bt=bt, btx=btx, px=px)
        gt = marginalized_or_from_parameters(px=px, b0=b0, bx=bx, bt=bt, btx=btx)

        # calculate the probability of p(t=t,x=x,u=u)
        ps = calculate_ps(pt, px, pu_t0, pu_t1)

        # get the results
        ## offset with correct offset value
        w_offset, _ = lbfgs_offset.run(w_init2, qs=q3, ps=ps, offset=bt)
        ## run constrained estimator
        w_constrained, _ = lagopt.run(w_init3, qs=q3, ps=ps, px=px, gamma=gt)
        ## fully observational
        w_full, _ = lbfgs_full.run(w_init4, qs=q3, ps=ps)

        # get the ATE pehe
        delta0, delta1 = calculate_deltas(b0=b0, bx=bx, bt=bt, btx=btx)
        delta_ate = (1-px)*delta0 + px*delta1
        pehe2_ate = (1-px)*(delta0 - delta_ate)**2 + px * (delta1 - delta_ate) ** 2

        # calculate pehe2s
        ## add back in the offset parameter and interaction parameter for calculating pehe2
        w_offset2 = jnp.append(w_offset, jnp.array((bt, 0.0)))
        w_constrained2 = jnp.append(w_constrained, jnp.array(0.0))
        # (not needed for full as this is a 4-parameter model)
        pehe2_offset = calculate_pehe2(w_offset2, beta_dict)
        pehe2_constrained = calculate_pehe2(w_constrained2, beta_dict)
        pehe2_full = calculate_pehe2(w_full, beta_dict)

        return pehe2_offset, pehe2_constrained, pehe2_full, pehe2_ate

    # run this over all settings
    results = vmap(run_setting)(setting_grid)

    return results


def make_grid_confounding_alphas():
    """
    make parameter grid for confounding experiment using alphas
    """
    a0 = logit(0.05).item()
    at = 1.
    axs = jnp.linspace(jnp.log(1), jnp.log(5), num=5)
    aus = jnp.log(jnp.array([1,2,5.]))
    gamma_ut_signs = jnp.array([-1, 1])
    atx = 0.
    atu = 0.
    axu = 0.
    atxu = 0.

    # marginals
    pu = 0.5
    px = 0.5
    pt = 0.5

    axs_aus_gut = jnp.asarray(list(itertools.product(axs, aus, gamma_ut_signs)))
    grid_df = pd.DataFrame(axs_aus_gut, columns=['ax', 'au', 'gamma_ut_sign'])
    grid_df['a0'] = a0
    grid_df['at'] = at
    grid_df['atx'] = atx
    grid_df['atu'] = atu
    grid_df['axu'] = axu
    grid_df['atxu'] = atxu

    grid_df['pt'] = pt
    grid_df['pu'] = pu
    grid_df['px'] = px

    grid_df = grid_df.astype(jnp.float32)

    grid = grid_df.to_dict(orient='list')
    grid = {k: jnp.asarray(v) for k, v in grid.items()}

    return grid_df, grid


def run_grid_confounding():
    outdir = Path('..') / 'results' / 'confounding'
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=False)

    # get the grid and save
    setting_df, setting_grid = make_grid_confounding_alphas()

    # run experiment
    result = experiment_confounding(setting_grid)
    pehe2_offset, pehe2_constrained, pehe2_full, pehe2_ate = result

    setting_df['offset'] = pehe2_offset
    setting_df['constrained'] = pehe2_constrained
    setting_df['full'] = pehe2_full
    setting_df['ate'] = pehe2_ate

    setting_df.to_csv(outdir / "settingdf.csv", index_label='settingidx')




def experiment_collapsibility(setting_grid):
    # initialize_parameters (2 parameters for offset)
    w_init2 = jnp.zeros(2)
    lbfgs_offset = LBFGS(L_offset_logreg3, tol=1e-6, maxiter=500)

    # constrained offset approach
    w_init3 = jnp.zeros(3)
    lagopt = MinimumLagrangian(L_logistic_3param, H3_expectation, LBFGS, tol=1e-6, maxiter=100)

    def run_setting(setting_prms):
        pt = setting_prms['pt']
        px = setting_prms['px']
        # calculate marginal offset
        gamma = marginalized_or_from_parameters(**setting_prms)
        # calculate probabilities from setting_parameters
        qs = parameters_to_probabilities2(**setting_prms)

        # run offset with wrong offset parameter
        w_offset, _ = lbfgs_offset.run(w_init2, qs=qs, pt=pt, px=px, offset=gamma)

        # run constrained estimator
        w_constrained, _ = lagopt.run(w_init3, qs=qs, pt=pt, px=px, gamma=gamma)

        # get the ATE pehe
        delta0, delta1 = calculate_deltas(**{k: v for k, v in setting_prms.items() if k in ['b0', 'bt', 'bx', 'btx']})
        delta_ate = (1-px)*delta0 + px*delta1
        pehe2_ate = (1-px)*(delta0 - delta_ate)**2 + px * (delta1 - delta_ate) ** 2

        # calculate pehe2s
        ## add back in the offset parameter and interaction parameter for calculating pehe2
        w_offset2 = jnp.append(w_offset, jnp.array((gamma, 0.0)))
        w_constrained2 = jnp.append(w_constrained, jnp.array(0.0))
        pehe2_offset = calculate_pehe2(w_offset2, setting_prms)
        pehe2_constrained = calculate_pehe2(w_constrained2, setting_prms)

        return pehe2_offset, pehe2_constrained, pehe2_ate

    # run this over all settings
    results = vmap(run_setting)(setting_grid)

    return results


def make_setting_grid_collapsibility():
    px = 0.5
    pt = 0.5
    b0 = logit(.5).item()
    bt = 1.
    btx = 0.0
    bxs = jnp.linspace(0., jnp.log(10), num=10)

    grid_df = pd.DataFrame(bxs, columns=['bx'])
    grid_df['b0'] = b0
    grid_df['bt'] = bt
    grid_df['btx'] = btx
    grid_df['px'] = px
    grid_df['pt'] = 0.5

    grid_df = grid_df.astype(jnp.float32)

    grid = grid_df.to_dict(orient='list')
    grid = {k: jnp.asarray(v) for k, v in grid.items()}

    return grid_df, grid


def run_grid_collapsibility():
    outdir = Path('..') / 'results' / 'collapsibility'
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=False)

    # get the grid and save
    setting_df, setting_grid = make_setting_grid_collapsibility()

    # save to file
    result = experiment_collapsibility(setting_grid)
    pehe2_offset, pehe2_constrained, pehe2_ate = result

    setting_df['offset'] = pehe2_offset
    setting_df['constrained'] = pehe2_constrained
    setting_df['ate'] = pehe2_ate

    setting_df.to_csv(outdir / "settingdf.csv", index_label='settingidx')


def experiment_efficiency(k, nreps, n, simgrid):
    nsims = simgrid['b0'].size

    # initialize optimizers
    w_init = jnp.zeros(4)

    # fully observational estimate
    # TODO: lower maxiter here
    lbfgs_unconstrained = LBFGS(fun=binary_logreg, tol=tol, maxiter=500)

    # custom objective for logreg so that it doesn't throw an error for getting gamma argument which is needed for constraint
    def binary_logreg2(params, data, gamma=None):
        return binary_logreg(params, data)

    # constrained offset approach
    lagopt = MinimumLagrangian(binary_logreg2, H4, LBFGS, tol=tol, maxiter=5, unroll=True)

    # vectorize over simgrid
    def run_sim(k, sim_prms):
        # calculate gamma
        gamma = marginalized_or_from_parameters(**sim_prms)

        with seed(rng_seed = k):
            data = make_data_noconfounding(n, **sim_prms)
        X, y = data

        # optimize
        w_unconstrained, state_unconstrained = lbfgs_unconstrained.run(w_init, data=data)
        w_constrained, state_constrained = lagopt.run(w_init, data=data, gamma=gamma)

        # get delta
        deltas_unconstrained = calculate_deltas(*w_unconstrained)
        deltas_constrained = calculate_deltas(*w_constrained)

        return deltas_unconstrained, deltas_constrained

    # scan over reps
    def run_reps(carry, k):
        ks = random.split(k, nsims)
        result = vmap(run_sim)(ks, simgrid)
        return carry, result

    # setup ks
    ks = random.split(k, nreps)
    _, results = lax.scan(run_reps, None, ks)

    return results


# setup experiment grid
def make_simprm_grid_efficiency():
    px = 0.5
    b0s = scipy_logit(np.array((.15, .5)))
    bt = 1.
    btx = 0.0
    bxs = np.log(np.array([1/5, 1/2, 1, 2, 5]))
    gxts = np.log(np.array([1/5, 1/2, 1, 2, 5]))
    b0_x_gxt = np.asarray(list(itertools.product(b0s, bxs, gxts)))
    gt0 = 0.0

    grid_df = pd.DataFrame(b0_x_gxt, columns=['b0', 'bx', 'gxt'])
    grid_df['gt0'] = gt0
    grid_df['bt'] = bt
    grid_df['btx'] = btx
    grid_df['px'] = px

    grid_df = grid_df.astype(np.float32)

    simgrid = grid_df.to_dict(orient='list')
    simgrid = {k: jnp.asarray(v) for k, v in simgrid.items()}

    return grid_df, simgrid


def run_grid_efficiency():
    """
    main experiments
    """
    outdir = Path('..') / "results" / "efficiency"
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=False)

    simprm_df, simgrid = make_simprm_grid_efficiency()
    # make ground truth for each sim prm
    delta0, delta1 = vmap(calculate_deltas)(simgrid['b0'], simgrid['bx'], simgrid['bt'], simgrid['btx'])
    simprm_df['delta0'] = delta0
    simprm_df['delta1'] = delta1

    simprm_df.to_csv(outdir / "simprm_df.csv", index_label='simprmsidx')

    k = PRNGKey(0)
    nsims = simprm_df.shape[0]

    ns = [50, 100, 200, 500]
    ns += [75, 150, 325, 400, 750]
    nreps = 250

    print(f"nsims: {nsims}, num ns: {len(ns)}, nreps: {nreps}")
    for n in ns:
        k, kn = random.split(k)
        outname = f"n{n}.csv"
        print(outname)
        deltas_unconstrained, deltas_constrained = experiment_efficiency(kn, nreps, n, simgrid)
        d0u, d1u = deltas_unconstrained
        d0c, d1c = deltas_constrained
        
        rep_dfs = []
        for d0ui, d1ui, d0ci, d1ci in zip(d0u, d1u, d0c, d1c):
            dfu = pd.DataFrame({'delta0': d0ui, 'delta1': d1ui})
            dfc = pd.DataFrame({'delta0': d0ci, 'delta1': d1ci})

            dfi = pd.concat({'unconstrained': dfu, 'constrained': dfc}).rename_axis(['estimator', 'simprmidx'])
            rep_dfs.append(dfi)

        result_df = pd.concat(rep_dfs).reset_index()
        result_df.to_csv(outdir / outname, index=False)


if __name__ == '__main__':
    args = parser.parse_args()

    def run_experiment():
        if args.experiment == 'efficiency':
            run_grid_efficiency()
        elif args.experiment == 'collapsibility':
            run_grid_collapsibility()
        elif args.experiment == 'confounding':
            run_grid_confounding()
        elif args.experiment == 'full':
            run_grid_full(args.p000, args.enforce_offset)
        else:
            raise ValueError(f"unknown experiment: {args.experiment}")

    if args.gpu:
        gpus = jax.devices()
        with jax.default_device(gpus[args.gpu]):
            run_experiment()
    else:
        run_experiment()

