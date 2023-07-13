"""Implementation of minimization based version of AugmentedLagrangian in JAX.:

    Objectives of the form:

    L(x) = f(x) + nu H(x)^2
    nu > 0 

    or 
    
    L(x) = f(x) + nu (G(x) - c)^2
    nu > 0 
    """

import inspect

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import objective
from jaxopt._src import tree_util


class MinLagState(NamedTuple):
    """Named tuple containing state information."""

    iter_num: int
    value: float
    nu: float
    obj_value: float
    H_value: float
    # Hhist: jnp.ndarray
    # value_hist: jnp.ndarray
    # param_hist: jnp.ndarray
    H2max: float
    error: float
    ilack: int
    # inner_iter_num: jnp.ndarray



@dataclass(eq=False)
class MinimumLagrangian(base.IterativeSolver):
    """Minimize Lagrangian-like objective

    This solver minimizes::

      objective(params, *args, **kwargs) =
        fun(params, *args, **kwargs)
        subject to the constraints in H

    Objectives of the form:

    L(x) = f(x) + nu H(x)^2
    nu > 0 

    or 
    
    L(x) = f(x) + nu (G(x) - c)^2
    nu > 0 

    Attributes:
      fun: a smooth function of the form ``fun(params, *args, **kwargs)``.
        It should be a ``objective.CompositeLinearFunction`` object.
        H: a function of the form ``fun(params, *args, **kwargs)`` that calculates the constraints
      l2_penalty: minimum penalization of constraints
      K_scaler: increase penalty at each timepoint
      sigma0: penalty size at first iteration
      sigma_scaling: increase of penatly of each outer iteration
      K_scaler: scaler of K parameter
      inner_solver: Solver (non-instantiated)
      inner_kwargs: kwargs for inner_solver (inner_solver(fun=..., **kwargs))
      maxiter: maximum number of outer iterations.
      ilack_max: maximum number of steps without updates to parameters within tol
      tol: tolerance to use.
      verbose: whether to print error on every iteration or not.
        Warning: verbose=True will automatically disable jit.
      implicit_diff: whether to enable implicit diff or autodiff of unrolled
        iterations.
      implicit_diff_solve: the linear system solver to use.
      jit: whether to JIT-compile the optimization loop (default: "auto").
      unroll: whether to unroll the optimization loop (default: "auto").
    """

    fun: Callable
    H: Callable
    inner_solver: base.Solver
    inner_kwargs: dict = field(default_factory=dict)
    maxiter: int = 500
    nu0: float = 0.01
    nu_scaling: float = 10.0
    ilack_max: int = 2
    tol: float = 1e-4
    verbose: int = 0
    implicit_diff: bool = True
    implicit_diff_solve: Optional[Callable] = None
    jit: base.AutoOrBoolean = "auto"
    unroll: base.AutoOrBoolean = "auto"

    def init_state(self, init_params: Any, *args, **kwargs) -> MinLagState:
        """Initialize the solver state.

        Args:
          init_params: pytree containing the initial parameters.
          *args: additional positional arguments to be passed to ``fun`` and ``H``.
          **kwargs: additional keyword arguments to be passed to ``fun`` and ``H``.
        Returns:
          state
        """
        # check solution state
        d0 = self.H(init_params, *args, **kwargs)
        d2 = d0 * d0

        # initialize lagrange multipliers
        ## if nu0 is a scalar (default=0.0) it gets broadcasted to shape of d0
        ## if nu0 is a vector (d0-shaped), it remains the same
        nu = self.nu0 * jnp.ones(d0.shape)

        # might define these for logging parameter values across iterations
        # Hhist = jnp.zeros((self.maxiter,nu.size))
        # value_hist = jnp.zeros(self.maxiter)
        # param_hist = jnp.zeros((self.maxiter, init_params.size))
        # inner_iter_num = jnp.zeros(self.maxiter)


        state = MinLagState(
            iter_num=jnp.asarray(0),
            value=jnp.asarray(jnp.inf),
            nu=nu,
            obj_value=jnp.asarray(jnp.inf),
            H_value=d0,
            H2max=jnp.max(d2),
            # Hhist =Hhist,
            # value_hist=value_hist,
            # param_hist=param_hist,
            error=jnp.inf,
            ilack=int(0),
            # inner_iter_num=inner_iter_num
        )

        return state

    def update(
        self, params: Any, state: NamedTuple, *args, **kwargs
    ) -> base.OptStep:
        """Performs one epoch of AugmentedLagrangian

        Args:
          params: pytree containing the parameters.
          state: named tuple containing the solver state.
          *args: additional positional arguments to be passed to ``fun``.
          **kwargs: additional keyword arguments to be passed to ``fun``.
        Returns:
          (params, state)
        """
        # run inner unconstrained optimization
        sol = self.solver.run(
            params, nu=state.nu, *args, **kwargs
        )
        new_params, inner_state = sol

        # check constraints on new solution
        d0 = self.H(new_params, *args, **kwargs)
        d2 = d0 * d0

        # update nu for each constraint
        nu_prev = state.nu
        constraints_crit = d2 < self.tol
        nu = jnp.where(constraints_crit, nu_prev, nu_prev * self.nu_scaling)

        # get value of original objective
        obj_value = self.fun(params, *args, **kwargs)

        # check if any update on objective value and constraint value, if not update, ilack
        # dparams = jnp.abs(params - new_params)
        dobj = jnp.abs(obj_value - state.obj_value)
        dH = jnp.max(jnp.abs(d0 - state.H_value))
        ilack = jnp.where(dobj + dH < 2 * self.tol, state.ilack + 1, 0)


        # stop, converged = self_check_converged(inner_state.value, state.obj_value, K, ilack)
        # converged if no movement in function value and constraints satisfied, or no update in nu
        converged = jnp.logical_or(ilack > self.ilack_max, jnp.all(constraints_crit))
        error = jnp.where(converged, 0., inner_state.value)

        # update some history checking
        # Hhist = state.Hhist.at[state.iter_num].set(d0)
        # value_hist = state.value_hist.at[state.iter_num].set(inner_state.value)
        # param_hist = state.param_hist.at[state.iter_num].set(new_params)
        # inner_iter_num = state.inner_iter_num.at[state.iter_num].set(inner_state.iter_num)

        state = MinLagState(
            iter_num=state.iter_num + 1,
            value=inner_state.value,
            nu=nu,
            obj_value=obj_value,
            H_value=d0,
            H2max=jnp.max(d2),
            # Hhist = Hhist,
            # value_hist=value_hist,
            # param_hist=param_hist,
            error=error,
            ilack=ilack,
            # inner_iter_num=inner_iter_num
        )

        # note: should this be MinLagState?
        return base.OptStep(new_params, state)

    # NOTE: not used
    def _check_converged(self, obj_value, prev_obj_value, K, ilack):
        if (
            jnp.isfinite(obj_value)
            and jnp.isfinite(prev_obj_value)
            and jnp.abs(obj_value - prev_obj_value) < self.tol
            and K < self.tol
        ):
            converged = True
            stop = True
        elif ilack >= ilack_max:
            converged = False
            stop = True
        else:
            converged = False
            stop = False
        return stop, converged


    def __post_init__(self):
        # Pre-compile useful functions.
        # create unconstrained optimization problem
        def optimality_fun(x, nu, *args, **kwargs):
            # compute constraints value
            d0 = self.H(x, *args, **kwargs)
            d2 = d0 * d0
            # update objective value based on constraints
            val = (
                self.fun(x, *args, **kwargs)
                + jnp.sum(nu * d2)
            )
            return val

        self.optimality_fun = optimality_fun

        # instantiate solver
        self.solver = self.inner_solver(self.optimality_fun, **self.inner_kwargs)

        # Sets up reference signature.
        self.reference_signature = self.fun
