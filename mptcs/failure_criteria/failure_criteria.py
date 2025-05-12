import jax.numpy as jnp  
from typing import Callable
from mptcs.simulators import Trajectory


def build_step_based_failure_criterion(step_threshold: int) -> Callable: 
    def failure_criterion(trajs: Trajectory) -> jnp.ndarray: 
        lengths = jnp.argmax(trajs.state.terminated, axis=0)
        lengths = jnp.where(lengths == 0, trajs.state.terminated.shape[0], lengths)
        failures = jnp.where(lengths <= step_threshold, 1, 0)
        return failures
    return failure_criterion