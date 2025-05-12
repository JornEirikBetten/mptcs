"""
In this file, we define functions used to handle states and data from trajectories. 
"""
import jax 
import jax.numpy as jnp 
import jax.random as jrandom
import numpy as np 

from typing import List, Tuple, NamedTuple

class TestCase(NamedTuple): 
    state: jax.Array
    key: jax.Array

def batch_states(states, batch_size): 
    """
    Batch the states into smaller states.
    """
    num_states = states.observation.shape[0]
    num_batches = num_states // batch_size 
    batched_states = [] 
    for i in range(num_batches): 
        batched_states.append(jax.tree.map(lambda x: x[i*batch_size:(i+1)*batch_size], states))
    if num_states % batch_size != 0: 
        batched_states.append(jax.tree.map(lambda x: x[(num_batches)*batch_size:], states))
    return batched_states

def concatenate_states(states): 
    """
    Concatenate the states into a single state.
    """
    concatenated_states = jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), *states)
    return concatenated_states

from functools import partial 

def check_shape(a, b):
    result_tree = jax.tree.map(lambda x, y: len(x.shape) == len(y.shape), a, b)
    return all(jax.tree_util.tree_leaves(result_tree))


def batch_test_cases(test_cases, batch_size): 
    """
    Batch the test cases into smaller test cases.
    """
    num_test_cases = test_cases.state.observation.shape[0]
    num_batches = num_test_cases // batch_size 
    batched_test_cases = [] 
    for i in range(num_batches): 
        batch_states = jax.tree.map(lambda x: x[i*batch_size:(i+1)*batch_size], test_cases.state) 
        batch_keys = jax.tree.map(lambda x: x[i*batch_size:(i+1)*batch_size], test_cases.key)
        batched_test_cases.append(TestCase(state=batch_states, key=batch_keys))
    if num_test_cases % batch_size != 0: 
        batch_states = jax.tree.map(lambda x: x[(num_batches)*batch_size:], test_cases.state) 
        batch_keys = jax.tree.map(lambda x: x[(num_batches)*batch_size:], test_cases.key)
        batched_test_cases.append(TestCase(state=batch_states, key=batch_keys))
    return batched_test_cases

def extract_states(states, policy_idxs, step_idxs, state_idxs, env_init):
    """
    Extracts the states from the trajectory data.
    """
    extracted_states = jax.tree.map(
        lambda x: x[policy_idxs, step_idxs, state_idxs], states
    )
    # We want to extract len(policy_idxs) states
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, len(policy_idxs))
    state = env_init(rng)
    assert check_shape(state, extracted_states)
    return extracted_states

def extract_states_with_single_policy(states, step_idxs, state_idxs, env_init):
    """
    Extracts the states from the trajectory data.
    """
    extracted_states = jax.tree.map(
        lambda x: x[step_idxs, state_idxs], states
    )
    # We want to extract len(policy_idxs) states
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, 1)
    state = env_init(rng)
    assert check_shape(state, extracted_states)
    return extracted_states

def build_state_extractor(env_init): 
    return jax.jit(partial(extract_states, env_init=env_init))

def build_state_extractor_with_single_policy(env_init): 
    return jax.jit(partial(extract_states_with_single_policy, env_init=env_init))