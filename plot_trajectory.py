import jax 
import jax.numpy as jnp 
import chex 
import haiku as hk 
import optax 


import pgx 

import os 
import sys 

from typing import Dict, Any, Tuple, Callable, Optional, List, Union, Sequence, NamedTuple
#from ga_utils_mdpfuzz import mutate_states, MUTATORS, OBSERVE_FN

import minimal_pats.candidate_generation as candidate_generation 
import minimal_pats.simulators as simulators 
import minimal_pats.forward_fns as forward_fns 
import minimal_pats.param_loader as param_loader 
import minimal_pats.failure_criteria as failure_criteria 
import minimal_pats.test_case_selection as test_case_selection 
import minimal_pats.utils as utils 
import minimal_pats.converters as converters 
from dataclasses import dataclass 
from minimal_pats.data_handler import batch_test_cases
from qdax_modified.core.containers.mapelites_repertoire import MapElitesRepertoire


class TestCase(NamedTuple): 
    state: pgx.State
    key: jax.random.PRNGKey


env_name = sys.argv[1]
centroids_dim = int(sys.argv[2])
method = sys.argv[3]
generation = int(sys.argv[4])
"""
        SETUP 
"""
# Configuration 
num_policies = 5
num_eval_policies = 20 
simulation_steps = 11 
evaluation_steps = 11
failure_threshold = 10
num_iterations = 10 
num_samples = 5000
total_number_of_simulation_steps = 10_000_000
max_corpus_size = 15_000
centroids_shape = (centroids_dim, centroids_dim)
batch_size = 200 

# Environment 
env_fn = pgx.make(env_name) 
if env_name == "minatar-asterix": 
    action_mapping = ["no-op", "left", "up", "right", "down", "dead"]
elif env_name == "minatar-breakout": 
    action_mapping = ['no-op', 'left', 'right', 'dead']
elif env_name == "minatar-space_invaders": 
    action_mapping = ['no-op', 'left', 'right', 'fire', 'dead']
elif env_name == "minatar-seaquest": 
    action_mapping = ['no-op', 'left', 'up', 'right', 'down', 'fire', 'dead']
mutators = candidate_generation.MUTATORS[env_name]
observe_fn = candidate_generation.OBSERVE_FN[env_name] 

# Policy network
network = forward_fns.make_forward_fn(env_fn.num_actions)
network = hk.without_apply_rng(hk.transform(network))

# Get parameters 
path_to_eval_data = os.getcwd() + "/evaluation_data/"
path_to_trained_policies = os.getcwd() + "/trained_policies/"
params_stacked, params_list, test_params_stacked, test_params_list = param_loader.load_params_with_fixed_num_test_policies(num_policies, 
                                                                                                                           num_eval_policies, 
                                                                                                                           env_name, 
                                                                                                                           path_to_eval_data, 
                                                                                                                           path_to_trained_policies, 
                                                                                                                           load_rashomon=True)


key = jax.random.PRNGKey(123123) 
key, _key = jax.random.split(key) 
keys = jax.random.split(_key, 50)
states = jax.vmap(env_fn.init)(keys)
# Build simulator 
simulate = jax.jit(simulators.build_standard_pgx_simulator(env_fn, network, simulation_steps))
fixed_randomness_simulator = jax.jit(simulators.build_pgx_simulator_with_fixed_randomness(env_fn, network, simulation_steps))
# Build mutator 
mutate_fn = jax.jit(candidate_generation.build_mutate_fn(mutators, observe_fn, mutation_prob=0.1, mutation_strength=0.5))

# Build failure criterion 
failure_criterion = jax.jit(failure_criteria.build_step_based_failure_criterion(failure_threshold))

#test_case_simulator = jax.jit(simulators.build_pgx_simulator_of_test_case(env_fn, network, simulation_steps))
test_case_simulator = jax.jit(jax.vmap(simulators.build_pgx_simulator_of_test_case(env_fn, network, simulation_steps), in_axes=(0, None))) 
simulator = jax.jit(jax.vmap(simulators.build_pgx_simulator_of_test_case(env_fn, network, 10_000), in_axes=(0, None))) 
reconstruction_fn = lambda x: x  

def load_repertoire(path: str, reconstruction_fn) -> MapElitesRepertoire:
    """Load a MapElites repertoire from disk.
    
    Args:
        path: Path to the directory containing the saved repertoire
        reconstruction_fn: Function to reconstruct genotypes from flat arrays
        
    Returns:
        The loaded MapElites repertoire
    """
    return MapElitesRepertoire.load(reconstruction_fn, path)


#base_path = os.path.join(os.getcwd(), "results", env_name, method, f"dimension_{centroids_dim}", f"generation_{generation}")
base_path = os.getcwd() + "/results/cost-quality/" + env_name + f"/generation_{generation}/"
multi_policy_path = base_path + "test_suite_20/"
#multi_policy_path = os.path.join(base_path, "multi_policy", "test_suite/")
#single_policy_path = os.path.join(base_path, "single_policy", "test_suite/")
multi_policy_repertoire = load_repertoire(multi_policy_path, reconstruction_fn)
#single_policy_repertoire = load_repertoire(single_policy_path, reconstruction_fn)

path_to_test_cases_plots = os.getcwd() + "/test_case_trajectories/" + method + "/" + env_name + "/"
if not os.path.exists(path_to_test_cases_plots): 
    os.makedirs(path_to_test_cases_plots)
# Extract test cases from repertoire 


key = jax.random.PRNGKey(123123)
key, _key = jax.random.split(key)
keys = jax.random.split(_key, 10)
initial_states = jax.vmap(env_fn.init)(keys) 
test_cases = TestCase(state=initial_states, key=keys) 

trajectories = simulator(params_stacked, test_cases)

termination_indices = jnp.argmax(trajectories.state.terminated, axis=1)


print(termination_indices.shape)
print(termination_indices)

policy_idxs = []; state_idxs = []; step_idxs = []
for policy_idx in range(termination_indices.shape[0]):
    for state_idx in range(termination_indices.shape[1]):
        policy_idxs.append(policy_idx)
        state_idxs.append(state_idx)
        step_idxs.append(termination_indices[policy_idx, state_idx])
policy_idxs = jnp.array(policy_idxs)
state_idxs = jnp.array(state_idxs)
step_idxs = jnp.array(step_idxs)
subset_ = jnp.array([0, 4])

states = jax.tree.map(lambda x: x[policy_idxs[subset_], step_idxs[subset_] - 10, state_idxs[subset_]], trajectories.state)

jax.tree.map(lambda x: print(x.shape), states)

for i in range(states.observation.shape[0]):
    state = jax.tree.map(lambda x: x[jnp.array([i])], states)
    jax.tree.map(lambda x: print(x.shape), state)
    test_case = TestCase(state=state, key=jax.random.PRNGKey(123123).reshape(1, -1))
    trajectories = test_case_simulator(params_stacked, test_case)
    save_path = path_to_test_cases_plots + f"test_case_{i+1}.pdf"
    utils.plot_test_case_trajectory(trajectories, save_path, action_mapping)


# genotypes = multi_policy_repertoire.genotypes[multi_policy_repertoire.fitnesses != -jnp.inf]
# descriptors = multi_policy_repertoire.descriptors[multi_policy_repertoire.fitnesses != -jnp.inf]
# fitnesses = multi_policy_repertoire.fitnesses[multi_policy_repertoire.fitnesses != -jnp.inf]
# rngs = multi_policy_repertoire.rngs[multi_policy_repertoire.fitnesses != -jnp.inf]
# reference_state = jax.vmap(env_fn.init)(jax.random.split(key, 1))
# key, sample_key = jax.random.split(key)
# random_indices = jax.random.randint(sample_key, (50,), 0, genotypes.shape[0])
# for i in range(50): 
#     index = random_indices[i]
#     single_genotype = genotypes[index].reshape(1, genotypes.shape[1])
#     #print(single_genotype.shape)
#     single_rng = rngs[index].reshape(1, rngs.shape[1])
#     single_state = jax.vmap(converters.vec2state_and_reshape, in_axes=(0, None))(single_genotype, reference_state)
#     #single_state = jax.tree.map(lambda x: x.squeeze(axis=0), single_state)
#     test_case = TestCase(state=single_state, key=single_rng)
#     trajectories = test_case_simulator(params_stacked, test_case)
#     print(trajectories.state.observation.shape)
#     save_path = path_to_test_cases_plots + f"test_case_{index+1}.pdf"
#     utils.plot_test_case_trajectory(trajectories, save_path, action_mapping)





