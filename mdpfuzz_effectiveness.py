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

import mptcs.candidate_generation as candidate_generation 
import mptcs.simulators as simulators 
import mptcs.forward_fns as forward_fns 
import mptcs.param_loader as param_loader 
import mptcs.failure_criteria as failure_criteria 
import mptcs.test_case_selection as test_case_selection 
import mptcs.utils as utils 
import mptcs.converters as converters 
from dataclasses import dataclass 
from mptcs.data_handler import batch_test_cases

class TestCase(NamedTuple): 
    state: pgx.State
    key: jax.random.PRNGKey


env_name = sys.argv[1]
centroids_dim = int(sys.argv[2])
experiment_name = int(sys.argv[3])
"""
        SETUP 
"""
# Configuration 
num_policies = 20 
num_eval_policies = 20 
simulation_steps = 11 
evaluation_steps = 11
failure_threshold = 10
num_iterations = 10 
num_samples = 5000
total_number_of_simulation_steps = 50_000_000
target_number_of_failures = 20_000
max_corpus_size = 25_000
centroids_shape = (centroids_dim, centroids_dim)
batch_size = 200 

# Environment 
env_fn = pgx.make(env_name) 
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


key = jax.random.PRNGKey(123123 * (experiment_name + 1)) 
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

"""
        CANDIDATE SELECTION 
"""
test_case_simulator = jax.jit(simulators.build_pgx_simulator_of_test_case(env_fn, network, simulation_steps))
mdpfuzz_candidate_generator = candidate_generation.build_mdpfuzz_candidate_generator(env_fn, 
                                                                                     mutate_fn, 
                                                                                     test_case_simulator, 
                                                                                     failure_criterion, 
                                                                                     num_samples=num_samples, 
                                                                                     num_iterations=num_iterations, 
                                                                                     total_number_of_simulation_steps=total_number_of_simulation_steps, 
                                                                                     max_corpus_size=max_corpus_size, 
                                                                                     target_number_of_failures=target_number_of_failures)

all_failed_test_cases, run_info = mdpfuzz_candidate_generator(params_list[0], jax.random.PRNGKey(123123 * (experiment_name + 1)), 0)

print("Summary of mdpfuzz candidate generation: ")
#all_test_cases = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *all_failed_test_cases) 
print(f"Number of failures/test cases: {run_info['found_tcs']}")
print(f"Number of simulation steps: {run_info['step_counts']}")



"""
        TEST CASE SELECTION 
"""
#from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire, compute_euclidean_centroids 
# Use modified MapElites repertoire  
from qdax_modified.core.containers.mapelites_repertoire import MapElitesRepertoire, compute_euclidean_centroids
from dataclasses import fields
# Setup for repertoire grid 
state2vec = jax.vmap(converters.state2vec)
n_actions = env_fn.num_actions
mean_entropy_limits = (0, jnp.log(n_actions))
entropy_limits = (0, jnp.log(n_actions))
if env_name == "minatar-asterix": 
    mean_entropy_limits = (jnp.log(n_actions)/4, jnp.log(n_actions) / 4 * 3)
    obsvar_limits = (0.020, 0.030)
    xtick_labels = utils.make_ticks_from_limits(obsvar_limits, 5, 3)
    ytick_labels = utils.make_ticks_from_limits(mean_entropy_limits, 5, 2)
    ticks = [0, int(centroids_shape[0] / 4), int(centroids_shape[0]/2), int(centroids_shape[0]/ 4 * 3), centroids_shape[0]-1]
elif env_name == "minatar-breakout": 
    mean_entropy_limits = (0, jnp.log(n_actions)*3/4)
    obsvar_limits = (0.04, 0.08)
    xtick_labels = utils.make_ticks_from_limits(obsvar_limits, 5, 3)
    ytick_labels = utils.make_ticks_from_limits(mean_entropy_limits, 5, 2)
    ticks = [0, int(centroids_shape[0] / 4), int(centroids_shape[0]/2), int(centroids_shape[0]/ 4 * 3), centroids_shape[0]-1]
elif env_name == "minatar-seaquest": 
    mean_entropy_limits = (0, jnp.log(n_actions)*3/4)
    obsvar_limits = (0.02, 0.025)
    xtick_labels = utils.make_ticks_from_limits(obsvar_limits, 5, 3)
    ytick_labels = utils.make_ticks_from_limits(mean_entropy_limits, 5, 2)
    ticks = [0, int(centroids_shape[0] / 4), int(centroids_shape[0]/2), int(centroids_shape[0]/ 4 * 3), centroids_shape[0]-1]
elif env_name == "minatar-space_invaders": 
    mean_entropy_limits = (0, jnp.log(n_actions)/2)
    obsvar_limits = (0.1, 0.15)
    xtick_labels = utils.make_ticks_from_limits(obsvar_limits, 5, 3)
    ytick_labels = utils.make_ticks_from_limits(mean_entropy_limits, 5, 2)
    ticks = [0, int(centroids_shape[0] / 4), int(centroids_shape[0]/2), int(centroids_shape[0]/ 4 * 3), centroids_shape[0]-1]

test_suite_centroids = compute_euclidean_centroids(
    grid_shape=centroids_shape, 
    minval=[obsvar_limits[0], mean_entropy_limits[0]], 
    maxval=[obsvar_limits[1], mean_entropy_limits[1]]
)

reference_state = jax.vmap(env_fn.init)(jax.random.split(key, 1))
reference_vector = converters.state2vec(reference_state)
test_suite = MapElitesRepertoire.init_default(
    genotype=reference_vector, 
    centroids=test_suite_centroids, 
    rng=jnp.zeros_like(jax.random.PRNGKey(123123))
)

single_policy_test_suite = MapElitesRepertoire.init_default(
    genotype=reference_vector, 
    centroids=test_suite_centroids, 
    rng=jnp.zeros_like(jax.random.PRNGKey(123123))
)

base_genotypes = test_suite.genotypes
test_case_simulator = jax.jit(jax.vmap(simulators.build_pgx_simulator_of_test_case(env_fn, network, simulation_steps), in_axes=(0, None))) 

# trajs = test_case_simulator(test_params_stacked, all_failed_test_cases)
#print("Simulated trajectories observation shape: ", trajs.state.observation.shape)
#print(all_failed_test_cases.state.observation.shape)
#print(trajs.state.observation.shape)
vmapped_failure_criterion = jax.jit(jax.vmap(failure_criterion, in_axes=(0)))
evaluation_simulator = jax.jit(jax.vmap(simulators.build_pgx_simulator_of_test_case(env_fn, network, evaluation_steps), in_axes=(0, None)))
test_case_evaluator = test_case_selection.build_test_case_evaluator(evaluation_simulator, vmapped_failure_criterion) 
tcp_scorer = test_case_selection.build_pats_candidate_evaluator(evaluation_simulator, vmapped_failure_criterion)
confirmed_solvable = 0 
batched_test_cases = batch_test_cases(all_failed_test_cases, batch_size)
number_of_generated_failures = 0 
for batch_of_test_cases in batched_test_cases: 
    tcp_scores, descriptors, solvability = tcp_scorer(params_stacked, batch_of_test_cases) 
    #failure_rates = test_case_evaluator(test_params_stacked, batch_of_test_cases)

    #print(f"Confirmed solvable test cases ratio in batch: {jnp.sum(solvability)/solvability.shape[0]}")
    confirmed_solvable += jnp.sum(solvability)
    tcp_solvability = tcp_scores != 0 
    # Assert that the TCP scores reflect the solvability of the test cases 
    assert jnp.sum(tcp_solvability != solvability) == 0 

    # Add to multi-policy test suite 
    tcp_scores = jnp.where(tcp_scores != 0, tcp_scores, -jnp.inf)
    genotypes = jax.vmap(converters.state2vec, in_axes=(0))(batch_of_test_cases.state)
    rngs = batch_of_test_cases.key 
    base_genotypes = jnp.repeat(reference_vector[jnp.newaxis, :], genotypes.shape[0], axis=0)
    performance_indices = jnp.where(tcp_scores != -jnp.inf, 1, 0).reshape(-1, 1)*jnp.ones(genotypes.shape)
    genotypes = jnp.where(performance_indices, genotypes, base_genotypes)
    test_suite = test_suite.add(
        genotypes, 
        descriptors, 
        tcp_scores, 
        rngs
    )
    # Add to single policy test suite 
    solvability = jnp.where(solvability, 1, -jnp.inf)
    single_policy_test_suite = single_policy_test_suite.add(
        genotypes, 
        descriptors, 
        solvability, 
        rngs
    )
    number_of_generated_failures += batch_of_test_cases.state.observation.shape[0]
    

confirmed_solvable_ratio = confirmed_solvable / number_of_generated_failures
print(f"Confirmed solvable test cases ratio: {confirmed_solvable_ratio}")
qd_test_case_priority_score = jnp.sum(test_suite.fitnesses[test_suite.fitnesses > 0])
path = os.getcwd() + "/results/mdpfuzz_candidate_generation/" + env_name + "/experiment_" + str(experiment_name) + "/"
# Plot multi-policy test suite 
#multi_policy_results_path = os.getcwd() + "/results/" + env_name + "/mdpfuzz_candidate_generation/dimension_" + str(centroids_dim) + "/multi_policy/"
multi_policy_results_path = path + "multi_policy/"
if not os.path.exists(multi_policy_results_path): 
    os.makedirs(multi_policy_results_path)
utils.plot_archive(test_suite, 0, multi_policy_results_path + "test_suite.pdf", "test case priority score", "observation variance", "mean entropy", xtick_labels, ytick_labels, ticks, shape=centroids_shape)

# Plot single policy test suite 
#single_policy_results_path = os.getcwd() + "/results/" + env_name + "/mdpfuzz_candidate_generation/dimension_" + str(centroids_dim) + "/single_policy/"
single_policy_results_path = path + "single_policy/"
if not os.path.exists(single_policy_results_path): 
    os.makedirs(single_policy_results_path)
utils.plot_archive(single_policy_test_suite, 0, single_policy_results_path + "test_suite.pdf", "solvable", "observation variance", "mean entropy", xtick_labels, ytick_labels, ticks, shape=centroids_shape)

"""
SAVE ARCHIVES 
"""
saving_mp_test_suite_path = multi_policy_results_path + f"test_suite/"
if not os.path.exists(saving_mp_test_suite_path): 
    os.makedirs(saving_mp_test_suite_path)
test_suite.save(saving_mp_test_suite_path)

saving_sp_test_suite_path = single_policy_results_path + f"test_suite/"
if not os.path.exists(saving_sp_test_suite_path): 
    os.makedirs(saving_sp_test_suite_path)
single_policy_test_suite.save(saving_sp_test_suite_path)


"""
        EVALUATE MULTI-POLICY TEST SUITE 
"""
# Extract test cases from test suite 
accepted_descriptors = test_suite.descriptors[test_suite.fitnesses != -jnp.inf] 
accepted_genotypes = test_suite.genotypes[test_suite.fitnesses != -jnp.inf]
accepted_keys = test_suite.rngs[test_suite.fitnesses != -jnp.inf]
accepted_states = jax.vmap(converters.vec2state_and_reshape, in_axes=(0, None))(test_suite.genotypes[test_suite.fitnesses != -jnp.inf], reference_state)
# Extracted test cases in test suite 
test_cases_in_test_suite = TestCase(state=accepted_states, key=accepted_keys)

# Empty validation archive 
validation_archive = MapElitesRepertoire.init_default(
    genotype=reference_vector, 
    centroids=test_suite_centroids, 
    rng=jnp.zeros_like(jax.random.PRNGKey(123123))
)

failure_rates = test_case_evaluator(test_params_stacked, test_cases_in_test_suite)

validation_archive = validation_archive.add(
    accepted_genotypes, 
    accepted_descriptors, 
    failure_rates, 
    accepted_keys
)
qd_failure_rate_mp_test_suite = jnp.sum(validation_archive.fitnesses[validation_archive.fitnesses > 0])
mean_failure_rate_mp_test_suite = jnp.mean(validation_archive.fitnesses[validation_archive.fitnesses > 0])

utils.plot_archive(validation_archive, 0, multi_policy_results_path + "validation.pdf", "failure rate", "observation variance", "mean entropy", xtick_labels, ytick_labels, ticks, shape=centroids_shape)
multi_policy_validation_archive_path = multi_policy_results_path + f"validation_archive/"
if not os.path.exists(multi_policy_validation_archive_path): 
    os.makedirs(multi_policy_validation_archive_path)
validation_archive.save(multi_policy_validation_archive_path)

"""
        EVALUATE SINGLE-POLICY TEST SUITE 
"""
# Extract test cases from single policy test suite 
accepted_descriptors = single_policy_test_suite.descriptors[single_policy_test_suite.fitnesses != -jnp.inf] 
accepted_genotypes = single_policy_test_suite.genotypes[single_policy_test_suite.fitnesses != -jnp.inf]
accepted_keys = single_policy_test_suite.rngs[single_policy_test_suite.fitnesses != -jnp.inf]
accepted_states = jax.vmap(converters.vec2state_and_reshape, in_axes=(0, None))(single_policy_test_suite.genotypes[single_policy_test_suite.fitnesses != -jnp.inf], reference_state)

# Extracted test cases in single policy test suite 
test_cases_in_single_policy_test_suite = TestCase(state=accepted_states, key=accepted_keys)
failure_rates = test_case_evaluator(test_params_stacked, test_cases_in_single_policy_test_suite)

# Empty validation archive 
validation_archive = MapElitesRepertoire.init_default(
    genotype=reference_vector, 
    centroids=test_suite_centroids, 
    rng=jnp.zeros_like(jax.random.PRNGKey(123123))
)

failure_rates = test_case_evaluator(test_params_stacked, test_cases_in_single_policy_test_suite)

validation_archive = validation_archive.add(
    accepted_genotypes, 
    accepted_descriptors, 
    failure_rates, 
    accepted_keys
)
qd_failure_rate_sp_test_suite = jnp.sum(validation_archive.fitnesses[validation_archive.fitnesses > 0])
mean_failure_rate_sp_test_suite = jnp.mean(validation_archive.fitnesses[validation_archive.fitnesses > 0])
utils.plot_archive(validation_archive, 0, single_policy_results_path + "validation.pdf", "failure rate", "observation variance", "mean entropy", xtick_labels, ytick_labels, ticks, shape=centroids_shape)
single_policy_validation_archive_path = single_policy_results_path + f"validation_archive/"
if not os.path.exists(single_policy_validation_archive_path): 
    os.makedirs(single_policy_validation_archive_path)
validation_archive.save(single_policy_validation_archive_path)
print(f"QD failure rate of multi-policy test suite: {qd_failure_rate_mp_test_suite}")
print(f"Mean failure rate of multi-policy test suite: {mean_failure_rate_mp_test_suite}")
print(f"QD failure rate of single-policy test suite: {qd_failure_rate_sp_test_suite}")
print(f"Mean failure rate of single-policy test suite: {mean_failure_rate_sp_test_suite}")

import pandas as pd 
number_of_test_cases_in_mp_test_suite = jnp.sum(test_suite.fitnesses != -jnp.inf)
number_of_test_cases_in_sp_test_suite = jnp.sum(single_policy_test_suite.fitnesses != -jnp.inf)
data_saving_path = path + "data.csv"
data = {
    "generated_failures": [number_of_generated_failures],
    "confirmed_solvable_ratio": [confirmed_solvable_ratio],
    "qd_test_case_priority_score": [qd_test_case_priority_score],
    "qd_failure_rate_mp_test_suite": [qd_failure_rate_mp_test_suite],
    "qd_failure_rate_sp_test_suite": [qd_failure_rate_sp_test_suite], 
    "total_number_of_simulation_steps": [jnp.sum(jnp.array(run_info["step_counts"]))],
    "test_cases_in_mp_test_suite": [number_of_test_cases_in_mp_test_suite],
    "test_cases_in_sp_test_suite": [number_of_test_cases_in_sp_test_suite], 
    "mean_failure_rate_mp_test_suite": [mean_failure_rate_mp_test_suite],
    "mean_failure_rate_sp_test_suite": [mean_failure_rate_sp_test_suite]
}
df = pd.DataFrame(data=data)
df.to_csv(data_saving_path)


