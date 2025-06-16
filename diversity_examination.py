import jax 
import jax.numpy as jnp 
import chex 
import haiku as hk 
import optax 

import pgx as pgx

import os 
import sys 

from typing import Dict, Any, Tuple, Callable, Optional, List, Union, Sequence, NamedTuple

import mptcs.candidate_generation as candidate_generation 
import mptcs.simulators as simulators 
import mptcs.forward_fns as forward_fns 
import mptcs.param_loader as param_loader 
import mptcs.failure_criteria as failure_criteria 
import mptcs.test_case_selection as test_case_selection 
import mptcs.utils as utils 
import mptcs.converters as converters 
from mptcs.evaluation import evaluation 
from dataclasses import dataclass 
from mptcs.data_handler import batch_test_cases
import pandas as pd 
from mptcs.topk_storage import TopKStorage

from qdax_modified.core.containers.mapelites_repertoire import MapElitesRepertoire, compute_euclidean_centroids
from dataclasses import fields

import pacmap
import matplotlib.pyplot as plt

class TestCase(NamedTuple): 
    state: pgx.State
    key: jax.random.PRNGKey

# Inputs 
env_name = sys.argv[1]
centroids_dim = int(sys.argv[2])
num_policies = int(sys.argv[3])
experiment_name = int(sys.argv[4])

# Configuration 
FAILURE_THRESHOLD = 10 
SIMULATION_STEPS = 11 
CENTROIDS_SHAPE = (centroids_dim, centroids_dim)
NUM_EVAL_POLICIES = 20
SELECTION_SIZE = 200
NUM_GENERATIONS = 1000
NUM_INITIAL_STATES = 200
EVALUATION_INTERVAL = 500
TOP_K_SIZE = 2500  # Size of the top-k selection

# Environment 
env_fn = pgx.make(env_name) 

# # Initialize the appropriate environment-specific state coverage object
# if "asterix" in env_name:
#     environment_state_space = AsterixStateCoverage()
# elif "breakout" in env_name:
#     environment_state_space = BreakoutStateCoverage()
# elif "seaquest" in env_name:
#     environment_state_space = SeaquestStateCoverage()
# elif "space_invaders" in env_name:
#     environment_state_space = SpaceInvadersStateCoverage()
# else:
#     raise ValueError(f"Unsupported environment: {env_name}")

# Policy network 
network = forward_fns.make_forward_fn(env_fn.num_actions)
network = hk.without_apply_rng(hk.transform(network))

# Converters from state to vector and back 
state2vec = jax.vmap(converters.state2vec)
reference_state = jax.vmap(env_fn.init)(jax.random.split(jax.random.PRNGKey(123123), 1))
reference_vector = state2vec(reference_state).reshape(-1)
vec2state = lambda x: converters.vec2state_and_reshape(x, reference_state)
vec2state = jax.vmap(vec2state)

# Mutation 
mutators = candidate_generation.MUTATORS[env_name]
observe_fn = candidate_generation.OBSERVE_FN[env_name] 
mutate_fn = jax.jit(candidate_generation.build_mutate_fn(mutators, observe_fn, mutation_prob=0.1, mutation_strength=0.5)) 

def build_mutate_test_case_fn(mutate_fn): 
    def mutate_test_case_fn(test_cases, mutate_key): 
        mutated_states, _ = mutate_fn(test_cases.state, mutate_key)
        mutated_test_cases = TestCase(
            state = mutated_states, 
            key = test_cases.key
        )
        return mutated_test_cases
    return mutate_test_case_fn

mutate_test_case_fn = build_mutate_test_case_fn(mutate_fn)

# Build failure criterion 
failure_criterion = jax.jit(jax.vmap(failure_criteria.build_step_based_failure_criterion(FAILURE_THRESHOLD), in_axes=(0))) 

# Build test case simulator (vectorized over the first dimension of the concatenated network parameters, i.e., over multiple policies)
test_case_simulator = jax.jit(jax.vmap(simulators.build_pgx_simulator_of_test_case(env_fn, network, SIMULATION_STEPS), in_axes=(0, None)))

evaluate_test_suite = evaluation.build_evaluation_of_test_suite(test_case_simulator, vec2state, test_case_selection.build_failure_rate_measure(failure_criterion), 10)
tcp_scorer = test_case_selection.build_tcp_scorer_on_trajectories(failure_criterion) 

"""
        ARCHIVE SETUPS 
"""

n_actions = env_fn.num_actions
mean_entropy_limits = (0, jnp.log(n_actions))
entropy_limits = (0, jnp.log(n_actions))
xlabel = "observation variance"
ylabel = "policy confidence"
if env_name == "minatar-asterix": 
    mean_entropy_limits = (jnp.log(n_actions)/4, jnp.log(n_actions) / 4 * 3)
    obsvar_limits = (0.013, 0.028)
    xtick_labels = utils.make_ticks_from_limits(obsvar_limits, 5, 3)
    ytick_labels = utils.make_ticks_from_limits(mean_entropy_limits, 5, 2)
    ticks = [0, int(CENTROIDS_SHAPE[0] / 4), int(CENTROIDS_SHAPE[0]/2), int(CENTROIDS_SHAPE[0]/ 4 * 3), CENTROIDS_SHAPE[0]-1]
elif env_name == "minatar-breakout": 
    mean_entropy_limits = (0, jnp.log(n_actions)*3/4)
    obsvar_limits = (0.04, 0.08)
    xtick_labels = utils.make_ticks_from_limits(obsvar_limits, 5, 3)
    ytick_labels = utils.make_ticks_from_limits(mean_entropy_limits, 5, 2)
    ticks = [0, int(CENTROIDS_SHAPE[0] / 4), int(CENTROIDS_SHAPE[0]/2), int(CENTROIDS_SHAPE[0]/ 4 * 3), CENTROIDS_SHAPE[0]-1]
elif env_name == "minatar-seaquest": 
    mean_entropy_limits = (0, jnp.log(n_actions)*3/4)
    obsvar_limits = (0.02, 0.025)
    xtick_labels = utils.make_ticks_from_limits(obsvar_limits, 5, 3)
    ytick_labels = utils.make_ticks_from_limits(mean_entropy_limits, 5, 2)
    ticks = [0, int(CENTROIDS_SHAPE[0] / 4), int(CENTROIDS_SHAPE[0]/2), int(CENTROIDS_SHAPE[0]/ 4 * 3), CENTROIDS_SHAPE[0]-1]
elif env_name == "minatar-space_invaders": 
    mean_entropy_limits = (0, jnp.log(n_actions)/2)
    obsvar_limits = (0.1, 0.15)
    xtick_labels = utils.make_ticks_from_limits(obsvar_limits, 5, 3)
    ytick_labels = utils.make_ticks_from_limits(mean_entropy_limits, 5, 2)
    ticks = [0, int(CENTROIDS_SHAPE[0] / 4), int(CENTROIDS_SHAPE[0]/2), int(CENTROIDS_SHAPE[0]/ 4 * 3), CENTROIDS_SHAPE[0]-1]

test_suite_centroids = compute_euclidean_centroids(
    grid_shape=CENTROIDS_SHAPE, 
    minval=[obsvar_limits[0], mean_entropy_limits[0]], 
    maxval=[obsvar_limits[1], mean_entropy_limits[1]]
)

# Get trained policies
path_to_eval_data = os.getcwd() + "/evaluation_data/"
path_to_trained_policies = os.getcwd() + "/trained_policies/"
params_stacked, params_list, test_params_stacked, test_params_list = param_loader.load_params_with_fixed_num_test_policies(num_policies, 
                                                                                                                           NUM_EVAL_POLICIES, 
                                                                                                                           env_name, 
                                                                                                                           path_to_eval_data, 
                                                                                                                           path_to_trained_policies, 
                                                                                                                           load_rashomon=True)


# Initialize test suites
mptcs_test_suite = MapElitesRepertoire.init_default(
    genotype=reference_vector, 
    centroids=test_suite_centroids, 
    rng=jnp.zeros_like(jax.random.PRNGKey(123123))
)

topk_storage = TopKStorage(TOP_K_SIZE, reference_vector)



def build_next_generation_fn(sample_from_archive, mutate_test_case_fn):
    def next_generation(test_suite, alternative_test_cases, key):
        key, sample_key, mutate_key = jax.random.split(key, 3)
        # Sample states from archive
        num_valid_test_cases_in_archive = jnp.sum(test_suite.fitnesses != -jnp.inf)
        if num_valid_test_cases_in_archive > 100:
            selected_candidates = sample_from_archive(test_suite, sample_key)
        else:
            selected_candidates = alternative_test_cases
        # Mutate states
        candidates = mutate_test_case_fn(selected_candidates, mutate_key)
        return candidates
    return next_generation

def add_to_archive(archive, genotypes, descriptors, performance, rngs):
    fitness_before = archive.fitnesses.copy()
    archive = archive.add(
        batch_of_genotypes=genotypes,
        batch_of_descriptors=descriptors,
        batch_of_fitnesses=performance,
        batch_of_rngs=rngs
    )
    found_failures = jnp.sum(performance > 0)
    changed_cells = archive.fitnesses > fitness_before
    added_to_archive = (changed_cells & (fitness_before == -jnp.inf)).sum()
    replaced_elite = (changed_cells & (fitness_before != -jnp.inf)).sum()
    return archive, found_failures, added_to_archive, replaced_elite


def build_sample_from_test_suite(selection_size, vec2state): 
    def sample_from_test_suite(test_suite, key): 
        valid_indices = jnp.where(test_suite.fitnesses != -jnp.inf) 
        probabilities = jnp.where(test_suite.fitnesses[valid_indices] > 0, test_suite.fitnesses[valid_indices], 0) + 1e-6
        probabilities = probabilities / jnp.sum(probabilities)
        sampled_indices = jax.random.choice(key, valid_indices[0], (selection_size,), p=probabilities) 
        sampled_genotypes = test_suite.genotypes[sampled_indices]
        sampled_rngs = jax.random.split(key, selection_size)
        sampled_states = vec2state(sampled_genotypes)
        sampled_test_cases = TestCase(
            state = sampled_states, 
            key = sampled_rngs
        )
        return sampled_test_cases
    return sample_from_test_suite

def build_evaluator_fn(test_case_simulator, failure_criterion): 
    def evaluate_test_cases(params, test_cases): 
        trajs = test_case_simulator(params, test_cases)
        failures = failure_criterion(trajs)
        failure_rates = jnp.sum(failures, axis=0) / failures.shape[0]
        return trajs, failures, failure_rates 
    return evaluate_test_cases

def build_execution_fn(evaluator_fn, number_of_policies, number_of_batches): 
    def execute_test_cases_and_collect_diversity_metrics(params, test_suite): 
        fail_counter = jnp.zeros(number_of_policies)
        
        # Extract test cases from test suite
        genotypes = test_suite.genotypes 
        descriptors = test_suite.descriptors 
        rngs = test_suite.rngs 
        indices = jnp.where(test_suite.fitnesses > 0)
        indices_subsets = jnp.array_split(indices[0], number_of_batches)
        all_unique_observations = set() 
        all_failure_rates = []
        # Execute test cases and collect diversity metrics
        for i in range(number_of_batches): 
            batch_indices = indices_subsets[i]
            batch_genotypes = genotypes[batch_indices]
            batch_descriptors = descriptors[batch_indices]
            batch_rngs = rngs[batch_indices]

            batch_test_cases = TestCase(
                state = vec2state(batch_genotypes), 
                    key = batch_rngs
                )

            # Simulate test cases
            trajectories, failures, failure_rates= evaluator_fn(params, batch_test_cases)
            # Reshape observations to (num_timesteps * num_test_cases, 10, 10, 4)
            observations = trajectories.state.observation.reshape((-1,) + trajectories.state.observation.shape[-3:])
            # Count unique observations by converting to bytes for hashing
            observations_bytes = set(map(lambda x: x.tobytes(), observations)) 
            all_unique_observations.update(observations_bytes)
            all_failure_rates.extend(failure_rates)
            fail_counter += jnp.sum(failures, axis=-1) 
        number_of_unique_observations = len(all_unique_observations) 
        mean_failure_rate = jnp.mean(jnp.array(all_failure_rates))
        return number_of_unique_observations, mean_failure_rate, fail_counter
    return execute_test_cases_and_collect_diversity_metrics


sample_from_archive = build_sample_from_test_suite(SELECTION_SIZE, vec2state)
next_generation = build_next_generation_fn(sample_from_archive, mutate_test_case_fn)
evaluator_fn = build_evaluator_fn(test_case_simulator, failure_criterion)
evaluate_and_collect = build_execution_fn(evaluator_fn, env_fn.num_actions, 5)
# Initialize a population of random initial states 
key = jax.random.PRNGKey(123123)
key, _init_key = jax.random.split(key)
init_keys = jax.random.split(_init_key, 10)
states = jax.vmap(env_fn.init)(init_keys)
test_cases = TestCase(state=states, key=init_keys)
print("Before simulation")
trajs = test_case_simulator(params_stacked, test_cases)
print("After simulation")
# Get descriptors and solvability
tcp_scores, descriptors, solvability = tcp_scorer(trajs)
print("After scoring")

genotypes = state2vec(test_cases.state) 
# Add to archive
mptcs_test_suite, found_failures, added_to_archive, replaced_elite = add_to_archive(mptcs_test_suite, 
                                                                                    genotypes, 
                                                                                    descriptors, 
                                                                                    tcp_scores, 
                                                                                    test_cases.key)

topk_storage, found_failures, added_to_topk, replaced_elite = add_to_archive(topk_storage, 
                                                                             genotypes, 
                                                                             descriptors, 
                                                                             tcp_scores, 
                                                                             test_cases.key)

mptcs_test_cases = test_cases 
topk_test_cases = test_cases 

for generation in range(1, NUM_GENERATIONS + 1): 
    print(f"Generation {generation}")
    key, _key = jax.random.split(key)
    # top-k candidate selection 
    topk_candidates = next_generation(topk_storage, topk_test_cases, _key)
    trajs_topk = test_case_simulator(params_stacked, topk_candidates)
    tcp_scores_topk, descriptors_topk, solvability_topk = tcp_scorer(trajs_topk)
    genotypes_topk = state2vec(topk_candidates.state)
    topk_storage, found_failures, added_to_topk, replaced_elite = add_to_archive(topk_storage, 
                                                                                 genotypes_topk, 
                                                                                 descriptors_topk, 
                                                                                 tcp_scores_topk, 
                                                                                 topk_candidates.key)
    topk_test_cases = topk_candidates 
    # MPTCS candidate selection 
    mptcs_candidates = next_generation(mptcs_test_suite, mptcs_test_cases, _key)
    trajs_mptcs = test_case_simulator(params_stacked, mptcs_candidates)
    tcp_scores_mptcs, descriptors_mptcs, solvability_mptcs = tcp_scorer(trajs_mptcs)
    genotypes_mptcs = state2vec(mptcs_candidates.state)
    mptcs_test_suite, found_failures, added_to_archive, replaced_elite = add_to_archive(mptcs_test_suite, 
                                                                                        genotypes_mptcs, 
                                                                                        descriptors_mptcs, 
                                                                                        tcp_scores_mptcs, 
                                                                                        mptcs_candidates.key)
    mptcs_test_cases = mptcs_candidates 
    
    
    
    
    if generation % EVALUATION_INTERVAL == 0: 
        # evaluate mptcs test suite 
        num_unique_observations, mean_failure_rate, fail_counter = evaluate_and_collect(test_params_stacked, mptcs_test_suite)
        # evaluate topk test suite 
        num_unique_observations, mean_failure_rate, fail_counter = evaluate_and_collect(test_params_stacked, topk_storage)
    

