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

class TestCase(NamedTuple): 
    state: pgx.State
    key: jax.random.PRNGKey

# Configuration
FAILURE_THRESHOLD = 10
SIMULATION_STEPS = 11
NUM_EVAL_POLICIES = 20
NUM_BATCHES = 25
CENTROIDS_SHAPE = (50, 50)




def build_evaluator_fn(test_case_simulator, failure_criterion): 
    def evaluate_test_cases(params, test_cases): 
        trajs = test_case_simulator(params, test_cases)
        failures = failure_criterion(trajs)
        failure_rates = jnp.sum(failures, axis=0) / failures.shape[0]
        return trajs, failures, failure_rates 
    return evaluate_test_cases

def build_execution_fn(evaluator_fn, number_of_policies, number_of_batches, reference_vector, test_suite_centroids): 
    def execute_test_cases_and_collect_diversity_metrics(params, test_suite): 
        fail_counter = jnp.zeros(number_of_policies)
        
        # Extract test cases from test suite
        genotypes = test_suite.genotypes 
        descriptors = test_suite.descriptors 
        rngs = test_suite.rngs 
        validation_archive = MapElitesRepertoire.init_default(
            genotype=reference_vector, 
            centroids=test_suite_centroids, 
            rng=jnp.zeros_like(jax.random.PRNGKey(123123))
        )
        indices = jnp.where(test_suite.fitnesses > 0)
        indices_subsets = jnp.array_split(indices[0], number_of_batches)
        all_unique_observations = set() 
        all_failure_rates = []
        
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
            trajectories, failures, failure_rates = evaluator_fn(params, batch_test_cases)
            
            validation_archive = validation_archive.add(
                batch_of_genotypes=batch_genotypes, 
                batch_of_descriptors=batch_descriptors, 
                batch_of_fitnesses=failure_rates, 
                batch_of_rngs=batch_rngs
            )
            # Reshape observations to collect unique ones
            observations = trajectories.state.observation.reshape((-1,) + trajectories.state.observation.shape[-3:])
            observations_bytes = set(map(lambda x: x.tobytes(), observations)) 
            all_unique_observations.update(observations_bytes)
            all_failure_rates.extend(failure_rates)
            fail_counter += jnp.sum(failures, axis=-1) 
            
        number_of_unique_observations = len(all_unique_observations) 
        mean_failure_rate = jnp.mean(jnp.array(all_failure_rates))
        return number_of_unique_observations, mean_failure_rate, fail_counter, validation_archive
    return execute_test_cases_and_collect_diversity_metrics

env_names = ["minatar-asterix", "minatar-breakout", "minatar-seaquest", "minatar-space_invaders"]


path_to_diversity_suites = os.getcwd() + "/results/diversity/"

results = {
    "env_name": [], 
    "mptcs_unique_observations": [], 
    "top_k_unique_observations": [], 
    "mptcs_mean_failure_rate": [], 
    "top_k_mean_failure_rate": [],
    "mptcs_num_test_cases": [],
    "top_k_num_test_cases": []
}

for policy_num in range(1, 21, 1): 
    results[f"mptcs_failures_{policy_num}_policies"] = []
    results[f"top_k_failures_{policy_num}_policies"] = []

for env_name in env_names: 
    path = path_to_diversity_suites + env_name + "/"
    # Get trained policies
    path_to_eval_data = os.getcwd() + "/evaluation_data/"
    path_to_trained_policies = os.getcwd() + "/trained_policies/"
    params_stacked, params_list, test_params_stacked, test_params_list = param_loader.load_params_with_fixed_num_test_policies(20, 
                                                                                                                            NUM_EVAL_POLICIES, 
                                                                                                                            env_name, 
                                                                                                                            path_to_eval_data, 
                                                                                                                            path_to_trained_policies, 
                                                                                                                            load_rashomon=True)
    
    """
            ARCHIVE SETUPS 
    """
    env_fn = pgx.make(env_name)
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

    
    # Converters from state to vector and back 
    state2vec = jax.vmap(converters.state2vec)
    reference_state = jax.vmap(env_fn.init)(jax.random.split(jax.random.PRNGKey(123123), 1))
    reference_vector = state2vec(reference_state).reshape(-1)
    vec2state = lambda x: converters.vec2state_and_reshape(x, reference_state)
    vec2state = jax.vmap(vec2state)
    
    # Policy network 
    network = forward_fns.make_forward_fn(env_fn.num_actions)
    network = hk.without_apply_rng(hk.transform(network))
    
    # Build failure criterion 
    failure_criterion = jax.jit(jax.vmap(failure_criteria.build_step_based_failure_criterion(FAILURE_THRESHOLD), in_axes=(0))) 

    # Build test case simulator (vectorized over the first dimension of the concatenated network parameters, i.e., over multiple policies)
    test_case_simulator = jax.jit(jax.vmap(simulators.build_pgx_simulator_of_test_case(env_fn, network, SIMULATION_STEPS), in_axes=(0, None)))
    evaluator_fn = build_evaluator_fn(test_case_simulator, failure_criterion)
    execute_test_cases_and_collect_diversity_metrics = build_execution_fn(evaluator_fn, NUM_EVAL_POLICIES, NUM_BATCHES, reference_vector, test_suite_centroids)
    
    for experiment in range(1, 11, 1): 
        experiment_path = path + "experiment_" + str(experiment) + "/generation_1000/"
        
        mptcs_path = experiment_path + "mptcs/"
        top_k_path = experiment_path + "topk/"
        mptcs_suite = MapElitesRepertoire.load(lambda x: x, mptcs_path)
        topk_storage = TopKStorage(2500, reference_vector)
        top_k_suite = topk_storage.load(top_k_path)
        
        mptcs_num_unique_observations, mptcs_mean_failure_rate, mptcs_fail_counter, mptcs_validation_archive = execute_test_cases_and_collect_diversity_metrics(params_stacked, mptcs_suite)
        top_k_num_unique_observations, top_k_mean_failure_rate, top_k_fail_counter, top_k_validation_archive = execute_test_cases_and_collect_diversity_metrics(params_stacked, top_k_suite)
        
        results["env_name"].append(env_name)
        results["mptcs_unique_observations"].append(mptcs_num_unique_observations)
        results["top_k_unique_observations"].append(top_k_num_unique_observations)
        results["mptcs_mean_failure_rate"].append(mptcs_mean_failure_rate)
        results["top_k_mean_failure_rate"].append(top_k_mean_failure_rate)
        results["mptcs_num_test_cases"].append(jnp.sum(jnp.where(mptcs_suite.fitnesses > 0, 1, 0)))
        results["top_k_num_test_cases"].append(jnp.sum(jnp.where(top_k_suite.fitnesses > 0, 1, 0)))
        
        for policy_num in range(1, 21, 1): 
            results[f"mptcs_failures_{policy_num}_policies"].append(mptcs_fail_counter[policy_num])
            results[f"top_k_failures_{policy_num}_policies"].append(top_k_fail_counter[policy_num])

        
        print(f"Finished experiment {experiment} for {env_name}")
        print(f"unique observations mptcs: {results['mptcs_unique_observations'][-1]}")
        print(f"unique observations topk: {results['top_k_unique_observations'][-1]}")
        print(f"mean failure rate mptcs: {results['mptcs_mean_failure_rate'][-1]}")
        print(f"mean failure rate topk: {results['top_k_mean_failure_rate'][-1]}")
        print(f"test cases mptcs: {results['mptcs_num_test_cases'][-1]}")
        print(f"test cases topk: {results['top_k_num_test_cases'][-1]}")

df = pd.DataFrame(results)
df.to_csv(path_to_diversity_suites + "results.csv", index=False)
        