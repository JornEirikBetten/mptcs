import jax 
import jax.numpy as jnp 
import chex 
#import haiku as hk 
from haiku import without_apply_rng, transform
#import optax 

import pgx as pgx

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
from mptcs.evaluation import evaluation 
from dataclasses import dataclass 
from mptcs.data_handler import batch_test_cases
import pandas as pd 

from qdax_modified.core.containers.mapelites_repertoire import MapElitesRepertoire, compute_euclidean_centroids
from dataclasses import fields

class TestCase(NamedTuple): 
    state: pgx.State
    key: jax.random.PRNGKey

# Inputs 
env_name = sys.argv[1]
centroids_dim = int(sys.argv[2])
num_policies = int(sys.argv[3])
experiment_number = int(sys.argv[4])

# Configuration 
FAILURE_THRESHOLD = 10 
SIMULATION_STEPS = 11 
CENTROIDS_SHAPE = (centroids_dim, centroids_dim)
NUM_EVAL_POLICIES = 20
SELECTION_SIZE = 200
NUM_GENERATIONS = 25_000
NUM_INITIAL_STATES = 200
EVALUATION_INTERVAL = 500
# Environment 
env_fn = pgx.make(env_name) 

# Policy network 
network = forward_fns.make_forward_fn(env_fn.num_actions)
network = without_apply_rng(transform(network))

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
        ARCHIVE SETUP 
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

def build_next_generation_fn(sample_from_archive, mutate_test_case_fn): 
    def next_generation(test_suite, alternative_test_cases, key): 
        key, sample_key, mutate_key = jax.random.split(key, 3)
        # Sample states from archive 
        num_valid_test_cases_in_archive = jnp.sum(test_suite.fitnesses != -jnp.inf)
        #print(num_valid_test_cases_in_archive)
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


        

# Get trained policies
path_to_eval_data = os.getcwd() + "/evaluation_data/"
path_to_trained_policies = os.getcwd() + "/trained_policies/"
params_stacked, params_list, test_params_stacked, test_params_list = param_loader.load_params_with_fixed_num_test_policies(num_policies, 
                                                                                                                           NUM_EVAL_POLICIES, 
                                                                                                                           env_name, 
                                                                                                                           path_to_eval_data, 
                                                                                                                           path_to_trained_policies, 
                                                                                                                           load_rashomon=True)




sample_from_test_suite = build_sample_from_test_suite(SELECTION_SIZE, vec2state)
next_generation = build_next_generation_fn(sample_from_test_suite, mutate_test_case_fn)

# Creating empty test suites for test suites using 1 through num_policies policies 
test_suites = []
for policy_number in range(1, 2, 1): 
    test_suite = MapElitesRepertoire.init_default(
        genotype=reference_vector, 
        centroids=test_suite_centroids, 
        rng=jnp.zeros_like(jax.random.PRNGKey(123123))
    )
    test_suites.append(test_suite)


key = jax.random.PRNGKey(123123 * experiment_number)
key, _init_key = jax.random.split(key) 
init_keys = jax.random.split(_init_key, NUM_INITIAL_STATES)
states = jax.vmap(env_fn.init)(init_keys)
candidates = TestCase(
    state = states, 
    key = init_keys
)

trajs = test_case_simulator(params_stacked, candidates)
tcp_scores, descriptors, solvability = tcp_scorer(trajs)

def extract_partial_trajectories(trajectories, num_policies):
    partial_trajectories = jax.tree.map(
        lambda x: x[:num_policies], 
        trajectories
    ) 
    return partial_trajectories

# Setting up data to be extracted from runs 
run_data = {
    "generation": []
}
    
extraction_data = ["found_failures", "mean_tcp_score", "mean_failure_rate", "qd_tcp_score", "qd_failure_rate", "added_to_archive", "replaced_elite", "num_valid_test_cases", "num_sim_steps", "not_confirmed_solvable"]
for number_of_policies in range(1, 2, 1):  
    for data in extraction_data: 
        run_data[str(number_of_policies) + "_" + data] = []

# for key, value in run_data.items(): 
#     print(key)
# print("Original shape: ", trajs.state.observation.shape)
# for policy_number in range(2, num_policies+1): 
#     print("Policy number: ", policy_number)
#     partial_trajectories = extract_partial_trajectories(trajs, policy_number)
#     print("Partial shape: ", partial_trajectories.state.observation.shape)
#     tcp_scores, descriptors, solvability = tcp_scorer(partial_trajectories)

path_to_results = os.getcwd() + "/results/cost-quality/" + env_name + f"/single_policy_{experiment_number}/"
if not os.path.exists(path_to_results): 
    os.makedirs(path_to_results)


def collect_data(data_dictionary, test_suite, trajectories, number_of_policies, generation, not_confirmed_solvable, added_to_archive, replaced_elite, found_failures):
    # If it is an evaluation generation 
    if generation % EVALUATION_INTERVAL == 0 and jnp.sum(test_suite.fitnesses > 0) > 0: 
        # Empty validation archive 
        empty_validation_archive = MapElitesRepertoire.init_default(
            genotype=reference_vector, 
            centroids=test_suite_centroids, 
            rng=jnp.zeros_like(jax.random.PRNGKey(123123))
        )
        saving_path = path_to_results + f"generation_{generation}/"
        if not os.path.exists(saving_path): 
            os.makedirs(saving_path)
        saving_path_test_suite = saving_path + f"test_suite_{number_of_policies}/"
        if not os.path.exists(saving_path_test_suite): 
            os.makedirs(saving_path_test_suite)
        test_suite.save(saving_path_test_suite)
        # Evaluate test suite 
        validation_archive = evaluate_test_suite(test_suite, test_params_stacked, empty_validation_archive)
        summed_failure_rate = jnp.sum(validation_archive.fitnesses[validation_archive.fitnesses > 0])
        mean_failure_rate = summed_failure_rate / jnp.sum(test_suite.fitnesses > 0)
        data_dictionary[str(number_of_policies) + "_mean_failure_rate"].append(mean_failure_rate)
        data_dictionary[str(number_of_policies) + "_qd_failure_rate"].append(summed_failure_rate)
        utils.plot_archive(test_suite, 
                           generation, 
                           saving_path + f"test_suite_{number_of_policies}.pdf", 
                           "test case priority", xlabel, ylabel, xtick_labels, ytick_labels, ticks, shape=CENTROIDS_SHAPE)
        utils.plot_archive(validation_archive, 
                           generation, 
                           saving_path + f"validation_{number_of_policies}.pdf", 
                           "failure rate", xlabel, ylabel, xtick_labels, ytick_labels, ticks, shape=CENTROIDS_SHAPE)
        
    elif generation < EVALUATION_INTERVAL and generation % 20 == 0 and jnp.sum(test_suite.fitnesses > 0) > 0: 
        # Empty validation archive 
        empty_validation_archive = MapElitesRepertoire.init_default(
            genotype=reference_vector, 
            centroids=test_suite_centroids, 
            rng=jnp.zeros_like(jax.random.PRNGKey(123123))
        )
        saving_path = path_to_results + f"generation_{generation}/"
        if not os.path.exists(saving_path): 
            os.makedirs(saving_path)
        saving_path_test_suite = saving_path + f"test_suite_{number_of_policies}/"
        if not os.path.exists(saving_path_test_suite): 
            os.makedirs(saving_path_test_suite)
        test_suite.save(saving_path_test_suite)
        # Evaluate test suite 
        validation_archive = evaluate_test_suite(test_suite, test_params_stacked, empty_validation_archive)
        summed_failure_rate = jnp.sum(validation_archive.fitnesses[validation_archive.fitnesses > 0])
        mean_failure_rate = summed_failure_rate / jnp.sum(test_suite.fitnesses > 0)
        data_dictionary[str(number_of_policies) + "_mean_failure_rate"].append(mean_failure_rate)
        data_dictionary[str(number_of_policies) + "_qd_failure_rate"].append(summed_failure_rate)
        utils.plot_archive(test_suite, 
                           generation, 
                           saving_path + f"test_suite_{number_of_policies}.pdf", 
                           "test case priority", xlabel, ylabel, xtick_labels, ytick_labels, ticks, shape=CENTROIDS_SHAPE)
        utils.plot_archive(validation_archive, 
                           generation, 
                           saving_path + f"validation_{number_of_policies}.pdf", 
                           "failure rate", xlabel, ylabel, xtick_labels, ytick_labels, ticks, shape=CENTROIDS_SHAPE)
    else: 
        data_dictionary[str(number_of_policies) + "_mean_failure_rate"].append(-1)
        data_dictionary[str(number_of_policies) + "_qd_failure_rate"].append(-1)
        
    # Test suite metrics collection 
    num_steps = jnp.sum(~trajectories.state.terminated) 
    data_dictionary[str(number_of_policies) + "_num_sim_steps"].append(num_steps)
    data_dictionary[str(number_of_policies) + "_found_failures"].append(found_failures)
    data_dictionary[str(number_of_policies) + "_not_confirmed_solvable"].append(not_confirmed_solvable)
    data_dictionary[str(number_of_policies) + "_added_to_archive"].append(added_to_archive)
    data_dictionary[str(number_of_policies) + "_replaced_elite"].append(replaced_elite)
    num_valid_test_cases = jnp.sum(test_suite.fitnesses > 0)
    data_dictionary[str(number_of_policies) + "_num_valid_test_cases"].append(num_valid_test_cases)
    mean_tcp_score = jnp.mean(test_suite.fitnesses[test_suite.fitnesses > 0])
    data_dictionary[str(number_of_policies) + "_mean_tcp_score"].append(mean_tcp_score)
    qd_tcp_score = jnp.sum(test_suite.fitnesses[test_suite.fitnesses > 0])
    data_dictionary[str(number_of_policies) + "_qd_tcp_score"].append(qd_tcp_score)
    return data_dictionary
    
candidate_storage = [candidates for i in range(num_policies)]


for generation in range(1, NUM_GENERATIONS+1):
    print(f"Generation {generation} of {NUM_GENERATIONS}")
    run_data["generation"].append(generation)
    key, generation_key = jax.random.split(key)
    # Single-policy test suite collection 
    previous_candidates = candidate_storage[0] 
    candidates = next_generation(test_suites[0], previous_candidates, generation_key)
    candidates_genotypes = state2vec(candidates.state)
    candidate_storage[0] = candidates
    trajs = test_case_simulator(params_stacked, candidates)
    # Getting descriptors and solvability 
    _, descriptors, solvability = tcp_scorer(trajs)
    single_policy_trajectories = extract_partial_trajectories(trajs, 1) 
    failures = jnp.sum(single_policy_trajectories.state.terminated, axis=1) > 0 
    single_policy_score = (failures * solvability).reshape(-1) 
    not_confirmed_solvable = jnp.sum(~solvability)
    test_suite, found_failures, added_to_archive, replaced_elite = add_to_archive(test_suites[0], candidates_genotypes, descriptors, single_policy_score, candidates.key)
    test_suites[0] = test_suite
    run_data = collect_data(run_data, test_suite, single_policy_trajectories, 1, generation, not_confirmed_solvable, added_to_archive, replaced_elite, found_failures)
    
    # # Multi-policy test suite collection 
    # for policy_number in range(2, num_policies+1): 
    #     test_suite = test_suites[policy_number-1]
    #     previous_candidates = candidate_storage[policy_number-1]
    #     candidates = next_generation(test_suite, previous_candidates, generation_key)
    #     candidates_genotypes = state2vec(candidates.state)
    #     candidate_storage[policy_number-1] = candidates
    #     trajs = test_case_simulator(params_stacked, candidates)
    #     # Getting descriptors and solvability 
    #     _, descriptors, solvability = tcp_scorer(trajs)
    #     multi_policy_trajectories = extract_partial_trajectories(trajs, policy_number)
    #     tcp_scores, _, _ = tcp_scorer(multi_policy_trajectories)
    #     test_suite, found_failures, added_to_archive, replaced_elite = add_to_archive(test_suite, candidates_genotypes, descriptors, tcp_scores, candidates.key)
    #     test_suites[policy_number-1] = test_suite
    #     not_confirmed_solvable = jnp.sum(~solvability)
    #     run_data = collect_data(run_data, test_suite, multi_policy_trajectories, policy_number, generation, not_confirmed_solvable, added_to_archive, replaced_elite, found_failures)

save_path = path_to_results  
df = pd.DataFrame(run_data)
df.to_csv(save_path + f"run_data.csv", index=False)