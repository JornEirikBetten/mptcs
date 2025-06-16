import jax 
import jax.numpy as jnp 
import chex 
import haiku as hk 
import optax 

import pgx as pgx

import os 
import sys 

from typing import Dict, Any, Tuple, Callable, Optional, List, Union, Sequence, NamedTuple

import minimal_pats.candidate_generation as candidate_generation 
import minimal_pats.simulators as simulators 
import minimal_pats.forward_fns as forward_fns 
import minimal_pats.param_loader as param_loader 
import minimal_pats.failure_criteria as failure_criteria 
import minimal_pats.test_case_selection as test_case_selection 
import minimal_pats.utils as utils 
import minimal_pats.converters as converters 
from minimal_pats.evaluation import evaluation 
from dataclasses import dataclass 
from minimal_pats.data_handler import batch_test_cases
import pandas as pd 


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

# Initialize the appropriate environment-specific state coverage object
if "asterix" in env_name:
    environment_state_space = AsterixStateCoverage()
elif "breakout" in env_name:
    environment_state_space = BreakoutStateCoverage()
elif "seaquest" in env_name:
    environment_state_space = SeaquestStateCoverage()
elif "space_invaders" in env_name:
    environment_state_space = SpaceInvadersStateCoverage()
else:
    raise ValueError(f"Unsupported environment: {env_name}")

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

# Add state coverage calculation function
# def calculate_state_coverage(trajectories, environment_state_space):
#     """
#     Calculate the state space coverage from trajectories.
    
#     Args:
#         trajectories: Trajectories from test case simulation
#         environment_state_space: Environment-specific state coverage object
        
#     Returns:
#         Dictionary of coverage metrics
#     """
#     # Use the get_state_coverage function we implemented
#     coverage_metrics = get_state_coverage(trajectories.state, environment_state_space)
#     return coverage_metrics

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

# Initialize top-k storage
topk_storage = {
    'genotypes': jnp.zeros((TOP_K_SIZE, reference_vector.shape[0])),
    'fitnesses': jnp.full(TOP_K_SIZE, -jnp.inf),
    'descriptors': jnp.zeros((TOP_K_SIZE, 2)),
    'rngs': jnp.zeros((TOP_K_SIZE, 2), dtype=jnp.uint32)
}

def update_topk_storage(storage, new_genotypes, new_fitnesses, new_descriptors, new_rngs):
    # Combine existing and new data
    all_genotypes = jnp.concatenate([storage['genotypes'], new_genotypes])
    all_fitnesses = jnp.concatenate([storage['fitnesses'], new_fitnesses])
    all_descriptors = jnp.concatenate([storage['descriptors'], new_descriptors])
    all_rngs = jnp.concatenate([storage['rngs'], new_rngs])
    
    # Sort by fitness
    sorted_indices = jnp.argsort(all_fitnesses)[::-1]
    
    # Take top k
    top_indices = sorted_indices[:TOP_K_SIZE]
    
    return {
        'genotypes': all_genotypes[top_indices],
        'fitnesses': all_fitnesses[top_indices],
        'descriptors': all_descriptors[top_indices],
        'rngs': all_rngs[top_indices]
    }

def build_sample_from_topk(selection_size, vec2state):
    def sample_from_topk(storage, key):
        valid_indices = jnp.where(storage['fitnesses'] != -jnp.inf)[0]
        probabilities = jnp.where(storage['fitnesses'][valid_indices] > 0, 
                                storage['fitnesses'][valid_indices], 0) + 1e-6
        probabilities = probabilities / jnp.sum(probabilities)
        sampled_indices = jax.random.choice(key, valid_indices, (selection_size,), p=probabilities)
        sampled_genotypes = storage['genotypes'][sampled_indices]
        sampled_rngs = jax.random.split(key, selection_size)
        sampled_states = vec2state(sampled_genotypes)
        sampled_test_cases = TestCase(
            state=sampled_states,
            key=sampled_rngs
        )
        return sampled_test_cases
    return sample_from_topk

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
        if num_valid_test_cases_in_archive > 100:
            selected_candidates = sample_from_archive(test_suite, sample_key)
        else:
            selected_candidates = alternative_test_cases
        # Mutate states
        candidates = mutate_test_case_fn(selected_candidates, mutate_key)
        return candidates
    return next_generation

def build_next_generation_fn_dictionary(sample_from_archive, mutate_test_case_fn):
    def next_generation(test_dictionary, alternative_test_cases, key):
        key, sample_key, mutate_key = jax.random.split(key, 3)
        # Sample states from archive
        num_valid_test_cases_in_archive = jnp.sum(test_dictionary['fitnesses'] != -jnp.inf)
        if num_valid_test_cases_in_archive > 100:
            selected_candidates = sample_from_archive(test_dictionary, sample_key)
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

# Initialize sampling and generation functions
sample_from_mapelites = build_sample_from_test_suite(SELECTION_SIZE, vec2state)
sample_from_topk = build_sample_from_topk(SELECTION_SIZE, vec2state)
next_generation_mapelites = build_next_generation_fn(sample_from_mapelites, mutate_test_case_fn)
next_generation_topk = build_next_generation_fn_dictionary(sample_from_topk, mutate_test_case_fn)

# Initialize initial states
key = jax.random.PRNGKey(123123)
key, _init_key = jax.random.split(key)
init_keys = jax.random.split(_init_key, NUM_INITIAL_STATES)
states = jax.vmap(env_fn.init)(init_keys)
candidates = TestCase(
    state=states,
    key=init_keys
)

# Setting up data to be extracted from runs
run_data = {
    "generation": []
}

extraction_data = ["found_failures", "mean_tcp_score", "mean_failure_rate", "qd_tcp_score", "qd_failure_rate", "added_to_archive", "replaced_elite", "num_valid_test_cases", "num_sim_steps", "not_confirmed_solvable"]
names = ["mapelites", "topk"]

# Add state coverage metrics to the list of data to extract
extraction_data.append("state_coverage_overall")
extraction_data.append("unique_states_coverage")  # Add unique state coverage metric

run_data = {
    "generation": [],
    "mptcs_num_unique_observations": [],
    "mptcs_mean_failure_rate": [], 
    "topk_num_unique_observations": [],
    "topk_mean_failure_rate": []
}
path_to_results = os.getcwd() + "/results/diversity/" + env_name + "/experiment_"+ str(experiment_name) + "/"
if not os.path.exists(path_to_results):
    os.makedirs(path_to_results)

#def collect_data(data_dictionary, test_suite, trajectories, name, generation, not_confirmed_solvable, added_to_archive, replaced_elite, found_failures):
    # # Calculate state coverage
    # coverage_metrics = calculate_state_coverage(trajectories, environment_state_space)
    # data_dictionary[name + "_state_coverage_overall"].append(coverage_metrics.get("overall", 0.0))
    
    # # Calculate unique state coverage
    # unique_coverage_metrics = get_unique_state_coverage(trajectories.state, environment_state_space)
    # data_dictionary[name + "_unique_states_coverage"].append(unique_coverage_metrics.get("unique_states_coverage", 0.0))
    
    
    # Save detailed state coverage to a separate file if it's an evaluation generation
    # if generation % EVALUATION_INTERVAL == 0 and jnp.sum(test_suite.fitnesses > 0) > 0:
    #     saving_path = path_to_results + f"generation_{generation}/"
    #     if not os.path.exists(saving_path):
    #         os.makedirs(saving_path)
        
    #     # Save detailed state coverage metrics
    #     coverage_df = pd.DataFrame({k: [v] for k, v in coverage_metrics.items()})
    #     coverage_df.to_csv(saving_path + f"state_coverage_{name}.csv", index=False)
        
    #     # Save detailed unique state coverage metrics
    #     unique_coverage_df = pd.DataFrame({k: [v] for k, v in unique_coverage_metrics.items()})
    #     unique_coverage_df.to_csv(saving_path + f"unique_state_coverage_{name}.csv", index=False)
    
    # If it is an evaluation generation
    # if generation % EVALUATION_INTERVAL == 0 and jnp.sum(test_suite.fitnesses > 0) > 0:
    #     # Empty validation archive
    #     empty_validation_archive = MapElitesRepertoire.init_default(
    #         genotype=reference_vector,
    #         centroids=test_suite_centroids,
    #         rng=jnp.zeros_like(jax.random.PRNGKey(123123))
    #     )
    #     saving_path = path_to_results + f"generation_{generation}/"
    #     if not os.path.exists(saving_path):
    #         os.makedirs(saving_path)
    #     saving_path_test_suite = saving_path + f"test_suite_{name}/"
    #     if not os.path.exists(saving_path_test_suite):
    #         os.makedirs(saving_path_test_suite)
    #     test_suite.save(saving_path_test_suite)
    #     # Evaluate test suite
    #     validation_archive = evaluate_test_suite(test_suite, test_params_stacked, empty_validation_archive)
    #     summed_failure_rate = jnp.sum(validation_archive.fitnesses[validation_archive.fitnesses > 0])
    #     mean_failure_rate = summed_failure_rate / jnp.sum(test_suite.fitnesses > 0)
    #     data_dictionary[name + "_mean_failure_rate"].append(mean_failure_rate)
    #     data_dictionary[name + "_qd_failure_rate"].append(summed_failure_rate)
    #     utils.plot_archive(test_suite,
    #                       generation,
    #                       saving_path + f"test_suite_{name}.pdf",
    #                       "test case priority", xlabel, ylabel, xtick_labels, ytick_labels, ticks, shape=CENTROIDS_SHAPE)
    #     utils.plot_archive(validation_archive,
    #                       generation,
    #                       saving_path + f"validation_{name}.pdf",
    #                       "failure rate", xlabel, ylabel, xtick_labels, ytick_labels, ticks, shape=CENTROIDS_SHAPE)
    # else:
    #     data_dictionary[name + "_mean_failure_rate"].append(-1)
    #     data_dictionary[name + "_qd_failure_rate"].append(-1)
        
    # # Test suite metrics collection
    # num_steps = jnp.sum(~trajectories.state.terminated)
    # data_dictionary[name + "_num_sim_steps"].append(num_steps)
    # data_dictionary[name + "_found_failures"].append(found_failures)
    # data_dictionary[name + "_not_confirmed_solvable"].append(not_confirmed_solvable)
    # data_dictionary[name + "_added_to_archive"].append(added_to_archive)
    # data_dictionary[name + "_replaced_elite"].append(replaced_elite)
    # num_valid_test_cases = jnp.sum(test_suite.fitnesses > 0)
    # data_dictionary[name + "_num_valid_test_cases"].append(num_valid_test_cases)
    # mean_tcp_score = jnp.mean(test_suite.fitnesses[test_suite.fitnesses > 0])
    # data_dictionary[name + "_mean_tcp_score"].append(mean_tcp_score)
    # qd_tcp_score = jnp.sum(test_suite.fitnesses[test_suite.fitnesses > 0])
    # data_dictionary[name + "_qd_tcp_score"].append(qd_tcp_score)
    # return data_dictionary
    
def execute_test_cases_and_collect_diversity_metrics(test_suite, type, number_of_batches, failure_rate_measure): 
    # Extract test cases from test suite
    if type == "mptcs":
        genotypes = test_suite.genotypes 
        descriptors = test_suite.descriptors 
        rngs = test_suite.rngs 
        indices = jnp.where(test_suite.fitnesses > 0)
        indices_subsets = jnp.array_split(indices[0], number_of_batches)
        all_unique_observations = set() 
        all_failure_rates = []
    elif type == "topk":
        genotypes = topk_storage['genotypes']
        descriptors = topk_storage['descriptors']
        rngs = topk_storage['rngs']
        indices = jnp.where(topk_storage['fitnesses'] > 0)
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
        trajectories = test_case_simulator(test_params_stacked, batch_test_cases)
        # Reshape observations to (num_timesteps * num_test_cases, 10, 10, 4)
        observations = trajectories.state.observation.reshape((-1,) + trajectories.state.observation.shape[-3:])
        # Count unique observations by converting to bytes for hashing
        observations_bytes = set(map(lambda x: x.tobytes(), observations)) 
        all_unique_observations.update(observations_bytes)
        failure_rates = failure_rate_measure(trajectories)
        all_failure_rates.extend(failure_rates)
        
    number_of_unique_observations = len(all_unique_observations) 
    mean_failure_rate = jnp.mean(jnp.array(all_failure_rates))
    return number_of_unique_observations, mean_failure_rate

def extract_trajectory_features(observations):
    """Extract features from trajectories for clustering"""
    # # Flatten the observations over time steps
    # if hasattr(trajectories.state, 'observation'):
    #     obs = trajectories.state.observation
    #     # Reshape to (num_test_cases, -1)
    #     if len(obs.shape) > 2:
    #         # For multi-dimensional observations, flatten everything except the first dimension
    #         features = jnp.reshape(obs, (obs.shape[0], obs.shape[1], -1))
    #         # Average over time dimension (dim 1)
    #         features = jnp.mean(features, axis=1)
    #     else:
    #         features = obs
    # else:
    #     # If no observations, use state directly
    #     features = jnp.array(trajectories.state)
    
    # # Include action probabilities if available
    # if hasattr(trajectories, 'action_probs'):
    #     action_probs = trajectories.action_probs
    #     # Reshape to (num_test_cases, -1)
    #     action_features = jnp.reshape(action_probs, (action_probs.shape[0], -1))
    #     features = jnp.concatenate([features, action_features], axis=1)
    
    # # Include rewards if available
    # if hasattr(trajectories, 'reward'):
    #     rewards = trajectories.reward
    #     # Reshape to (num_test_cases, -1)
    #     reward_features = jnp.reshape(rewards, (rewards.shape[0], -1))
    #     features = jnp.concatenate([features, reward_features], axis=1)
    
    return jnp.array(observations)

def run_pacmap_clustering(features, n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0):
    """Run PaCMAP clustering on trajectory features"""
    print(f"Running PaCMAP clustering on {features.shape[0]} test cases with {features.shape[1]} features")
    
    # Initialize and fit PaCMAP
    embedding = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio, 
        apply_pca=False
    )
    
    # Fit and transform the data
    result = embedding.fit_transform(features)
    return result

def visualize_clusters(embedding, fitnesses, output_path):
    """Visualize the PaCMAP clusters"""
    # Get fitnesses from test suite for coloring
    #fitnesses = np.array(test_suite.fitnesses)
    # Create a mask for valid test cases (fitness > 0)
    #valid_mask = fitnesses > 0
    
    # if np.sum(valid_mask) == 0:
    #     print("No valid test cases found in test suite")
    #     return
    # Get only the valid test cases
    #valid_embedding = embedding[valid_mask]
    #valid_fitnesses = fitnesses[valid_mask]
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot with fitness values as color
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=fitnesses,
        cmap='viridis',
        alpha=0.1,
        s=50
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('low = mptcs, high = topk')
    
    plt.title('PaCMAP Clustering of Test Cases')
    plt.xlabel('PaCMAP Dimension 1')
    plt.ylabel('PaCMAP Dimension 2')
    
    # Save the figure
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.close()


def execute_test_cases_and_collect_diversity_metrics_and_plot_clusters(test_suite, test_dict, number_of_batches, failure_rate_measure): 
    # Extract test cases from test suite
    genotypes = test_suite.genotypes 
    descriptors = test_suite.descriptors 
    rngs = test_suite.rngs 
    #indices = jnp.where(test_suite.fitnesses > 0)
    #indices_subsets = jnp.array_split(indices[0], number_of_batches)
    topk_genotypes = test_dict['genotypes']
    topk_descriptors = test_dict['descriptors']
    topk_rngs = test_dict['rngs']
    #indices = jnp.where(topk_storage['fitnesses'] > 0)
    #indices_subsets = jnp.array_split(indices[0], number_of_batches)
    
    mptcs_test_cases = TestCase(
        state = vec2state(genotypes),
        key = rngs
    )
    topk_test_cases = TestCase(
        state = vec2state(topk_genotypes),
        key = topk_rngs
    )
    mptcs_trajectories = test_case_simulator(test_params_stacked, mptcs_test_cases)
    observations_mptcs = mptcs_trajectories.state.observation.reshape((-1,) + mptcs_trajectories.state.observation.shape[-3:])
    observation_features_pac = jnp.mean(mptcs_trajectories.state.observation, axis=1).reshape((mptcs_trajectories.state.observation.shape[2], -1))
    features_mptcs = extract_trajectory_features(observation_features_pac)
    failure_rates_mptcs = failure_rate_measure(mptcs_trajectories)
    mean_failure_rate_mptcs = jnp.mean(failure_rates_mptcs)
    
    topk_trajectories = test_case_simulator(test_params_stacked, topk_test_cases)
    observations_topk = topk_trajectories.state.observation.reshape((-1,) + topk_trajectories.state.observation.shape[-3:])
    observation_features_pac = jnp.mean(topk_trajectories.state.observation, axis=1).reshape((topk_trajectories.state.observation.shape[2], -1))
    features_topk = extract_trajectory_features(observation_features_pac)
    failure_rates_topk = failure_rate_measure(topk_trajectories)
    mean_failure_rate_topk = jnp.mean(failure_rates_topk)
    
    labels = jnp.concatenate([jnp.zeros(features_mptcs.shape[0]), jnp.ones(features_topk.shape[0])])
    features = jnp.concatenate([features_mptcs, features_topk], axis=0)
    
    embedding_30 = run_pacmap_clustering(features, n_components=2, n_neighbors=30, MN_ratio=0.5, FP_ratio=2.0)
    embedding_20 = run_pacmap_clustering(features, n_components=2, n_neighbors=20, MN_ratio=0.5, FP_ratio=2.0)
    embedding_10 = run_pacmap_clustering(features, n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    embedding_5 = run_pacmap_clustering(features, n_components=2, n_neighbors=5, MN_ratio=0.5, FP_ratio=2.0)
    path = path_to_results + f"generation_{generation}/"
    if not os.path.exists(path):
        os.makedirs(path)
    output_path = path + f"pacmap_clusters_neighbors_5.png"
    visualize_clusters(embedding_5, labels, output_path)
    output_path = path + f"pacmap_clusters_neighbors_10.png"    
    visualize_clusters(embedding_10, labels, output_path)
    output_path = path + f"pacmap_clusters_neighbors_20.png"
    visualize_clusters(embedding_20, labels, output_path)
    output_path = path + f"pacmap_clusters_neighbors_30.png"
    visualize_clusters(embedding_30, labels, output_path)
    observations_bytes_topk = set(map(lambda x: x.tobytes(), observations_topk)) 
    observations_bytes_mptcs = set(map(lambda x: x.tobytes(), observations_mptcs)) 
    number_of_unique_observations_topk = len(observations_bytes_topk) 
    number_of_unique_observations_mptcs = len(observations_bytes_mptcs)
    
    
    return number_of_unique_observations_topk, number_of_unique_observations_mptcs, mean_failure_rate_topk, mean_failure_rate_mptcs
    
failure_rate_measure = test_case_selection.build_failure_rate_measure(failure_criterion)

previous_candidates = candidates

for generation in range(1, NUM_GENERATIONS+1):
    print(f"Generation {generation} of {NUM_GENERATIONS}")
    run_data["generation"].append(generation)
    key, generation_key = jax.random.split(key)
    
    # Generate candidates for both approaches
    candidates_mapelites = next_generation_mapelites(mptcs_test_suite, previous_candidates, generation_key)
    #candidates_topk = next_generation_topk(topk_storage, previous_candidates, generation_key)
    
    # Evaluate candidates
    trajs_mapelites = test_case_simulator(params_stacked, candidates_mapelites)
    obss_mapelites = trajs_mapelites.state.observation.reshape((-1,) + trajs_mapelites.state.observation.shape[-3:])
    #trajs_topk = test_case_simulator(params_stacked, candidates_topk)
    
    # Get descriptors and solvability
    tcp_scores_mapelites, descriptors_mapelites, solvability_mapelites = tcp_scorer(trajs_mapelites)
    #tcp_scores_topk, descriptors_topk, solvability_topk = tcp_scorer(trajs_topk)
    
    # Update MAP-Elites archive
    candidates_genotypes_mapelites = state2vec(candidates_mapelites.state)
    mptcs_test_suite, found_failures_mapelites, added_to_archive_mapelites, replaced_elite_mapelites = add_to_archive(
        mptcs_test_suite, candidates_genotypes_mapelites, descriptors_mapelites, tcp_scores_mapelites, candidates_mapelites.key
    )
        
    # Update top-k storage
    candidates_genotypes_topk = state2vec(candidates_mapelites.state)
    topk_storage = update_topk_storage(
        topk_storage,
        candidates_genotypes_topk,
        tcp_scores_mapelites,
        descriptors_mapelites,
        candidates_mapelites.key
    )
    
    # Collect data for both approaches
    #run_data = collect_data(run_data, mptcs_test_suite, trajs_mapelites, "mapelites", generation,
                           #jnp.sum(~solvability_mapelites), added_to_archive_mapelites, replaced_elite_mapelites, found_failures_mapelites)
    
    # Create a temporary MapElitesRepertoire for top-k to use the same evaluation functions
    temp_topk_archive = MapElitesRepertoire.init_default(
        genotype=reference_vector,
        centroids=test_suite_centroids,
        rng=jnp.zeros_like(jax.random.PRNGKey(123123))
    )
    temp_topk_archive = temp_topk_archive.add(
        batch_of_genotypes=topk_storage['genotypes'],
        batch_of_descriptors=topk_storage['descriptors'],
        batch_of_fitnesses=topk_storage['fitnesses'],
        batch_of_rngs=topk_storage['rngs']
    )
    
    #run_data = collect_data(run_data, temp_topk_archive, trajs_topk, "topk", generation,
    #                       jnp.sum(~solvability_topk), 0, 0, jnp.sum(tcp_scores_topk > 0))
    if generation % EVALUATION_INTERVAL == 0: 
        number_of_unique_observations_topk, number_of_unique_observations_mptcs, mean_failure_rate_topk, mean_failure_rate_mptcs = execute_test_cases_and_collect_diversity_metrics_and_plot_clusters(mptcs_test_suite, topk_storage, 10, failure_rate_measure)
        run_data["mptcs_num_unique_observations"].append(number_of_unique_observations_mptcs)
        run_data["mptcs_mean_failure_rate"].append(mean_failure_rate_mptcs)
        #number_of_unique_observations, mean_failure_rate = execute_test_cases_and_collect_diversity_metrics(topk_storage, "topk", 10, failure_rate_measure)
        run_data["topk_num_unique_observations"].append(number_of_unique_observations_topk)
        run_data["topk_mean_failure_rate"].append(mean_failure_rate_topk)
        print(f"Generation {generation} of {NUM_GENERATIONS}:")
        print(f"Number of unique observations: {number_of_unique_observations_topk} and {number_of_unique_observations_mptcs}")
        print(f"Mean failure rate: {mean_failure_rate_topk} and {mean_failure_rate_mptcs}")
    else: 
        run_data["mptcs_num_unique_observations"].append(-1)
        run_data["mptcs_mean_failure_rate"].append(-1)
        run_data["topk_num_unique_observations"].append(-1)
        run_data["topk_mean_failure_rate"].append(-1)
    # Update previous candidates for next generation
    previous_candidates = candidates_mapelites

# Save results
save_path = path_to_results
df = pd.DataFrame(run_data)
df.to_csv(save_path + f"run_data.csv", index=False) 