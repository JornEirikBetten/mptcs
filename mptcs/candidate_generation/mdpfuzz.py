import jax 
import jax.numpy as jnp 
import chex 
import haiku as hk 
import pgx 
from typing import Tuple, List, Dict, Any, Optional, Union, Sequence, NamedTuple, Callable

class TestCase(NamedTuple): 
    state: pgx.State
    key: jax.random.PRNGKey

class Population(NamedTuple):
    state: jnp.ndarray
    reward: jnp.ndarray
    freshness: jnp.ndarray
    sensitivity: jnp.ndarray

# def build_mdpfuzz_candidate_generator(
#     env_fn: Callable, 
#     mutate_fn: Callable, 
#     simulator: Callable,
#     reproducible_simulator: Callable,
#     failure_criterion: Callable,  
#     num_samples: int, 
#     num_iterations: int, 
#     total_number_of_simulation_steps: int
# ) -> Callable: 
#     """
#     Builds a candidate generation function where the candidates are generated through the MPDFuzz algorithm. 
#     """
    
#     def calculate_sensitivity(params: hk.Params, state: pgx.State, key: jax.random.PRNGKey, rewards_non_mutated: jnp.ndarray) -> Population: 
#         """Calculates how sensitive the reward is to a mutation of the state. 
#         """
#         mutate_key, simulate_key = jax.random.split(key)
#         mutated_state, magnitude_diff = mutate_fn(state, mutate_key)
#         #simulate_keys = jax.random.split(simulate_key, mutated_state.observation.shape[0])
#         trajs_mutated = simulator(params, mutated_state, simulate_key)
#         rewards_mutated = trajs_mutated.accumulated_rewards 
#         sensitivities = jnp.where(magnitude_diff, jnp.abs(rewards_mutated - rewards_non_mutated)/jnp.sqrt(magnitude_diff), jnp.abs(rewards_mutated - rewards_non_mutated))
#         return sensitivities 
    
#     #def build_iteration(mutate_fn: Callable, sensitivity_calculation: Callable, failure_criterion: Callable) -> Callable:
#     def iteration(parameters: hk.Params, corpus: Population, key: jax.random.PRNGKey, simulate_key: jax.random.PRNGKey, num_simulation_steps: int) -> Population:
#         draw_key, mutate_key, sensitivity_key = jax.random.split(key, 3)
#         # Calculate distribution over corpus 
#         distribution = corpus.sensitivity / jnp.sum(corpus.sensitivity)
#         # Draw from distribution 
#         selected_idx = jax.random.choice(draw_key, jnp.arange(corpus.state.observation.shape[0]), shape=(num_samples,), p=distribution)
#         selected_states = jax.tree.map(lambda x: x[selected_idx], corpus.state)
#         # Mutate  
#         mutated_states, mutations = mutate_fn(selected_states, mutate_key)
#         # Simulate
#         trajs = simulator(parameters, mutated_states, simulate_key)
#         num_simulation_steps += mutated_states.observation.shape[0] * trajs.state.observation.shape[0]
#         # Check for failures 
#         failures = failure_criterion(trajs)
#         failure_idxs = jnp.where(failures) 
#         failure_states = jax.tree.map(lambda x: x[failure_idxs], mutated_states)
#         if failure_states.observation.shape[0] > 0: 
#             failure_keys = jnp.stack([simulate_key] * failure_states.observation.shape[0]) 
#         else: 
#             failure_keys = jnp.array([])
#         rewards_mutated = trajs.accumulated_rewards 
#         rewards_less_idxs = jnp.where(rewards_mutated <= corpus.reward[selected_idx]) 
#         selected_states = jax.tree.map(lambda x: x[rewards_less_idxs], mutated_states)
#         selected_states_rewards = rewards_mutated[rewards_less_idxs]
#         # Update corpus 
#         sensitivities = calculate_sensitivity(parameters, selected_states, sensitivity_key, selected_states_rewards)
#         num_simulation_steps += selected_states.observation.shape[0] * trajs.state.observation.shape[0]
#         new_corpus = Population(
#             state=jax.tree.map(lambda x, y: jnp.concatenate([x, y]), corpus.state, selected_states),
#             reward=jnp.concatenate([corpus.reward, selected_states_rewards]),
#             freshness=jnp.concatenate([corpus.freshness, jnp.zeros_like(sensitivities)]),
#             sensitivity=jnp.concatenate([corpus.sensitivity, sensitivities])
#         )
#         if new_corpus.state.observation.shape[0] > 5000:
#             keep_indices = jnp.argsort(new_corpus.sensitivity)[-2500:]
#             keep_states = jax.tree.map(lambda x: x[keep_indices], new_corpus.state)
#             keep_rewards = new_corpus.reward[keep_indices]
#             keep_freshness = new_corpus.freshness[keep_indices]
#             keep_sensitivity = new_corpus.sensitivity[keep_indices]
            
#             new_corpus = Population(
#                 state=keep_states,
#                 reward=keep_rewards,
#                 freshness=keep_freshness,
#                 sensitivity=keep_sensitivity
#             )
#         return new_corpus, TestCase(state=failure_states, key=failure_keys), num_simulation_steps

#     def mdpfuzz_candidate_generation(parameters: hk.Params, key: jax.random.PRNGKey, simulate_key: jax.random.PRNGKey, num_simulation_steps: int) -> Tuple[List[TestCase], List[int]]: 
#         """
#         """
#         all_failures = [] 
#         steps_per_iteration = total_number_of_simulation_steps // num_iterations
#         step_counts = [] 
#         # Sample initial states 
#         key, _key = jax.random.split(key) 
#         initialize_key, sensitivity_key = jax.random.split(_key) 
#         initialize_keys = jax.random.split(initialize_key, num_samples)
#         initial_states = jax.vmap(env_fn.init)(initialize_keys)
#         # Simulate trajectories from the initial states 
#         trajs = simulator(parameters, initial_states, simulate_key)
#         print(trajs.rng.shape)
#         # Collect rewards 
#         rewards = trajs.accumulated_rewards 
#         # Check for failures 
#         failures = failure_criterion(trajs)
#         failure_idxs = jnp.where(failures)
#         failure_keys = trajs.rng[failure_idxs, :]
#         failure_states = jax.tree.map(lambda x: x[failure_idxs], initial_states)
#         print("Failure keys shape: ", failure_keys.shape)
#         print("Failure states shape: ", failure_states.observation.shape)
#         total_count = failure_states.observation.shape[0]
#         # Calculate sensitivities 
#         sensitivities = calculate_sensitivity(parameters, initial_states, sensitivity_key, rewards)
#         # Add number of simulation steps 
#         num_simulation_steps += initial_states.observation.shape[0] * trajs.state.observation.shape[0]
#         # Create corpus  
#         corpus = Population(
#             state=initial_states, 
#             reward=rewards, 
#             freshness=jnp.zeros_like(sensitivities),
#             sensitivity=sensitivities
#             )
#         # Append failures and step counts 
#         if failure_states.observation.shape[0] > 0: 
#             failure_test_cases = TestCase(state=failure_states, key=failure_keys)
#             all_failures.append(failure_test_cases); 
#         step_counts.append(num_simulation_steps)
#         # Iterate until the number of simulation steps is greater than the total number of simulation steps 
#         for j in range(num_iterations): 
#             print(f"Iteration {j + 1} out of {num_iterations} started")
#             num_simulation_steps = 0
#             failures = [] 
#             while num_simulation_steps < steps_per_iteration: 
#                 corpus, test_cases, num_simulation_steps = iteration(parameters, corpus, key, simulate_key, num_simulation_steps)
#                 total_count += test_cases.state.observation.shape[0]
#                 #print("Number of failures: ", total_count)
#                 #print("Number of states in corpus: ", corpus.state.observation.shape[0])
#                 if test_cases.state.observation.shape[0] > 0: 
#                     failures.append(test_cases)
#             print(f"Iteration {j +1} out of {num_iterations} completed")
#             step_counts.append(num_simulation_steps)
#             # If there are failures, concatenate them and append to all_failures 
#             if len(failures) > 0: 
#                 test_cases = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *failures)
#                 print("Shape of test cases states: ", test_cases.state.observation.shape)
#                 all_failures.append(test_cases)
#             else: 
#                 print("No failures in iteration ", j + 1)
#         all_failures = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *all_failures)
#         print("Finished mdpfuzz candidate generation")
#         return all_failures, step_counts
#     return mdpfuzz_candidate_generation

class Corpus(NamedTuple):
    candidate: TestCase
    reward: jnp.ndarray
    freshness: jnp.ndarray
    sensitivity: jnp.ndarray


def build_mdpfuzz_candidate_generator(
    env_fn: Callable, 
    mutate_fn: Callable, 
    test_case_simulator: Callable,
    failure_criterion: Callable,  
    num_samples: int, 
    num_iterations: int, 
    total_number_of_simulation_steps: int,
    max_corpus_size: int, 
    target_number_of_failures: int
) -> Callable: 
    """
    Builds a candidate generation function where the candidates are generated through the MPDFuzz algorithm. 
    """
    
    def calculate_sensitivity(params: hk.Params, test_cases: TestCase, mutate_key: jax.random.PRNGKey, rewards_non_mutated: jnp.ndarray) -> Population: 
        """Calculates how sensitive the reward is to a mutation of the state. 
        """
        mutated_state, magnitude_diff = mutate_fn(test_cases.state, mutate_key)
        # Using same key for the mutated test cases in sensitivity calculation
        mutated_test_cases = TestCase(state=mutated_state, key=test_cases.key) 
        trajs_mutated = test_case_simulator(params, mutated_test_cases)
        rewards_mutated = trajs_mutated.accumulated_rewards 
        sensitivities = jnp.where(magnitude_diff, jnp.abs(rewards_mutated - rewards_non_mutated)/jnp.sqrt(magnitude_diff), jnp.abs(rewards_mutated - rewards_non_mutated))
        return sensitivities 
    
    #def build_iteration(mutate_fn: Callable, sensitivity_calculation: Callable, failure_criterion: Callable) -> Callable:
    def iteration(parameters: hk.Params, corpus: Population, key: jax.random.PRNGKey, num_simulation_steps: int) -> Population:
        draw_key, mutate_key, sensitivity_key = jax.random.split(key, 3)
        # Calculate distribution over corpus 
        distribution = corpus.sensitivity / jnp.sum(corpus.sensitivity)
        # Draw from distribution 
        selected_idx = jax.random.choice(draw_key, jnp.arange(corpus.candidate.state.observation.shape[0]), shape=(num_samples,), p=distribution)
        selected_test_cases = jax.tree.map(lambda x: x[selected_idx], corpus.candidate)
        # Mutate  
        mutated_states, mutations = mutate_fn(selected_test_cases.state, mutate_key) 
        keys = jax.random.split(mutate_key, mutated_states.observation.shape[0])
        mutated_candidates = TestCase(state=mutated_states, key=keys)
        # Simulate
        trajs = test_case_simulator(parameters, mutated_candidates)
        # Check if the first trajectory keys are the same as the input test case keys 
        assert jnp.all(trajs.rng[0] == keys)
        num_simulation_steps += mutated_states.observation.shape[0] * trajs.state.observation.shape[0]
        # Check for failures 
        failures = failure_criterion(trajs)
        failure_idxs = jnp.where(failures) 
        failure_candidates = jax.tree.map(lambda x: x[failure_idxs], mutated_candidates)
        rewards_mutated = trajs.accumulated_rewards 
        rewards_less_idxs = jnp.where(rewards_mutated <= corpus.reward[selected_idx]) 
        selected_states = jax.tree.map(lambda x: x[rewards_less_idxs], mutated_states)
        selected_test_cases = jax.tree.map(lambda x: x[rewards_less_idxs], mutated_candidates)
        selected_states_rewards = rewards_mutated[rewards_less_idxs]
        # Update corpus 
        sensitivities = calculate_sensitivity(parameters, selected_test_cases, sensitivity_key, selected_states_rewards)
        num_simulation_steps += selected_states.observation.shape[0] * trajs.state.observation.shape[0]
        new_corpus = Corpus(
            candidate=jax.tree.map(lambda x, y: jnp.concatenate([x, y]), corpus.candidate, selected_test_cases),
            reward=jnp.concatenate([corpus.reward, selected_states_rewards]),
            freshness=jnp.concatenate([corpus.freshness, jnp.zeros_like(sensitivities)]),
            sensitivity=jnp.concatenate([corpus.sensitivity, sensitivities])
        )
        if new_corpus.candidate.state.observation.shape[0] > max_corpus_size:
            keep_indices = jnp.argsort(new_corpus.sensitivity)[-max_corpus_size//2:]
            keep_candidates = jax.tree.map(lambda x: x[keep_indices], new_corpus.candidate)
            keep_rewards = new_corpus.reward[keep_indices]
            keep_freshness = new_corpus.freshness[keep_indices]
            keep_sensitivity = new_corpus.sensitivity[keep_indices]
            
            new_corpus = Corpus(
                candidate=keep_candidates,
                reward=keep_rewards,
                freshness=keep_freshness,
                sensitivity=keep_sensitivity
            )
        return new_corpus, failure_candidates, num_simulation_steps

    def mdpfuzz_candidate_generation(parameters: hk.Params, key: jax.random.PRNGKey, num_simulation_steps: int) -> Tuple[List[TestCase], List[int]]: 
        
        all_failures = [] 
        steps_per_iteration = total_number_of_simulation_steps // num_iterations
        step_counts = [] 
        num_tcs = [] 
        # Sample initial states 
        key, initialize_key, sensitivity_key, simulate_key = jax.random.split(key, 4) 
        initialize_keys = jax.random.split(initialize_key, num_samples)
        initial_states = jax.vmap(env_fn.init)(initialize_keys)
        
        simulation_keys = jax.random.split(simulate_key, num_samples)
        candidates = TestCase(
            state=initial_states, 
            key=simulation_keys
        )
        
        # Simulate trajectories from the initial states 
        trajs = test_case_simulator(parameters, candidates)
        # Collect rewards 
        rewards = trajs.accumulated_rewards 
        # Check for failures 
        failures = failure_criterion(trajs)
        failure_idxs = jnp.where(failures)
        # Extract eventual failure test cases 
        failure_test_cases = jax.tree.map(lambda x: x[failure_idxs], candidates)
        
        total_count = failure_test_cases.state.observation.shape[0]
        # Calculate sensitivities 
        sensitivities = calculate_sensitivity(parameters, candidates, sensitivity_key, rewards)
        # Add number of simulation steps 
        num_simulation_steps += initial_states.observation.shape[0] * trajs.state.observation.shape[0]
        # Create corpus  
        corpus = Corpus(
            candidate=candidates, 
            reward=rewards, 
            freshness=jnp.zeros_like(sensitivities),
            sensitivity=sensitivities
            )
        # Append failures and step counts 
        if failure_test_cases.state.observation.shape[0] > 0: 
            all_failures.append(failure_test_cases); 
        step_counts.append(num_simulation_steps)
        num_tcs.append(failure_test_cases.state.observation.shape[0])
        # Iterate until the number of simulation steps is greater than the total number of simulation steps 
        for j in range(num_iterations): 
            print(f"Iteration {j + 1} out of {num_iterations} started")
            num_simulation_steps = 0
            failures = [] 
            while num_simulation_steps < steps_per_iteration: 
                key, _key = jax.random.split(key)
                corpus, test_cases, num_simulation_steps = iteration(parameters, corpus, key, num_simulation_steps)
                total_count += test_cases.state.observation.shape[0]
                print(f"Number of failures: {total_count}")
                print(f"Number of states in corpus: {corpus.candidate.state.observation.shape[0]}")
                if test_cases.state.observation.shape[0] > 0: 
                    failures.append(test_cases)
            print(f"Iteration {j +1} out of {num_iterations} completed")
            step_counts.append(num_simulation_steps)
    
            # If there are failures, concatenate them and append to all_failures 
            if len(failures) > 0: 
                test_cases = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *failures)
                all_failures.append(test_cases)
                num_tcs.append(test_cases.state.observation.shape[0])
            else: 
                print("No failures in iteration ", j + 1)
                num_tcs.append(0)
                
            total_failures = jnp.sum(jnp.array(num_tcs))
            if total_failures > target_number_of_failures:
                break
        all_failures = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *all_failures)
        run_info = {
            "found_tcs": num_tcs,
            "step_counts": step_counts
        }
        print("Finished mdpfuzz candidate generation")
        return all_failures, run_info
    return mdpfuzz_candidate_generation