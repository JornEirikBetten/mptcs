import jax 
import jax.numpy as jnp  
from typing import Callable, List
from mptcs.simulators import Trajectory
from mptcs.candidate_generation import TestCase
import haiku as hk 
import qdax_modified 

def build_observation_variance_measure() -> Callable: 
    def observation_variance_calculation(trajectories: Trajectory) -> jnp.ndarray: 
        """
        trajectories.observation.shape = (num_policies, max_steps, num_starting_states, observation_shape)
        """
        observation_variance = jnp.var(trajectories.state.observation, axis=(0, 3, 4, 5))
        mean_observation_variance = jnp.mean(observation_variance, axis=0)
        return mean_observation_variance
    return observation_variance_calculation

def build_policy_uncertainty_measure() -> Callable: 
    def policy_uncertainty_calculation(trajectories: Trajectory) -> jnp.ndarray: 
        """The policy uncertainty measure is the mean of the entropy of the action distributions in the initial states. 
        """     
        action_distributions = trajectories.action_distribution[:, 0, :, :]
        mean_policy_uncertainty = -jnp.mean(jnp.sum(jnp.where(action_distributions > 0, action_distributions * jnp.log(action_distributions), 0), axis=-1), axis=0)
        return mean_policy_uncertainty
    return policy_uncertainty_calculation
    

def build_test_case_priority_measure(failure_criterion: Callable) -> Callable: 
    def test_case_priority_measure(trajectories: Trajectory) -> jnp.ndarray:
        """
        Trajectories from num_states states using num_policies policies that all have lengths max_steps. 
        """
        failures = failure_criterion(trajectories)
        confirmed_solvable = jnp.sum(failures, axis=0) < failures.shape[0]
        number_of_policies = failures.shape[0]
        return confirmed_solvable * jnp.sum(failures, axis=0) / number_of_policies
    return test_case_priority_measure 


def build_failure_rate_measure(failure_criterion: Callable) -> Callable: 
    def failure_rate_measure(trajectories: Trajectory) -> jnp.ndarray: 
        failures = failure_criterion(trajectories)
        return jnp.sum(failures, axis=0) / failures.shape[0]
    return failure_rate_measure  


def build_solvability_screener(failure_criterion: Callable) -> Callable: 
    def solvability_screener(trajectories: Trajectory) -> jnp.ndarray: 
        failures = failure_criterion(trajectories)
        confirmed_solvable = jnp.sum(failures, axis=0) < failures.shape[0]
        return confirmed_solvable
    return solvability_screener 


def build_tcp_scorer_on_trajectories(failure_criterion: Callable) -> Callable:  
    observation_variance_calculation = build_observation_variance_measure() 
    policy_uncertainty_calculation = build_policy_uncertainty_measure() 
    test_case_priority_measure = build_test_case_priority_measure(failure_criterion)
    solvability_screener = build_solvability_screener(failure_criterion)
    def tcp_scoring_trajectories(trajectories: Trajectory) -> jnp.ndarray: 
        test_case_priority = test_case_priority_measure(trajectories)
        solvability = solvability_screener(trajectories)
        # Calculate policy uncertainty scores 
        policy_uncertainty = policy_uncertainty_calculation(trajectories)
        # Calculate the mean observation variance from each state 
        observation_variance = observation_variance_calculation(trajectories)
        # Concatenate descriptors into behavioral descriptor vectors 
        descriptors = jnp.array([observation_variance, policy_uncertainty]).T
        return test_case_priority, descriptors, solvability
    return tcp_scoring_trajectories

def build_pats_candidate_evaluator(batched_simulator: Callable, failure_criterion: Callable):
    # Test case priority measure 
    test_case_priority_measure = build_test_case_priority_measure(failure_criterion)
    observation_variance_calculation = build_observation_variance_measure() 
    policy_uncertainty_calculation = build_policy_uncertainty_measure() 
    solvability_screener = build_solvability_screener(failure_criterion)
    # Candidate evaluation 
    def evaluate_candidates(stacked_parameters: hk.Params, candidates: TestCase) -> jnp.ndarray: 
        # Simulate trajectories by running all policies from each candidate in parallel
        # Generates trajectories where 
        # trajectories.state.observation.shape = (num_policies, max_steps, num_starting_states, observation_shape) 
        trajectories = batched_simulator(stacked_parameters, candidates)
        
        # Calculate test case priority scores for each state 
        test_case_priority = test_case_priority_measure(trajectories)
        solvability = solvability_screener(trajectories)
        # Calculate behavioral descriptors for QD grid 
        # Calculate policy uncertainty scores 
        policy_uncertainty = policy_uncertainty_calculation(trajectories)
        # Calculate the mean observation variance from each state 
        observation_variance = observation_variance_calculation(trajectories)
        # Concatenate descriptors into behavioral descriptor vectors 
        descriptors = jnp.array([observation_variance, policy_uncertainty]).T
        
        # Return test case priority and descriptors 
        return test_case_priority, descriptors, solvability
    return evaluate_candidates 

def build_pats_candidate_evaluator_with_step_count(batched_simulator: Callable, failure_criterion: Callable):
    # Test case priority measure 
    test_case_priority_measure = build_test_case_priority_measure(failure_criterion)
    observation_variance_calculation = build_observation_variance_measure() 
    policy_uncertainty_calculation = build_policy_uncertainty_measure() 
    solvability_screener = build_solvability_screener(failure_criterion)
    # Candidate evaluation 
    def evaluate_candidates(stacked_parameters: hk.Params, candidates: TestCase) -> jnp.ndarray: 
        # Simulate trajectories by running all policies from each candidate in parallel
        # Generates trajectories where 
        # trajectories.state.observation.shape = (num_policies, max_steps, num_starting_states, observation_shape) 
        trajectories = batched_simulator(stacked_parameters, candidates)
        simulation_steps = jnp.sum(~trajectories.state.terminated)
        # Calculate test case priority scores for each state 
        test_case_priority = test_case_priority_measure(trajectories)
        solvability = solvability_screener(trajectories)
        # Calculate behavioral descriptors for QD grid 
        # Calculate policy uncertainty scores 
        policy_uncertainty = policy_uncertainty_calculation(trajectories)
        # Calculate the mean observation variance from each state 
        observation_variance = observation_variance_calculation(trajectories)
        # Concatenate descriptors into behavioral descriptor vectors 
        descriptors = jnp.array([observation_variance, policy_uncertainty]).T
        
        # Return test case priority and descriptors 
        return test_case_priority, descriptors, solvability, simulation_steps
    return evaluate_candidates 

def build_test_case_evaluator(batched_simulator: Callable, failure_criterion: Callable):  
    # Failure rate measure 
    failure_rate_measure = build_failure_rate_measure(failure_criterion) 
    def evaluate_test_case(stacked_test_parameters: hk.Params, test_case: TestCase) -> jnp.ndarray: 
        # Simulate trajectories by running all policies from each candidate in parallel
        # Generates trajectories where 
        # trajectories.state.observation.shape = (num_policies, max_steps, num_starting_states, observation_shape) 
        trajectories = batched_simulator(stacked_test_parameters, test_case)
        # Find failure rates of the test cases 
        failure_rates = failure_rate_measure(trajectories) 
        return failure_rates 
    return evaluate_test_case 
    
    
# def build_test_case_insertion_into_diversity_map() -> Callable:
    
#     def test_case_insertion_into_diversity_map(diversity_map: jnp.ndarray, test_case: TestCase, test_case_score: jnp.ndarray, behavioral_descriptor: jnp.ndarray) -> jnp.ndarray: 
        
#     return test_case_insertion_into_diversity_map 
