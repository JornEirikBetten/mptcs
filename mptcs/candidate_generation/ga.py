import jax
import jax.numpy as jnp
import haiku as hk
import chex 
from typing import Tuple, List, Dict, Any, Optional, Union, Sequence, NamedTuple, Callable
from .mdpfuzz import TestCase, Corpus
from .mutation_utils import build_mutate_fn
import pgx 

class TestCase(NamedTuple): 
    state: pgx.State
    key: jax.random.PRNGKey

class Population(NamedTuple): 
    candidates: TestCase
    fitness: chex.Array 
    



def build_single_policy_genetic_algorithm(
    env_fn: Callable, 
    test_case_simulator: Callable, 
    failure_criterion: Callable, 
    mutate_fn: Callable, 
): 
    """
    Builds a genetic algorithm candidate generation function. 
    """
    
    
    def ga_candidate_generation(
        parameters: hk.Params, 
        key: jax.random.PRNGKey, 
        config: Dict[str, Any]
    ): 
        """
        Run the genetic algorithm to generate test cases. 
        """
        run_info = {
            "mean_fitness": [], 
            "max_fitness": [], 
            "min_fitness": [], 
            "num_failures": [], 
            "num_simulation_steps": [],     
            "num_test_cases": []
        }
        # Initialize population 
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, config["initial_population_size"])
        initial_states = jax.vmap(env_fn.init)(init_keys)
        key, _key = jax.random.split(key) 
        run_keys = jax.random.split(_key, config["initial_population_size"])
        initial_test_cases = TestCase(state=initial_states, key=run_keys)
        # Run initial simulation 
        trajs = test_case_simulator(parameters, initial_test_cases)
        failures = failure_criterion(trajs)
        fitness = failures 
        population = Population(
            candidates=initial_test_cases, 
            fitness=fitness)
        run_info["num_test_cases"].append(config["initial_population_size"])
        run_info["num_simulation_steps"].append(jnp.sum(~trajs.state.terminated))
        run_info["num_failures"].append(jnp.sum(fitness))
        run_info["mean_fitness"].append(jnp.mean(fitness))
        run_info["max_fitness"].append(jnp.max(fitness))
        run_info["min_fitness"].append(jnp.min(fitness))
        
        def selection(population: Population, sample_key: jax.random.PRNGKey) -> TestCase: 
            """
            Select the fittest individuals from the population. 
            """
            fitness = population.fitness + 1e-6
            sampled_indices = jax.random.choice(sample_key, jnp.arange(fitness.shape[0]), shape=(config["selection_size"],), p=fitness/jnp.sum(fitness))
            sampled_candidates = jax.tree.map(lambda x: x[sampled_indices], population.candidates)
            return sampled_candidates
        
        def trim_population(population: Population) -> Population: 
            """
            Trim the population to the size of the initial population. 
            """
            fitness = population.fitness 
            sorted_indices = jnp.argsort(fitness)[-config["max_population_size"]:]
            new_population = Population(
                candidates=jax.tree.map(lambda x: x[sorted_indices], population.candidates), 
                fitness=jax.tree.map(lambda x: x[sorted_indices], population.fitness))
            return new_population
        
        
        def crossover(population: Population) -> Population: 
            """
            Perform crossover on the population. 
            """
            pass 
        
        def mutation(candidates: TestCase, mutate_key: jax.random.PRNGKey) -> TestCase: 
            candidate_states = candidates.state 
            mutated_states, _ = mutate_fn(candidate_states, mutate_key)
            mutated_test_cases = TestCase(state=mutated_states, key=candidates.key)
            return mutated_test_cases
        
        for generation in range(config["num_generations"]): 
            key, mutate_key, sample_key = jax.random.split(key, 3)
            # Selection 
            selected_candidates = selection(population, sample_key)
            # Mutation 
            mutated_candidates = mutation(selected_candidates, mutate_key)
            # Run simulation 
            trajs = test_case_simulator(parameters, mutated_candidates)
            failures = failure_criterion(trajs)
            fitness = failures 
            
            candidates_in_population = population.candidates  
            fitness_in_population = population.fitness 
            new_states = jax.tree.map(lambda x, y: jnp.concatenate([x, y]), candidates_in_population.state, mutated_candidates.state) 
            new_keys = jax.tree.map(lambda x, y: jnp.concatenate([x, y]), candidates_in_population.key, mutated_candidates.key)
            new_candidates = TestCase(state=new_states, key=new_keys)
            new_fitness = jnp.concatenate([fitness_in_population, fitness])
            population = Population(
                candidates=new_candidates, 
                fitness=new_fitness)
            
            if population.fitness.shape[0] > config["max_population_size"]: 
                population = trim_population(population)
            
            run_info["num_test_cases"].append(population.candidates.key.shape[0])
            run_info["num_simulation_steps"].append(jnp.sum(~trajs.state.terminated))
            run_info["num_failures"].append(jnp.sum(population.fitness))
            run_info["mean_fitness"].append(jnp.mean(population.fitness))
            run_info["max_fitness"].append(jnp.max(population.fitness))
            run_info["min_fitness"].append(jnp.min(population.fitness))
            if generation % 10 == 0: 
                print(f"Generation {generation + 1} completed:")
                print(f"Number of test cases: {run_info['num_test_cases'][-1]}")
                print(f"Number of simulation steps: {run_info['num_simulation_steps'][-1]}")
                print(f"Number of failures: {run_info['num_failures'][-1]}")
                print(f"Mean fitness: {run_info['mean_fitness'][-1]:.2f}")
                print(f"Max fitness: {run_info['max_fitness'][-1]:.2f}")
                print(f"Min fitness: {run_info['min_fitness'][-1]:.2f}")
            
        return population, run_info 
    
    return ga_candidate_generation


def build_multi_policy_genetic_algorithm(
    env_fn: Callable, 
    test_case_simulator: Callable, 
    test_case_priority: Callable, 
    mutate_fn: Callable, 
): 
    """
    Builds a genetic algorithm candidate generation function for multi-policy environments. 
    """
    def ga_candidate_generation(
        stacked_parameters: hk.Params, 
        key: jax.random.PRNGKey, 
        config: Dict[str, Any]
    ): 
        """
        Run the genetic algorithm to generate test cases. 
        """
        run_info = {
            "mean_fitness": [], 
            "max_fitness": [], 
            "min_fitness": [], 
            "num_failures": [], 
            "step_counts": [], 
            "num_test_cases": []
        }
        # Initialize population 
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, config["initial_population_size"])
        initial_states = jax.vmap(env_fn.init)(init_keys)
        key, _key = jax.random.split(key) 
        run_keys = jax.random.split(_key, config["initial_population_size"])
        initial_test_cases = TestCase(state=initial_states, key=run_keys)
        
        print(initial_test_cases.state.observation.shape[0])
        print(initial_test_cases.key.shape[0])
        # Run initial simulation 
        fitness, solvabilities, descriptors, simulation_steps = test_case_priority(stacked_parameters,initial_test_cases)
        population = Population(
            candidates=initial_test_cases, 
            fitness=fitness)
        run_info["num_test_cases"].append(config["initial_population_size"])
        run_info["step_counts"].append(simulation_steps)
        run_info["num_failures"].append(jnp.sum(fitness) * config["num_policies"])
        run_info["mean_fitness"].append(jnp.mean(fitness))
        run_info["max_fitness"].append(jnp.max(fitness))
        run_info["min_fitness"].append(jnp.min(fitness))
        
        def selection(population: Population, sample_key: jax.random.PRNGKey) -> TestCase: 
            """
            Sample candidates from the population based on fitness. 
            """
            fitness = population.fitness + 1e-6
            sampled_indices = jax.random.choice(sample_key, jnp.arange(fitness.shape[0]), shape=(config["selection_size"],), p=fitness/jnp.sum(fitness))
            sampled_candidates = jax.tree.map(lambda x: x[sampled_indices], population.candidates)
            return sampled_candidates
        
        
        def mutation(candidates: TestCase, mutate_key: jax.random.PRNGKey) -> TestCase: 
            """
            Mutate the candidates. 
            """
            candidate_states = candidates.state 
            mutated_states, _ = mutate_fn(candidate_states, mutate_key)
            mutated_test_cases = TestCase(state=mutated_states, key=candidates.key)
            return mutated_test_cases
    
    
        def trim_population(population: Population) -> Population: 
            """
            Trim the population to the size of the initial population. 
            """
            fitness = population.fitness 
            sorted_indices = jnp.argsort(fitness)[-config["max_population_size"]:]
            new_population = Population(
                candidates=jax.tree.map(lambda x: x[sorted_indices], population.candidates), 
                fitness=jax.tree.map(lambda x: x[sorted_indices], population.fitness))
            return new_population
        
        for generation in range(config["num_generations"]): 
            key, mutate_key, sample_key = jax.random.split(key, 3)
            # Selection 
            selected_candidates = selection(population, sample_key)
            # Mutation 
            mutated_candidates = mutation(selected_candidates, mutate_key)
            # Run simulation 
            fitness, descriptors, solvability, simulation_steps = test_case_priority(stacked_parameters, mutated_candidates)
            candidates_in_population = population.candidates    
            fitness_in_population = population.fitness 
            new_states = jax.tree.map(lambda x, y: jnp.concatenate([x, y]), candidates_in_population.state, mutated_candidates.state) 
            new_keys = jax.tree.map(lambda x, y: jnp.concatenate([x, y]), candidates_in_population.key, mutated_candidates.key)
            new_candidates = TestCase(state=new_states, key=new_keys)
            new_fitness = jnp.concatenate([fitness_in_population, fitness])
            new_population = Population(
                candidates=new_candidates, 
                fitness=new_fitness)
            if new_population.fitness.shape[0] > config["max_population_size"]: 
                new_population = trim_population(new_population)
            population = new_population 
            run_info["num_test_cases"].append(population.candidates.key.shape[0])
            run_info["num_failures"].append(jnp.sum(population.fitness) * config["num_policies"])
            run_info["mean_fitness"].append(jnp.mean(population.fitness))
            run_info["max_fitness"].append(jnp.max(population.fitness))
            run_info["min_fitness"].append(jnp.min(population.fitness))
            run_info["step_counts"].append(simulation_steps)
            if generation % 10 == 0: 
                print(f"Generation {generation + 1} completed:")
                print(f"Number of test cases: {run_info['num_test_cases'][-1]}")
                print(f"Number of steps: {run_info['step_counts'][-1]}")
                print(f"Number of failures: {run_info['num_failures'][-1]}")
                print(f"Mean fitness: {run_info['mean_fitness'][-1]:.2f}")
                print(f"Max fitness: {run_info['max_fitness'][-1]:.2f}")
                print(f"Min fitness: {run_info['min_fitness'][-1]:.2f}")
                
        print(f"Generation {generation + 1} completed:")
        print(f"Number of test cases: {run_info['num_test_cases'][-1]}")
        print(f"Number of steps: {run_info['step_counts'][-1]}")
        print(f"Number of failures: {run_info['num_failures'][-1]}")
        print(f"Mean fitness: {run_info['mean_fitness'][-1]:.2f}")
        print(f"Max fitness: {run_info['max_fitness'][-1]:.2f}")
        print(f"Min fitness: {run_info['min_fitness'][-1]:.2f}")
        return population, run_info 
    
    return ga_candidate_generation
            