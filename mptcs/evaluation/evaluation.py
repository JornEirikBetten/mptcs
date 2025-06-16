from typing import Callable, Tuple, List, Dict, Any, Optional, NamedTuple
import jax
import jax.numpy as jnp
import pgx as pgx

class TestCase(NamedTuple): 
    state: pgx.State
    key: jax.random.PRNGKey


def build_evaluation_of_test_suite(test_case_simulator, vec2state_and_reshape, failure_rate_measure, number_of_batches): 
    def evaluate_test_suite(test_suite, test_params, empty_validation_archive):
        
        validation_archive = empty_validation_archive 
        genotypes = test_suite.genotypes 
        descriptors = test_suite.descriptors 
        rngs = test_suite.rngs 
        indices = jnp.where(test_suite.fitnesses > 0)
        indices_subsets = jnp.array_split(indices[0], number_of_batches)
        
        for indices_subset in indices_subsets: 
            genotypes_subset = genotypes[indices_subset]
            descriptors_subset = descriptors[indices_subset]
            rngs_subset = rngs[indices_subset]
            test_cases_subset = TestCase(
                state = vec2state_and_reshape(genotypes_subset),
                key = rngs_subset
            )
            trajs = test_case_simulator(test_params, test_cases_subset) 
            failure_rates = failure_rate_measure(trajs)
            validation_archive = validation_archive.add(
                batch_of_genotypes=genotypes_subset,
                batch_of_descriptors=descriptors_subset,
                batch_of_fitnesses=failure_rates,
                batch_of_rngs=rngs_subset
            )
        return validation_archive
    return evaluate_test_suite 


def build_evaluation_of_test_suite_collect_policy_statistics(test_case_simulator, vec2state_and_reshape, failure_rate_measure, number_of_batches): 
    def evaluate_test_suite(test_suite, test_params, empty_validation_archive):
        validation_archive = empty_validation_archive 
        genotypes = test_suite.genotypes 
        descriptors = test_suite.descriptors 
        rngs = test_suite.rngs 
        indices = jnp.where(test_suite.fitnesses > 0)
        indices_subsets = jnp.array_split(indices[0], number_of_batches)

        for indices_subset in indices_subsets: 
            genotypes_subset = genotypes[indices_subset]
            descriptors_subset = descriptors[indices_subset]
            rngs_subset = rngs[indices_subset]      
            test_cases_subset = TestCase(
                state = vec2state_and_reshape(genotypes_subset),
                key = rngs_subset
            )
            trajs = test_case_simulator(test_params, test_cases_subset) 
            failure_rates = failure_rate_measure(trajs) 
            validation_archive = validation_archive.add(
                batch_of_genotypes=genotypes_subset,
                batch_of_descriptors=descriptors_subset,
                batch_of_fitnesses=failure_rates,
                batch_of_rngs=rngs_subset
            )   
        return validation_archive
    return evaluate_test_suite