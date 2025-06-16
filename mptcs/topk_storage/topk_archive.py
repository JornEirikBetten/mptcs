import jax
import jax.numpy as jnp
import pgx as pgx

from typing import NamedTuple

class TestCase(NamedTuple): 
    state: pgx.State
    key: jax.random.PRNGKey




class TopKStorage: 
    def __init__(self, k: int, reference_vector: jnp.ndarray): 
        self.K = k 
        self.fitnesses = -jnp.ones(k)*jnp.inf
        self.genotypes = jnp.zeros((k, reference_vector.shape[0]))
        self.descriptors = jnp.zeros((k, 2))
        self.rngs = jnp.zeros((k, 2), dtype=jnp.uint32)

    def add(self, batch_of_genotypes: jnp.ndarray, batch_of_descriptors: jnp.ndarray, batch_of_fitnesses: jnp.ndarray, batch_of_rngs: jnp.ndarray): 
        all_fitnesses = jnp.concatenate([self.fitnesses, batch_of_fitnesses])
        all_genotypes = jnp.concatenate([self.genotypes, batch_of_genotypes])
        all_descriptors = jnp.concatenate([self.descriptors, batch_of_descriptors])
        all_rngs = jnp.concatenate([self.rngs, batch_of_rngs])
        
        # Get unique descriptors - keep only the one with highest fitness for each unique descriptor
        # First create a string representation of descriptors for hashing
        # descriptor_keys = jnp.array([hash(tuple(d)) for d in all_descriptors])
        # unique_keys, unique_indices = jnp.unique(descriptor_keys, return_index=True)
        unique_descriptors, unique_indices = jnp.unique(jnp.sum(all_descriptors, axis=1), return_index=True)
        # For duplicate descriptors, we'll only keep the one with highest fitness
        unique_fitnesses = all_fitnesses[unique_indices]
        unique_genotypes = all_genotypes[unique_indices]
        unique_descriptors = all_descriptors[unique_indices]
        unique_rngs = all_rngs[unique_indices]
        if len(unique_descriptors) > self.K:
            # Sort by fitness and take top-k
            sorted_indices = jnp.argsort(unique_fitnesses)[::-1]
            top_k_indices = sorted_indices[:self.K]
        else:
            # Fill in the rest with filler values so that the archive is stays in the size of K
            unique_fitnesses = jnp.concatenate([unique_fitnesses, -jnp.ones(self.K - len(unique_descriptors))*jnp.inf])
            unique_genotypes = jnp.concatenate([unique_genotypes, jnp.zeros((self.K - len(unique_descriptors), self.genotypes.shape[1]))])
            unique_descriptors = jnp.concatenate([unique_descriptors, jnp.zeros((self.K - len(unique_descriptors), self.descriptors.shape[1]))])
            unique_rngs = jnp.concatenate([unique_rngs, jnp.zeros((self.K - len(unique_descriptors), self.rngs.shape[1]), dtype=jnp.uint32)])
            top_k_indices = jnp.arange(len(unique_descriptors))

        self.fitnesses = unique_fitnesses[top_k_indices]
        self.genotypes = unique_genotypes[top_k_indices]
        self.descriptors = unique_descriptors[top_k_indices]
        self.rngs = unique_rngs[top_k_indices]
        
        return self 
    
    def save(self, path: str): 
        saving_fitnesses = jnp.array(self.fitnesses)
        saving_genotypes = jnp.array(self.genotypes)
        saving_descriptors = jnp.array(self.descriptors)
        saving_rngs = jnp.array(self.rngs)

        jnp.save(path + "fitnesses.npy", saving_fitnesses)
        jnp.save(path + "genotypes.npy", saving_genotypes)
        jnp.save(path + "descriptors.npy", saving_descriptors)
        jnp.save(path + "rngs.npy", saving_rngs)
        
        print(f"Saved topk archive to {path}")

    def load(self, path: str): 
        self.fitnesses = jnp.load(path + "fitnesses.npy")
        self.genotypes = jnp.load(path + "genotypes.npy")
        self.descriptors = jnp.load(path + "descriptors.npy")
        self.rngs = jnp.load(path + "rngs.npy")
        
        return self 
    