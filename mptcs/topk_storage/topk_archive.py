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
        self.fitnesses = jnp.zeros(k)
        self.genotypes = jnp.zeros((k, reference_vector.shape[0]))
        self.descriptors = jnp.zeros((k, 2))
        self.rngs = jnp.zeros((k, 2), dtype=jnp.uint32)

    def add(self, batch_of_genotypes: jnp.ndarray, batch_of_descriptors: jnp.ndarray, batch_of_fitnesses: jnp.ndarray, batch_of_rngs: jnp.ndarray): 
        all_fitnesses = jnp.concatenate([self.fitnesses, batch_of_fitnesses])
        all_genotypes = jnp.concatenate([self.genotypes, batch_of_genotypes])
        all_descriptors = jnp.concatenate([self.descriptors, batch_of_descriptors])
        all_rngs = jnp.concatenate([self.rngs, batch_of_rngs])

        sorted_indices = jnp.argsort(all_fitnesses)[::-1]
        top_k_indices = sorted_indices[:self.K]

        self.fitnesses = all_fitnesses[top_k_indices]
        self.genotypes = all_genotypes[top_k_indices]
        self.descriptors = all_descriptors[top_k_indices]
        self.rngs = all_rngs[top_k_indices]
        
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
    