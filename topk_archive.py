import jax
import jax.numpy as jnp
import pgx as pgx

class TestCase(NamedTuple): 
    state: pgx.State
    key: jax.random.PRNGKey



class TopKStorage: 
    def __init__(self, k: int): 
        self.K = k 
        self.fitnesses = []
        self.genotypes = []
        self.descriptors = []
        self.rngs = []

    def add(self, fitness, genotype, descriptor, rng): 
        all_fitnesses = jnp.concatenate([self.fitnesses, [fitness]])
        all_genotypes = jnp.concatenate([self.genotypes, [genotype]])
        all_descriptors = jnp.concatenate([self.descriptors, [descriptor]])
        all_rngs = jnp.concatenate([self.rngs, [rng]])

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