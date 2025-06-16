import jax 
import jax.numpy as jnp 
import chex 
import pgx 
import haiku as hk 

from typing import Dict, Any, Tuple, Callable, Optional, List, Union, Sequence, NamedTuple


class TestCase(NamedTuple): 
    state: pgx.State
    key: jax.random.PRNGKey


class Trajectory(NamedTuple): 
    state: pgx.State
    action: chex.Array
    accumulated_rewards: chex.Array
    action_distribution: chex.Array
    rng: jax.random.PRNGKey


def build_standard_pgx_simulator(env_fn: Callable, forward_pass: Callable, max_steps: int) -> Callable: 
    """
    Builds a standard simulator that runs a policy for maximally max_steps interaction steps with the environment and returns the trajectory. 
    """
    step_fn = jax.jit(jax.vmap(env_fn.step))
    def simulator(params: hk.Params, state: pgx.State, key: jax.random.PRNGKey) -> Trajectory: 
        def cond_fn(tup: Tuple[pgx.State, jax.random.PRNGKey, int]) -> bool: 
            state, key, step, states, actions, action_distributions, rewards, rngs = tup
            return jnp.logical_and(~(state.terminated).all(), step <= max_steps)
        
        def body_fn(tup: Tuple[pgx.State, jax.random.PRNGKey, int, chex.Array, chex.Array, chex.Array, chex.Array]) -> Tuple[pgx.State, jax.random.PRNGKey, int, chex.Array, chex.Array, chex.Array, chex.Array]: 
            state, key, step, states, actions, action_distributions, rewards, rngs = tup
            logits, values = forward_pass.apply(params, state.observation)
            action_distribution = jax.nn.softmax(logits, axis=-1)
            # Deterministic action choice 
            action = logits.argmax(axis=-1)
            # Sample next state 
            key, _key = jax.random.split(key)
            keys = jax.random.split(_key, state.observation.shape[0])
            rngs = rngs.at[step].set(keys)
            new_state = step_fn(state, action, keys)
            rewards = rewards + state.rewards.squeeze()
            states = jax.tree.map(
                lambda arr, x: arr.at[step].set(x), states, state # Get old state where action is taken
            )
            actions = actions.at[step].set(action)
            action_distributions = jax.tree.map(
                lambda arr, x: arr.at[step].set(x), action_distributions, action_distribution
            )
            return (new_state, key, step + 1, states, actions, action_distributions, rewards, rngs) 
        
        # Setup data to extract
        num_states = state.observation.shape[0]; num_actions = env_fn.num_actions
        actions = -jnp.ones(
            (max_steps, num_states)
        )
        states = jax.tree.map(
            lambda x: jnp.ones((max_steps,) + x.shape, x.dtype), state
        )
        action_distributions = jnp.zeros(
            (max_steps, num_states, num_actions)
        )
        rewards = jnp.zeros(num_states)
        rngs = jnp.zeros((max_steps, num_states, 2))
        step = 0
        # Run loop 
        state, key, step, states, actions, action_distributions, rewards, rngs = jax.lax.while_loop(
            cond_fn, 
            body_fn, 
            (state, key, step, states, actions, action_distributions, rewards, rngs))
        
        trajectory = Trajectory(
            state=states, 
            action=actions, 
            accumulated_rewards=rewards, 
            action_distribution=action_distributions, 
            rng=rngs)
        
        return trajectory
    return simulator

def build_pgx_simulator_with_fixed_randomness(env_fn: Callable, forward_pass: Callable, max_steps: int) -> Callable: 
    """
    Builds a simulator that runs a policy for maximally max_steps interaction steps with the environment and returns the trajectory. 
    """
    def simulator(params: hk.Params, state: pgx.State, key: jax.random.PRNGKey) -> Trajectory: 
        step_fn = jax.jit(jax.vmap(env_fn.step))
        def cond_fn(tup: Tuple[pgx.State, jax.random.PRNGKey, int]) -> bool: 
            state, key, step, actions, action_distributions, rewards, rngs = tup
            return jnp.logical_and(~(state.terminated).all(), step <= max_steps)
        
        def body_fn(tup: Tuple[pgx.State, jax.random.PRNGKey, int]) -> Tuple[pgx.State, jax.random.PRNGKey, int]: 
            state, key, step, actions, action_distributions, rewards, rngs = tup
            logits, values = forward_pass.apply(params, state.observation)
            # Deterministic action choice 
            action = logits.argmax(axis=-1)
            actions = actions.at[step].set(action)
            action_distribution = jax.nn.softmax(logits, axis=-1) 
            action_distributions = action_distributions.at[step].set(action_distribution) 
            rewards = rewards + state.rewards.squeeze() 
            rngs = rngs.at[step].set(jnp.stack([key] * state.observation.shape[0]))
            # Sample next state 
            key, _key = jax.random.split(key)
            keys = jnp.stack([key] * state.observation.shape[0])
            new_state = step_fn(state, action, keys)
            return (new_state, key, step + 1, actions, action_distributions, rewards, rngs) 
        
        # Setup data to extract
        num_states = state.observation.shape[0]; num_actions = env_fn.num_actions
        actions = -jnp.ones(
            (max_steps, num_states)
        )
        states = jax.tree.map(
            lambda x: jnp.ones((max_steps,) + x.shape, x.dtype), state
        )
        action_distributions = jnp.zeros(
            (max_steps, num_states, num_actions)
        )
        rewards = jnp.zeros((num_states))
        rngs = jnp.zeros((max_steps, num_states, 2))
        
        # Run loop 
        state, key, step, actions, action_distributions, rewards, rngs = jax.lax.while_loop(
            cond_fn, 
            body_fn, 
            (state, key, 0, actions, action_distributions, rewards, rngs))
        
        trajectory = Trajectory(
            state=states, 
            action=actions, 
            accumulated_rewards=rewards, 
            action_distribution=action_distributions, 
            rng=rngs)
        
        return trajectory
    return simulator


def build_pgx_simulator_of_test_case(env_fn: Callable, forward_pass: Callable, max_steps: int) -> Callable: 
    """
    Builds a simulator that runs a policy for maximally max_steps interaction steps with the environment and returns the trajectory. 
    """
    def simulator(params: hk.Params, test_case: TestCase) -> Trajectory: 
        step_fn = jax.jit(jax.vmap(env_fn.step))
        def sample_action(logits: chex.Array) -> chex.Array: 
            probabilities = jax.nn.softmax(logits*0.5, axis=-1)
            return jax.random.choice(jax.random.PRNGKey(123), jnp.arange(logits.shape[-1]), p=probabilities)
        
        def cond_fn(tup: Tuple[pgx.State, jax.random.PRNGKey, int, pgx.State, chex.Array, chex.Array, float, chex.Array]) -> bool: 
            state, key, step, states, actions, action_distributions, rewards, rngs = tup
            return jnp.logical_and(~(state.terminated).all(), step <= max_steps)
        
        def body_fn(tup: Tuple[pgx.State, jax.random.PRNGKey, int, pgx.State, chex.Array, chex.Array, float, chex.Array]) -> Tuple[pgx.State, jax.random.PRNGKey, int, pgx.State, chex.Array, chex.Array, float, chex.Array]: 
            state, key, step, states, actions, action_distributions, rewards, rngs = tup
            logits, values = forward_pass.apply(params, state.observation)
            # Deterministic action choice 
            #action = jax.vmap(sample_action)(logits)
            #action = jax.random.choice(jax.random.PRNGKey(123), jnp.arange(logits.shape[-1]), shape=(logits.shape[0],))
            action = logits.argmax(axis=-1)
            rngs = rngs.at[step].set(key) 
            # Using test case key to sample next state 
            keys = jax.vmap(jax.random.split, in_axes=(0))(key) 
            key = keys[:, 0, :]
            _key = keys[:, 1, :]
            new_state = step_fn(state, action, _key)
            # Get trajectory information 
            action_distribution = jax.nn.softmax(logits, axis=-1) 
            action_distributions = action_distributions.at[step].set(action_distribution) 
            rewards = rewards + state.rewards.squeeze() 
            actions = actions.at[step].set(action)
            # Update state 
            states = jax.tree.map(
                lambda arr, x: arr.at[step].set(x), states, state
            )
            return (new_state, key, step + 1, states, actions, action_distributions, rewards, rngs) 
        
        
        # Setup data to extract
        assert test_case.key.shape[-1] == 2 
        assert test_case.key.shape[0] == test_case.state.observation.shape[0]
        num_states = test_case.state.observation.shape[0]; num_actions = env_fn.num_actions
        actions = -jnp.ones(
            (max_steps, num_states)
        )
        states = jax.tree.map(
            lambda x: jnp.ones((max_steps,) + x.shape, x.dtype), test_case.state
        )
        action_distributions = jnp.zeros(
            (max_steps, num_states, num_actions)
        )
        rewards = jnp.zeros((num_states))
        rngs = jnp.zeros((max_steps, num_states, 2))
        # Run loop 
        state, key, step, states, actions, action_distributions, rewards, rngs = jax.lax.while_loop(
            cond_fn, 
            body_fn, 
            (test_case.state, test_case.key, 0, states, actions, action_distributions, rewards, rngs))
        
        trajectory = Trajectory(
            state=states, 
            action=actions, 
            accumulated_rewards=rewards, 
            action_distribution=action_distributions, 
            rng=rngs)
        
        return trajectory
    return simulator