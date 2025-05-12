from dataclasses import fields 
import pgx
import jax  
import jax.numpy as jnp 
from typing import Dict, List 

ENVIRONMENTS = (
    "minatar-asterix",
    "minatar-breakout",
    "minatar-freeway",
    "minatar-seaquest",
    "minatar-space_invaders",
)


FIELDS_TO_EXPORT: Dict[str, List[str]] = {
    "minatar-asterix": ["_player_x", "_player_y", "_entities", "_shot_timer", "_spawn_speed", "_spawn_timer", "_move_speed", "_move_timer", "_ramp_timer", "_ramp_index"], 
    "minatar-breakout": ["_ball_x", "_ball_y", "_ball_dir", "_pos", "_brick_map", "_strike", "_last_x", "_last_y"], 
    "minatar-seaquest": ["_oxygen", "_diver_count", "_sub_x", "_sub_y", "_sub_or", "_f_bullets", "_e_bullets", "_e_fish", "_e_subs", "_divers", "_e_spawn_speed", "_e_spawn_timer", "_d_spawn_timer", "_move_speed", "_ramp_index", "_shot_timer", "_surface"], 
    "minatar-space_invaders": ["_pos", "_f_bullet_map", "_e_bullet_map", "_alien_map", "_alien_dir", "_enemy_move_interval", "_alien_move_timer", "_alien_shot_timer", "_ramp_index", "_shot_timer"]
}


def state2vec(state: pgx.State) -> jnp.ndarray:
    """
    This function serializes the state object into a vector.
    """
    # Flatten and concatenate all the arrays in the state object
    vec = jnp.concatenate([jnp.ravel(getattr(state, f.name)) for f in fields(state)])

    return vec

def state2vec_filtered(state: pgx.State, fields_to_export: list[str]) -> jnp.ndarray:
    """
    This function serializes the state object into a vector.
    """
    # Flatten and concatenate all the arrays in the state object
    vec = jnp.concatenate([jnp.ravel(getattr(state, f.name)) for f in fields(state) if f.name in fields_to_export])

    return vec

def vec2state(vec: jnp.ndarray, reference: pgx.State) -> pgx.State:
    """
    This function deserializes a vector back into a state object.
    """
    # Start index for slicing the vector
    start = 0

    # Collect the deserialized fields in a dictionary
    state_dict = {}

    # Assign each slice of the vector to the corresponding field in the state object
    for f in fields(reference):
        ref_attr = getattr(reference, f.name)
        end = start + jnp.prod(jnp.array(ref_attr.shape))
        state_dict[f.name] = (
            vec[start:end].reshape(ref_attr.shape).astype(ref_attr.dtype)
        )
        start = end

    # Create a new state object with the deserialized fields
    new_state = type(reference)(**state_dict)

    return new_state

def vec2state_and_reshape(vec, reference): 
    state = vec2state(vec, reference)
    state = jax.tree.map(lambda x: x.squeeze(axis=0), state)
    return state


def test_converters():
    rng = jax.random.PRNGKey(1)

    for env_name in ENVIRONMENTS:
        print(env_name)
        env = pgx.make(env_name)
        env_init = jax.jit(jax.vmap(env.init))
        rng, rng_init = jax.random.split(rng)
        # It's very important to make the initial state with vmap,
        # otherwise the batching dimension is missing
        state = env_init(jax.random.split(rng_init, 1))
        state_vec = state2vec(state)
        restored_state = vec2state(state_vec, reference=state)

        # First compare the shapes
        cmp_shape = jax.tree.map(
            lambda r, s: (r.shape == s.shape), restored_state, state
        )
        for k, v in cmp_shape.__dict__.items():
            assert (
                isinstance(v, bool) and v
            ), f"Env.: {env_name}, Field {k}: Shapes don't match"

        # Then compare the types
        cmp_type = jax.tree.map(
            lambda r, s: (r.dtype == s.dtype), restored_state, state
        )
        for k, v in cmp_type.__dict__.items():
            assert (
                isinstance(v, bool) and v
            ), f"Env.: {env_name}, Field {k}: Dtypes don't match"

        # Then compare the contents
        cmp_state = jax.tree.map(lambda r, s: r == s, restored_state, state)

        for k, v in cmp_state.__dict__.items():
            assert v.all(), f"Env.: {env_name}, Field {k} not correctly restored"
            
            
