from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import pgx
import pgx.minatar
import pgx.minatar.asterix
import pgx.minatar.breakout
import pgx.minatar.freeway
import pgx.minatar.seaquest
import pgx.minatar.space_invaders
from pgx._src.types import PRNGKey


# Mutation functions
def mutate_states(
    state_batch: pgx.State,
    key: PRNGKey,
    mutators: Dict[str, Callable],
    observe_fn: Callable,
    mutation_prob: float = 0.1,
    mutation_strength: float = 1.0,
) -> Tuple[pgx.State, float]:
    batch_size = len(jax.tree_util.tree_leaves(state_batch)[0])

    keys = jax.random.split(key, 3)

    # Determine which fields to mutate
    fields_to_mutate = jax.random.bernoulli(
        keys[0], mutation_prob, shape=(len(mutators), batch_size)
    )
    # Apply mutations
    new_states = state_batch
    # mutations = {}
    changes = jnp.zeros((batch_size,))
    for i, (field, mutate_func) in enumerate(mutators.items()):
        old_values = getattr(new_states, field)
        new_values = jax.vmap(
            lambda d, s, k, old_v: jax.lax.select(
                d, mutate_func(s, k, strength=mutation_strength), old_v
            )
        )(
            fields_to_mutate[i],
            new_states,
            jax.random.split(keys[1], batch_size),
            old_values,
        )

        changes += jax.vmap(lambda new_v, old_v: jnp.sum(jnp.abs(new_v.astype(jnp.float32) - old_v.astype(jnp.float32))))(new_values, old_values)
        new_states = new_states.replace(**{field: new_values})

    # Update the observation
    new_observation = jax.vmap(observe_fn)(new_states)
    new_states = new_states.replace(observation=new_observation)

    return new_states, changes  # mutations


def mutate_state(
    state: pgx.State,
    key: jax.random.PRNGKey,
    mutators: Dict[str, Callable],
    observe_fn: Callable,
    mutation_prob: float = 0.1,
    mutation_strength: float = 1.0,
) -> Tuple[pgx.State, Dict[str, Any]]:
    key_select, key_mutate = jax.random.split(key)

    # Determine which fields to mutate
    fields_to_mutate = jax.random.bernoulli(
        key_select, mutation_prob, shape=(len(mutators),)
    )

    # Create lists of field names and mutator functions
    field_names = list(mutators.keys())
    mutator_funcs = list(mutators.values())

    # Create mutation keys for each field
    mutation_keys = jax.random.split(key_mutate, len(mutators))

    def mutate_single_field(i, carry):
        current_state, mutations_dict = carry
        field_name = field_names[i]
        mutate_func = mutator_funcs[i]
        do_mutate = fields_to_mutate[i]
        mutation_key = mutation_keys[i]

        current_value = getattr(current_state, field_name)

        def apply_mutation(args):
            value, key = args
            return mutate_func(value=value, key=key, strength=mutation_strength)

        new_value = jax.lax.cond(
            do_mutate,
            apply_mutation,
            lambda args: args[0],
            (current_value, mutation_key),
        )

        new_state = current_state.replace(**{field_name: new_value})
        new_mutations_dict = {**mutations_dict, field_name: new_value}

        return new_state, new_mutations_dict

    # Use fori_loop to apply mutations across all fields
    final_state, mutations_dict = jax.lax.fori_loop(
        0, len(field_names), mutate_single_field, (state, {})
    )

    # Update the observation for the new state
    new_observation = observe_fn(final_state)
    final_state = final_state.replace(observation=new_observation)

    return final_state, mutations_dict


def _mutate_int(value, key, strength, low, high):
    """
    Mutates an integer value by a random amount within a range.
    Returns the new value and the difference between the new and old value.
    """
    
    range_size = high - low
    delta = jax.random.randint(key, (), -range_size, range_size + 1, dtype=jnp.int32)
    new_value = jnp.clip(value + jnp.floor(delta * strength).astype(jnp.int32), low, high - 1)
    return new_value


def _mutate_bool(value, key, strength):
    """
    Mutates a boolean value by a random amount within a range.
    Returns the new value and the difference between the new and old value.
    """
    mutate = jax.random.bernoulli(key, p=strength)
    return mutate


def _mutate_float(value, key, low, high, strength):
    """
    Mutates a float value by a random amount within a range.
    Returns the new value and the difference between the new and old value.
    """
    new_value = jnp.clip(value + jax.random.normal(key) * strength * (high - low), low, high)
    return new_value#, jnp.abs(new_value - value)


# Asterix-specific
def asterix_mutate_entities(value, key, strength):
    keys = jax.random.split(key, 3)

    # Determine number of entities to modify using Zipf-like distribution
    num_entities = 8  # value.shape[0]
    zipf_probs = 1 / jnp.arange(1, num_entities + 1)
    zipf_probs /= zipf_probs.sum()
    num_to_modify = jax.random.choice(
        keys[0], jnp.arange(1, num_entities + 1), p=zipf_probs
    )

    mod_mask = jax.random.bernoulli(
        keys[1], p=num_to_modify / num_entities, shape=(num_entities,)
    )

    # Perform the mutation on selected entities
    new_entities = jax.vmap(
        lambda m, e, k: jax.lax.cond(
            m,
            lambda: asterix_mutate_single_entity(e, k, strength),
            lambda: e,
        )
    )(mod_mask, value, jax.random.split(keys[2], num_entities))

    return new_entities


def asterix_mutate_single_entity(entity, key, strength):
    """
    Mutates a single entity and returns the new entity and the difference between the new and old entity.
    """
    keys = jax.random.split(key, 4)
    x = _mutate_int(entity[0], keys[0], strength, low=0, high=11)  # 0-9 or INF (10)
    y = _mutate_int(entity[1], keys[1], strength, low=0, high=11)  # 0-9 or INF (10)
    direction = _mutate_bool(entity[2], keys[2], strength)
    is_gold = _mutate_bool(entity[3], keys[3], strength)
    return jnp.array([x, y, direction, is_gold])


ASTERIX_MUTATORS: Dict[str, Callable] = {
    "_player_x": lambda s, k, strength: _mutate_int(
        s._player_x, key=k, strength=strength, low=0, high=10
    ),
    "_player_y": lambda s, k, strength: _mutate_int(
        s._player_y, key=k, strength=strength, low=0, high=10
    ),
    "_entities": lambda s, k, strength: asterix_mutate_entities(
        s._entities, key=k, strength=strength
    ),
    "_spawn_speed": lambda s, k, strength: _mutate_int(
        s._spawn_speed, key=k, strength=strength, low=1, high=11
    ),
    "_spawn_timer": lambda s, k, strength: _mutate_int(
        s._spawn_timer, k, strength, low=0, high=s._spawn_speed + 1
    ),
    "_move_speed": lambda s, k, strength: _mutate_int(
        s._move_speed, key=k, strength=strength, low=1, high=6
    ),
    "_move_timer": lambda s, k, strength: _mutate_int(
        s._move_timer, k, strength, low=0, high=s._move_speed + 1
    ),
    "_ramp_timer": lambda s, k, strength: _mutate_int(
        s._ramp_timer, key=k, strength=strength, low=0, high=101
    ),
    # "_ramp_index": lambda s, k, strength: _mutate_int(
    #     s._ramp_index, key=k, strength=strength, low=0, high=1000
    # ),
    # "_terminal": lambda s, k, strength: _mutate_bool(
    #     s._terminal, key=k, strength=strength
    # ),
    # "_last_action": lambda s, k, strength: _mutate_int(
    #     s._last_action, key=k, strength=strength, low=0, high=6
    # ),
}


# Breakout-specific
def breakout_mutate_map(value, key, strength):
    # Flip some bricks randomly
    mask = jax.random.bernoulli(key, p=strength, shape=value[1:4, :].shape)
    new_value = jnp.zeros_like(value).at[1:4, :].set(mask)
    return jnp.logical_xor(value, new_value)


BREAKOUT_MUTATORS: Dict[str, Callable] = {
    "_ball_y": lambda s, k, strength: _mutate_int(
        value=s._ball_y,
        key=k,
        strength=strength,
        low=0,
        high=8,  # exclude 8+9, because it terminates
    ),
    "_ball_x": lambda s, k, strength: _mutate_int(
        value=s._ball_x, key=k, strength=strength, low=0, high=10
    ),
    "_ball_dir": lambda s, k, strength: _mutate_int(
        value=s._ball_dir, key=k, strength=strength, low=0, high=4
    ),
    "_pos": lambda s, k, strength: _mutate_int(
        value=s._pos, key=k, strength=strength, low=0, high=10
    ),
    "_brick_map": lambda s, k, strength: breakout_mutate_map(
        value=s._brick_map, key=k, strength=strength
    ),
    "_strike": lambda s, k, strength: _mutate_bool(
        value=s._strike, key=k, strength=strength
    ),
    "_last_x": lambda s, k, strength: _mutate_int(
        value=s._last_x, key=k, strength=strength, low=0, high=10
    ),
    "_last_y": lambda s, k, strength: _mutate_int(
        value=s._last_y, key=k, strength=strength, low=0, high=10
    ),
    # "_terminal": lambda s, k, strength:  _mutate_bool(value=s._terminal, key=k, strength=strength),
    # "_last_action": lambda s, k, strength: _mutate_int(value=s._last_action, key=k, strength=strength, low=0, high=4),
    # "rewards": partial(_mutate_float, low=0, high=1),  # Assuming rewards are normalized
    # "terminated": _mutate_bool,
    # "truncated": _mutate_bool,
    # "_step_count": lambda s, k, strength: _mutate_int(
    #     value=s._step_count, key=k, strength=strength, low=0, high=1000
    # ),  # Assuming a reasonable max step count
}


# Freeway-specific
def freeway_mutate_cars(value, key, strength):
    keys = jax.random.split(key, 4)

    # Mutate car positions (x, y)
    new_positions = jax.vmap(
        lambda v, k: jnp.array(
            [
                _mutate_int(v[0], k, strength, low=0, high=10),
                _mutate_int(v[1], k, strength, low=1, high=9),
            ]
        )
    )(value._cars[:, :2], jax.random.split(keys[0], 8))

    # Mutate car timers
    new_timers = jax.vmap(lambda v, k: _mutate_int(v, k, strength, low=0, high=6))(
        value._cars[:, 2], jax.random.split(keys[1], 8)
    )

    # Mutate car speeds (direction)
    new_speeds = jax.vmap(lambda v, k: _mutate_int(v, k, strength, low=-5, high=6))(
        value._cars[:, 3], jax.random.split(keys[2], 8)
    )

    return jnp.column_stack([new_positions, new_timers, new_speeds])


FREEWAY_MUTATORS: Dict[str, Callable] = {
    "_cars": freeway_mutate_cars,
    "_pos": lambda s, k, strength: _mutate_int(
        s._pos, key=k, strength=strength, low=0, high=10
    ),
    "_move_timer": lambda s, k, strength: _mutate_int(
        s._move_timer, key=k, strength=strength, low=0, high=4
    ),  # player_speed + 1
    "_terminate_timer": lambda s, k, strength: _mutate_int(
        s._terminate_timer, key=k, strength=strength, low=0, high=2501
    ),  # time_limit + 1
    # "_terminal": lambda s, k, strength: _mutate_bool(
    #     s._terminal, key=k, strength=strength
    # ),
    # "_last_action": lambda s, k, strength: _mutate_int(
    #     s._last_action, key=k, strength=strength, low=0, high=5
    # ),  # 0, 2, 4 are valid actions
    # "_step_count": lambda s, k, strength: _mutate_int(
    #     s._step_count, key=k, strength=strength, low=0, high=2501
    # ),  # Assuming max steps is time_limit
}


# Seaquest-specific
def seaquest_mutate_entities(value, key, strength, entity_size):
    keys = jax.random.split(key, 3)
    num_entities = value.shape[0]

    # Zipf distribution
    zipf_probs = 1 / jnp.arange(1, num_entities + 1)
    zipf_probs /= zipf_probs.sum()

    # Choose number of entities to modify
    num_to_modify = jax.random.choice(
        keys[0], jnp.arange(1, num_entities + 1), p=zipf_probs, shape=()
    )

    # Create a random mask for entities to modify
    entity_mask = jax.random.uniform(keys[1], (num_entities,)) < (
        num_to_modify / num_entities
    )

    # Define mutation function
    def mutate_entity(entity, subkey):
        # Choose which dimension to mutate
        dim_to_mutate = jax.random.randint(subkey, (), 0, entity_size)

        def mutate_dim(i, e):
            if i == 0 or i == 1:  # x and y positions
                return _mutate_int(e, subkey, strength, low=-1, high=10)
            elif i == 2:  # direction (enemy_lr or sub_or)
                return _mutate_bool(e, subkey, strength).astype(jnp.int32)
            elif i == 3:  # move_speed/interval
                return _mutate_int(e, subkey, strength, low=1, high=6)
            elif i == 4:  # ENEMY_SHOT_INTERVAL (only for _e_subs)
                return _mutate_int(e, subkey, strength, low=1, high=60)
            else:
                return e

        new_value = jax.lax.switch(
            dim_to_mutate,
            [lambda _: mutate_dim(i, entity[i]) for i in range(entity_size)],
            None,
        )

        return entity.at[dim_to_mutate].set(new_value)

    # Apply mutations
    new_entities = jax.vmap(
        lambda m, e, k: jax.lax.cond(
            m, lambda args: mutate_entity(*args), lambda args: args[0], (e, k)
        )
    )(entity_mask, value, jax.random.split(keys[2], num_entities))

    return new_entities


SEAQUEST_MUTATORS: Dict[str, Callable] = {
    "_oxygen": lambda s, k, strength: _mutate_int(
        s._oxygen, key=k, strength=strength, low=0, high=200
    ),  # MAX_OXYGEN = 200
    "_diver_count": lambda s, k, strength: _mutate_int(
        s._diver_count, key=k, strength=strength, low=0, high=6
    ),
    "_sub_x": lambda s, k, strength: _mutate_int(
        s._sub_x, key=k, strength=strength, low=0, high=9
    ),
    "_sub_y": lambda s, k, strength: _mutate_int(
        s._sub_y, key=k, strength=strength, low=0, high=9
    ),
    "_sub_or": lambda s, k, strength: _mutate_bool(s._sub_or, key=k, strength=strength),
    "_f_bullets": lambda s, k, strength: seaquest_mutate_entities(
        s._f_bullets, key=k, strength=strength, entity_size=3
    ),
    "_e_bullets": lambda s, k, strength: seaquest_mutate_entities(
        s._e_bullets, key=k, strength=strength, entity_size=3
    ),
    "_e_fish": lambda s, k, strength: seaquest_mutate_entities(
        s._e_fish, key=k, strength=strength, entity_size=4
    ),
    "_e_subs": lambda s, k, strength: seaquest_mutate_entities(
        s._e_subs, key=k, strength=strength, entity_size=5
    ),
    "_divers": lambda s, k, strength: seaquest_mutate_entities(
        s._divers, key=k, strength=strength, entity_size=4
    ),
    "_e_spawn_speed": lambda s, k, strength: _mutate_int(
        s._e_spawn_speed, k, strength, low=1, high=20
    ),  # INIT_SPAWN_SPEED = 20
    "_e_spawn_timer": lambda s, k, strength: _mutate_int(
        s._e_spawn_timer, k, strength, low=0, high=s._e_spawn_speed + 1
    ),
    "_d_spawn_timer": lambda s, k, strength: _mutate_int(
        s._d_spawn_timer, k, strength, low=0, high=31
    ),  # DIVER_SPAWN_SPEED = 30
    "_move_speed": lambda s, k, strength: _mutate_int(
        s._move_speed, k, strength, low=1, high=6
    ),  # INIT_MOVE_INTERVAL = 5
    # "_ramp_index": lambda s, k, strength: _mutate_int(
    #     s._ramp_index, k, strength, low=0, high=1000
    # ),
    "_shot_timer": lambda s, k, strength: _mutate_int(
        s._shot_timer, k, strength, low=0, high=6
    ),  # SHOT_COOL_DOWN = 5
    # "_surface": lambda s, k, strength: _mutate_bool(
    #     s._surface, key=k, strength=strength
    # ),  # This doesn't make much sense to mutate, it checks whether the player was just at the surface
    # "_last_action": lambda s, k, strength: _mutate_int(
    #     s._last_action, k, strength, low=0, high=5
    # ),
}


# Space Invaders-specific
def space_invaders_mutate_map2(value, key, strength):
    mask = jax.random.bernoulli(key, p=strength, shape=value[0:4, 2:8].shape)
    print(mask.sum())
    new_value = jnp.zeros_like(value).at[0:4, 2:8].set(mask)
    return jnp.logical_xor(value, new_value)


def space_invaders_mutate_map(value, key, strength):
    # TODO Make range of the map a parameter first, then mutate in that area
    keys = jax.random.split(key, 3)

    # Generate the initial mask
    mask = jax.random.bernoulli(keys[0], p=strength, shape=value[0:6, :].shape)  # 2:8

    # Ensure at least one mutation
    all_same = jnp.all(jnp.equal(mask, value[0:6, :]))  # 2:8
    random_position = jax.random.randint(
        keys[1], shape=(2,), minval=0, maxval=jnp.array(mask.shape)
    )

    def flip_random_bit(m):
        return m.at[tuple(random_position)].set(
            jnp.logical_not(m[tuple(random_position)])
        )

    mask = jax.lax.cond(all_same, flip_random_bit, lambda m: m, mask)

    # Apply the mask
    new_value = value.at[0:6, :].set(jnp.logical_xor(value[0:6, :], mask))

    return new_value


SPACE_INVADERS_MUTATORS: Dict[str, Callable] = {
    "_pos": lambda s, k, strength: _mutate_int(
        s._pos, key=k, strength=strength, low=0, high=10
    ),
    "_f_bullet_map": lambda s, k, strength: space_invaders_mutate_map(
        s._f_bullet_map, key=k, strength=strength
    ),
    "_e_bullet_map": lambda s, k, strength: space_invaders_mutate_map(
        s._e_bullet_map, key=k, strength=strength
    ),
    "_alien_map": lambda s, k, strength: space_invaders_mutate_map(
        s._alien_map, key=k, strength=strength
    ),
    "_alien_dir": lambda s, k, strength: _mutate_int(
        s._alien_dir, key=k, strength=strength, low=-1, high=2
    ),
    "_enemy_move_interval": lambda s, k, strength: _mutate_int(
        s._enemy_move_interval, key=k, strength=strength, low=1, high=24
    ),  # This is the speed of the aliens, lower=faster
    "_alien_move_timer": lambda s, k, strength: _mutate_int(
        s._alien_move_timer, k, strength, low=0, high=s._enemy_move_interval + 1
    ),
    "_alien_shot_timer": lambda s, k, strength: _mutate_int(
        s._alien_shot_timer, key=k, strength=strength, low=0, high=11
    ),
    # "_ramp_index": partial(_mutate_int, low=0, high=1000),
    "_shot_timer": lambda s, k, strength: _mutate_int(
        s._shot_timer, key=k, strength=strength, low=0, high=6
    ),
    # "_terminal": mutate_bool,  # Commented out as in the Asterix example
    # "_last_action": lambda s, k, strength: _mutate_int(
    #     s._last_action, key=k, strength=strength, low=0, high=6
    # ),
}

# Aggregation
OBSERVE_FN = {
    "minatar-asterix": pgx.minatar.asterix._observe,
    "minatar-breakout": pgx.minatar.breakout._observe,
    "minatar-freeway": pgx.minatar.freeway._observe,
    "minatar-seaquest": pgx.minatar.seaquest._observe,
    "minatar-space_invaders": pgx.minatar.space_invaders._observe,
}

MUTATORS = {
    "minatar-asterix": ASTERIX_MUTATORS,
    "minatar-breakout": BREAKOUT_MUTATORS,
    "minatar-freeway": FREEWAY_MUTATORS,
    "minatar-seaquest": SEAQUEST_MUTATORS,
    "minatar-space_invaders": SPACE_INVADERS_MUTATORS,
}


# `vmap` because `reference_state` is batched
# jax.vmap(mutate_state, in_axes=(0, None, None, None))(
#     reference_state, rng, MUTATORS[env_name], OBSERVE_FN[env_name]
# )
